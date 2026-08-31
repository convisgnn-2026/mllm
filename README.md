# vidbench — MIG 분할 기반 video DiT 서빙 solver

**연구 질문:** 단일 노드에서 SLO를 만족시키면서 GPU 자원을 최소로 쓰려면,
GPU를 몇 장 쓰고 각각을 어떻게 쪼개야 하는가.

```
min  N                                   [물리 GPU 수]
s.t. Capacity(σ) ≥ λ                     도착률을 감당할 것
     ∀ 템플릿 t,  ∃ 워커 w : 실행 가능      메모리 하드 제약
     SAR_c(σ) ≥ α_c   ∀ 요청 종류 c        [Phase 2]
     레이아웃이 MIG 배치 규칙을 만족
```

전체 서사와 근거는 **`~/papers/study/1_research_flow.md`** 에 정리되어 있다.

---

## 폴더 구조

```
vidbench/
├── solver/        ★ 메인 — 모델과 solver
│   ├── model.py       서비스시간·용량·실행가능성 + MIG 레이아웃 카탈로그
│   ├── solver.py      min N s.t. Capacity ≥ λ  (필터 + 완전탐색)
│   └── validate.py    실측 대조 (요청 7,130건 · 구성 39개)
│
├── bench/         실험 harness — TetriServe 규약 이식
│   ├── workload.py         감마 도착(rate, cv) + 요청 크기 분포
│   ├── serve_bench.py      워커풀 + 디스패처, 타임스탬프만 기록
│   ├── score_slo.py        사후 채점 — 한 번 돌리고 SLO 여러 개로 평가
│   ├── profile_stages.py   스테이지별 지연·메모리 프로파일
│   ├── bench_concurrency.py  동일 요청 K개 (직렬 vs 동시)
│   └── bench_mixed.py        혼합 워크로드 + 요청별 마감
│
├── figures/       논문 그림
│   ├── plot_pareto.py     ★ 메인 — 비용 vs SAR 프론티어
│   ├── plot_tail.py       p95 꼬리 지연 (부하 스윕)
│   ├── plot_threshold.py  SLO 임계점
│   └── exploratory/       버려진 시도 (기록용)
│
├── scripts/       실험 실행 셸 스크립트
├── slo_configs/   SLO 정의 — 템플릿별 절대 초, 0.75x ~ 6.0x 스윕
├── results/       모든 실측 데이터와 그림
└── logs/
```

**경로 규칙:** 모든 스크립트는 실행 위치와 무관하게 저장소 루트로 `chdir` 하므로
`results/`, `slo_configs/` 를 상대경로로 그대로 쓴다. 어디서 실행해도 된다.

---

## 사용법

```bash
PY=~/venv/jinwoo/bin/python

# ── solver ─────────────────────────────────────────────
$PY solver/model.py                              # 레이아웃 카탈로그와 실행가능성
$PY solver/validate.py                           # 모델을 실측과 대조
$PY solver/solver.py --rate 1.65 --mix short --validated-only
$PY solver/solver.py --sweep --mix short --validated-only

# ── 실험 ───────────────────────────────────────────────
$PY bench/workload.py -n 150 --rate 1.0 --cv 1.0 \
    --shape-dist short_seqlen_high -o results/wl.json
$PY bench/serve_bench.py --workload results/wl.json --policy 2F \
    --devices GPU-<uuid>,GPU-<uuid> --n-gpus 2 -o results/run.json
$PY bench/score_slo.py "results/run.json" --sweep

# ── 그림 ───────────────────────────────────────────────
$PY figures/plot_pareto.py
$PY figures/plot_tail.py
```

---

## 검증 상태

| 대상 | 결과 | 비교 |
|---|---|---|
| 서비스 시간 `t = t_full·(148(1−B)/S + B)`, B=0.184 | **MAPE 2.0%** | LLM-Pilot 6% |
| 처리 용량 `Σ 1/E[service]` | **MAPE 2.2%** (구성 39개) | — |

**모델이 틀리는 두 곳** (체계적 편향, 방향은 보수적):
- 70 SM × 작은 요청을 **과대예측** (+13%) → 2분할 용량을 과소예측 (−6%)
- **이종 풀**을 과대예측 (+11%) — 용량식이 "모든 워커가 같은 요청 구성을 처리한다"고
  가정하는데, 크기 맞춤 배정에서는 거짓. Phase 2 시뮬레이터로 해결 예정

`solver.py --validated-only` 는 실제로 트래픽을 돌려본 레이아웃
(Full · 2분할 · 7분할) 으로만 답을 제한한다. 그 밖은 외삽이며 `⚠` 로 표시된다.

---

## 하드웨어 주의

**공유 서버다.** GPU 0·2·4·7 에는 다른 사용자의 tritonserver 가 상주하므로
**1·3·5·6 만 사용한다.** 실행 스크립트에 보호 목록이 하드코딩되어 있다.

`CUDA_VISIBLE_DEVICES` 에 **숫자 인덱스를 쓰지 말 것.** 노드의 어느 GPU라도
MIG 모드이면 CUDA 의 디바이스 열거가 밀려서 엉뚱한 GPU를 잡는다
(실제로 이 때문에 GPU 4·7 위에서 작업이 돌아간 적이 있다). **GPU UUID 를 쓴다.**

```bash
nvidia-smi --query-gpu=uuid --format=csv,noheader --id=1     # GPU-xxxx...
nvidia-smi -L | grep -oP 'MIG-[0-9a-f-]+'                    # MIG 조각 UUID
```

MIG 원복:
```bash
sudo nvidia-smi mig -i <N> -dci && sudo nvidia-smi mig -i <N> -dgi \
  && sudo nvidia-smi -i <N> -mig 0
```

---

## 다음 (Phase 2)

이산사건 시뮬레이터로 **배정까지 모사**하여
(a) 이종 풀의 용량 예측을 고치고, (b) SAR 제약을 solver 에 추가한다.
