# B300 / RTX PRO 6000 포팅 인수인계 (2026-11-02)

이 문서는 4090+3090 풀에서 끝낸 Stage 0~2b를 B300 8장(main 노드) + RTX PRO 6000 노드 환경으로 옮길 때, 그쪽 Claude Code와 사람이 먼저 읽을 문서다. 근거 문서는 `docs/solver_need_experiments.md`(실험 기록 전체)와 `docs/stage1_protocol.md`(프로토콜)이고, 여기에는 **무엇이 환경에 묶여 있어 다시 정해야 하는지**만 적는다.

## 1. 지금까지의 결론 (4090+3090, 옮겨 가는 주장)

| 번호 | 결론 | 근거 위치 |
|---|---|---|
| R1 | 서비스를 다른 카드에 두면 실행 간섭이 없다(혼합/단독 p90 비율 0.99~1.07). | Stage 2, `results/hetero/stage2_report` |
| R2 | 카드를 공유하면 간섭 비율 k가 수요 α에 따라 커진다. 앵커 α {0.5, 1.0}의 k로 그은 직선이 앵커 상한의 약 1.25배까지는 ±8%(DiT +25%, 보수적)로 맞고, 포화 무릎 뒤에서는 어떤 2점 모델도 맞지 않는다. | Stage 2b, `results/hetero/stage2b_report` |
| R3 | 단독 프로파일만으로 배치를 고르면 SLO 위반 62%. 공유 배치는 혼합 앵커 2회 + 단독 프로파일, 분리 배치는 단독 프로파일만 써도, 처음 보는 α·도착열에서 위반 9%·과지불 3%·놓침 20%. 고정 배치 중 최선은 위반 19%·과지불 10%. | Stage 2b |
| R4 | 결정 지도의 승자 구조(TTFT 빡빡 → 서비스 분리, 느슨 → PD 분할, 매우 느슨 → 한 장)는 seed 3개에서 유지되고 경계 α는 격자 한 칸 흔들린다. | Stage 1 seed 검증, `results/hetero/stage1_validation` |
| R5 | KV 캐시 크기는 결과에 영향이 없다(PD의 α 1.0 통과는 배치 효과). | `results/hetero/stage1_kv2g`, `stage1_kv8g` |

**B300에서의 질문은 R1~R3이 풀이 바뀌어도 성립하느냐다.** 성립하면 "풀이 바뀌어도 프로파일 비용이 같다"는 주장이 되고, 깨지면 어디서 깨지는지가 결과다.

## 2. 환경에 묶여 있어 다시 정해야 하는 것

| 항목 | 현재 (4090+3090) | 위치 | B300에서 |
|---|---|---|---|
| GPU 식별 | UUID 두 개 하드코딩, cuda:0=4090, cuda:1=3090 | `bench/hetero/run_mixed_modal.sh` 29행 | 노드·장 수에 맞게 재작성. 노드 간은 호스트:포트가 추가된다 |
| 후보 집합 | 7개: S4_4090, S3_3090, S4_3090, S3_4090, C_4090, C_3090, C_split (S=단일 엔진, C=PD 분할, 접미사=DiT 위치) | `run_stage1_pilot.py` CANDIDATES, `cards_of`, `expected_servers` | 3절 참고 |
| 모델 | Llama-3.2-3B-Instruct + SD3.5-medium 512² 28 step | `run_stage1_pilot.py` MODEL/DMODEL, `dit_server.py` | 3B는 B300에서 간섭이 거의 안 보일 것. 70B급(TP 2~4) + FLUX.1-dev/SD3.5-large 1024² 권장. 3B·SD3.5-medium은 하네스 동작 확인용으로만 |
| KV 예산 | 엔진당 4 GiB 고정(`--kv-cache-memory-bytes`) | `run_mixed_modal.sh` KV_BYTES, `run_stage1_pilot.py` KV_BYTES/KV_TOKENS_EXPECTED, `run_stage1_refine.py` BLOCK_BYTES | 고정 규칙은 유지, 절대량과 블록 바이트(모델 의존)는 다시 계산 |
| 수요 기준 λ0 | LLM 2.0 rps, DiT 0.3 rps (α=1) | `run_stage1_pilot.py` LAMBDA0 | 단독 포화점을 재서 다시 정한다. α 격자 {0.5, 1, 1.5, 2} + 세분화는 그대로 |
| 비용비 r | p_3090/p_4090 ∈ {0.5, 0.75, 0.94, 1.0} | `run_stage1_pilot.py` PRICE_MODEL, 분석 스크립트 R_PRICE | p_PRO6000/p_B300. 노드 간 배치는 네트워크 비용을 넣을지 결정 |
| PD 전송 | vLLM NixlConnector, 같은 노드 PCIe, UCX tcp/cuda_copy | `run_mixed_modal.sh` 30~31행, `pd_smoke.sh`, `pd_xfer_probe.sh` | 노드 간이면 IB/RoCE RDMA 필요. `UCX_TLS`·`UCX_NET_DEVICES` 재설정 |
| 파이썬 환경 | `~/venv/jinwoo`, vLLM 0.28.0 | — | Blackwell(B300 sm_103, PRO 6000 sm_120)은 CUDA 12.8+ 빌드 필요. NGC vLLM 컨테이너를 인터넷 되는 쪽에서 pull → save → 반입 권장. NIXL/UCX 포함 여부 확인 |
| 트레이스 | 입력/출력 길이 분포 CSV | `run_stage1_pilot.py` TRACE | 그대로 써도 되지만 70B급이면 길이 분포를 다시 볼 것 |

환경에 묶이지 않아 그대로 쓰는 것: `controlled_mixed.py`(플랜 생성·재생), `stage1_report.py`(판정 정의), `stage2_report.py`/`stage2b_report.py`/`plot_*.py`(분석), SLO 격자(TTFT {0.3,0.5,1,2} s × TPOT {12,20,30,50} ms × DiT {3,5,10,20} s × {90,95,99}%), 60/600/300 s 창, α_max의 보간 금지 규칙, 프로토콜 동결(protocol.json 해시) 절차.

## 3. B300 풀의 후보 집합 (초안, Stage 0 뒤 확정)

축이 "카드 2장 중 어디"에서 "노드 내 vs 노드 간"으로 바뀐다.

- 노드 내 (B300만): 한 장에 둘 다 / 두 장에 서비스 분리 / PD 분할(prefill·decode 각 1장) + DiT 별도 장 / PD 분할 + DiT가 prefill 장 공유
- 노드 간: LLM은 B300, DiT는 PRO 6000 / prefill은 B300, decode는 PRO 6000 (+ DiT 위치 2가지)
- 비교 축: 간섭이 있는 배치(공유) vs 없는 배치(분리)가 양쪽에 다 있어야 R1·R2를 검증할 수 있다.
- 노드 간 PD 분할은 Stage 0에서 KV 전송 대역·지연을 재고 성립할 때만 넣는다. IB/RoCE가 없으면 빼고, 노드 간은 서비스 분리만 남긴다.

## 4. Stage 0 체크리스트 (여기서 끝낸 뒤 Stage 1로)

1. 컨테이너 안에서 `vllm serve`(TP 1)와 `dit_server.py`가 B300에서 뜨는지. 3B·SD3.5-medium로 확인.
2. `pd_smoke.sh`를 같은 노드 B300 두 장으로. 그 다음 `pd_xfer_probe.sh`를 노드 간(B300 → PRO 6000)으로. 전송 대역·요청당 지연 기록.
3. 큰 모델로 서비스별 단독 포화점 측정 → λ0 결정. 측정 창에서 DiT 측정 요청이 최소 60건 이상 되도록 λ0·α 하한을 잡는다(99% 목표의 표본 문제).
4. KV 바이트 고정값과 `KV_TOKENS_EXPECTED` 재계산(블록 바이트 = layers × 2 × kv_heads × head_dim × 16 × 2 B).
5. `run_stage1_pilot.py`의 CANDIDATES·cards_of·expected_servers·LAMBDA0·MODEL·DMODEL·PRICE_MODEL을 바꾼 뒤 `prepare`로 protocol.json 동결, `dry-run` 통과.
6. seed 901 Stage 1 → Stage 2(앵커 α 0.5·1.0 solo) → held-out α·seed 검증. 4090+3090 때와 같은 표·그림이 나오게 분석 스크립트는 그대로 쓴다.

## 5. 반입 묶음

- 이 저장소(코드·문서·`results/hetero/stage*`의 protocol/plans/요청 레코드/요약/그림). 서버·프록시 로그, nvidia-smi 샘플, 프로파일러 트레이스는 `.gitignore`로 제외했다(재생성 가능).
- 모델 가중치는 인터넷 되는 쪽에서 새로 받는다. 컨테이너도 같다.
- `papers/refs`에 참고 논문 PDF가 들어 있다(폐쇄망에서 못 받으므로 포함).

## 6. 그쪽 Claude Code에게 주는 첫 지시 (복사용)

> `docs/handoff_b300.md`와 `docs/solver_need_experiments.md`의 "Stage 2b 결과" 절을 읽고, 4절 체크리스트 1~2번만 먼저 진행해 줘. 하네스(`bench/hetero/run_mixed_modal.sh`, `run_stage1_pilot.py`)의 환경 상수는 바꾸되 판정 정의(`stage1_report.py`)와 분석 스크립트는 건드리지 마. GPU를 쓰는 실행은 시작 전에 나에게 확인받고, 결과는 관측과 제안을 구분해서 `docs/solver_need_experiments.md`에 이어서 기록해.
