# B300 / RTX PRO 6000 포팅 인수인계 (2026-11-02, 7절 추가 2026-11-03)

> **2026-11-03 추가:** 7절에 공유 GPU 간섭의 원인(프로세스 간 시간 분할) 확인 결과와, 그에 따라 B300에서 먼저 확인할 항목·환경 의존 항목·미커밋 파일 목록을 덧붙였다. 1~6절은 11-02 상태 그대로이며, 7절과 다른 부분은 7절이 우선한다.

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

---

## 7. 추가 (2026-11-03): 간섭의 원인 확인과 그에 따른 변경

11-02 이후 한 일은 Nsight Systems 캡처 6회(C_3090 3회, S4_4090 3회)와 GPU 없이 한 k 계산 1건이다. 근거는 `docs/solver_need_experiments.md`의 "Nsight 진단", "Nsight 진단 2", "k를 시간 분할 기제에서 계산하기" 세 절이다. 아래는 그중 인수인계에 필요한 것만 적는다.

### 7.1 추가된 결론

| 번호 | 결론 | 상태 | 근거 위치 |
|---|---|---|---|
| R6 | 같은 GPU에 두 프로세스를 두면 CUDA가 두 컨텍스트를 약 2.3 ms 단위로 번갈아 실행한다(기본 시간 분할). 실행 중인 커널 자체는 느려지지 않고(DiT 커널 p50 단독 = 혼합), 스텝의 벽시계 시간만 늘어난다. decode 스텝 2.06~2.14×, 스텝 사이 간격 +4.5 ms, prefill eager gemm 1.9×. 4090(S4_4090)과 3090(C_3090)에서 같다. | **확정(관측)** | `results/hetero/nsys_c3090/`, `results/hetero/nsys_s4_4090/`, 각 `nsys_summary.json`, `timeline_*.png` |
| R6-a | DRAM 대역폭(dmon mem% 47~54), 클럭, 배치 크기(Running 0~5), vLLM 큐(Waiting 0~4, KV ≤ 46%)는 원인이 아니다. 설정 오류가 아니라 compute mode Default·MPS 없음·MIG 없음 상태의 CUDA 기본 동작이다. | **확정(관측)** | 같은 위치 |
| R6-b | 공유의 결과가 "지연 증가"로 끝나는지 "포화"로 가는지는 공유 전 단독 점유율로 갈린다. DiT 단독 점유 0.53(4090)이면 처리량 유지·지연 1.76×, 0.97(3090)이면 처리량 0.68×·큐 증가. | **확정(관측)** | 같은 위치 |
| R7 | 시간 분할 기제를 식으로 옮긴 k 모델(k = 1 + p, p = 내 스텝 동안 상대가 GPU 일을 가진 시간 비율, 적합 매개변수 0개)은 비포화 16점에서 중앙값 9%, 15/16이 ±25% 이내. 포화(부풀린 점유 합 ≥ 1) 30점은 맞지 않으며 "불가"로 처리해야 한다. | **가설(선택 편향 있음)** | `bench/hetero/analyze_k_timeshare.py`, `results/hetero/k_model/k_timeshare.{md,json}`, `k_timeshare_explained.png` |

R7의 신뢰도에 대한 사용자 판단(2026-11-03): **낮다.** 46점을 본 뒤 규칙을 골랐고, seed 902·903 held-out과 선택기 수준 비교를 하지 않았으며, PD 배치의 TTFT 항이 없다. 따라서 **옮겨 가는 주장은 R1~R6이고, R7은 "검증 전 가설"로만 넘긴다.** solver의 k는 여전히 R2의 2-앵커 선형이다. "측정 0회 k"는 주장하지 않는다.

### 7.2 R1~R3의 재해석

- R1(서비스를 다른 카드에 두면 간섭 없음)의 원인이 R6으로 설명된다. 시간 분할은 같은 GPU 위의 컨텍스트 사이에서만 일어난다.
- R2의 k 값은 **"MPS·MIG 없는 기본 시간 분할"에서의 값**이다. 공유 방식이 바뀌면(MPS: 커널 동시 실행, MIG: 하드웨어 분리) k의 모양이 바뀌므로 R2의 수치는 그 환경에 옮기지 않는다.
- P/D 분할의 TPOT 이득은 decode GPU를 다른 프로세스와 나누지 않을 때만 남는다(C_4090 TPOT k 1.00, C_3090 2.8). 단일 엔진의 TPOT 꼬리는 다른 요청의 prefill 뒤에서 decode 스텝이 기다리는 것(엔진 내부, head-of-line)이고, 공유 GPU의 TPOT 증가는 프로세스 간 시간 분할(엔진 외부)이다. 두 가지를 구분해서 설명할 것.

### 7.3 B300에서 Stage 0 전에 먼저 확인할 것 (4절 체크리스트 앞에 추가)

B300과 RTX PRO 6000은 MIG를 지원한다. 따라서 "공유"가 세 가지 중 어느 것인지부터 정해야 R2·R6이 옮겨지는지 말할 수 있다.

0-a. **공유 방식 기록.** `nvidia-smi -q | grep -i "compute mode\|MIG"`, MPS 데몬 유무(`pgrep nvidia-cuda-mps`). 기본 상태(Default, MIG off, MPS 없음)인지 적는다. 이 상태가 아니면 R6의 전제가 다르다.
0-b. **같은 캡처 1쌍.** `bench/hetero/profile_mixed_nsys.sh`를 B300 한 장에 대해 mixed/llm/dit 3회(약 15분). 확인 항목은 세 개: (1) 상대 프로세스가 끼어든 커널의 늘어난 길이 분포에서 슬라이스 길이가 얼마인가(4090/3090은 2.3 ms), (2) decode 스텝이 단독의 약 2배가 되는가, (3) 커널 p50이 단독과 같은가. 셋이 맞으면 R6은 B300에서도 성립하고, 규칙 B(아래)를 그대로 쓴다.
0-c. **MIG·MPS는 도구로 1회씩.** 같은 두 프로세스를 MIG 인스턴스 둘에 나눠 k가 1이 되는지, MPS를 켜면 k가 어떻게 바뀌는지 각 1회. 이것은 간섭을 "해결"하는 실험이 아니라 solver가 "어떤 공유 방식에서 어떤 k를 써야 하는가"를 아는 데 필요한 측정이다.

solver에 들어가는 규칙 세 개와 이식 여부:

| 규칙 | 내용 | B300에서 |
|---|---|---|
| A | 공유하지 않는 단계·지표는 k = 1 | 그대로(R1, 21쌍 0.99~1.07) |
| B | 기본 시간 분할로 공유하면 k는 R2의 2-앵커 선형(또는 검증되면 R7) | 0-b로 확인. MPS면 모양이 바뀌고, MIG면 A로 돌아간다 |
| C | 부풀린 점유 합 ≥ 1이면 포화 → 배치 불가 | 공유 방식과 무관하게 유지 |

### 7.4 환경에 묶인 항목 추가 (2절에 더함)

| 항목 | 현재 | 위치 | B300에서 |
|---|---|---|---|
| Nsight CLI | `~/tools/nsys/.../2026.5.1/.../nsys` (사용자 홈에 수동 설치, root 없음) | `profile_mixed_nsys.sh` 15행 `NSYS=` | Blackwell을 지원하는 버전으로 재설치. GPU 메트릭 샘플링은 `RmProfilingAdminOnly=1`이면 막힌다(여기서도 막혀 dmon으로 대체) |
| 캡처 대상 GPU | UUID 2개 하드코딩, `PLACE` 분기 C_3090/S4_4090 | `profile_mixed_nsys.sh` 12·19행 | 후보 집합이 바뀌므로 `PLACE` 분기를 다시 쓴다. 분석(`analyze_nsys.py`, `plot_nsys_timeline.py`)은 그대로 |
| 슬라이스 상수 | 2.3 ms, 스텝 사이 손실 2슬라이스 | `analyze_k_timeshare.py` | 0-b에서 재측정한 값으로 바꾼다(R7을 쓸 때만 해당) |

### 7.5 반입 묶음과 저장소 상태 (5절 수정)

- **Nsight 산출물은 저장소에 넣지 않는다.** `results/hetero/nsys_c3090/`(1.1 GB)와 `results/hetero/nsys_s4_4090/`(0.9 GB)의 `report.sqlite`·`report.nsys-rep`은 `.gitignore`에 **아직 없다**(11-03 기준). 커밋 전에 `results/hetero/nsys_*/**/report.sqlite`, `results/hetero/nsys_*/**/report.nsys-rep`을 추가하고, `nsys_summary.json`·`timeline_*.png`·`dmon.txt`·콘솔 로그만 남긴다. 캡처는 `profile_mixed_nsys.sh`로 재생성 가능하다.
- **11-03 기준 미커밋 파일.** 코드: `bench/hetero/profile_mixed_nsys.sh`, `analyze_nsys.py`, `plot_nsys_timeline.py`, `analyze_k_model.py`, `analyze_k_timeshare.py`, `plot_k_timeshare_explained.py`, `plot_ieee.py`, `plot_ieee_pd.py`. 문서: `docs/solver_need_experiments.md`(세 절 추가), `docs/1102_mail.md`, `docs/1102_mail_full.md`, `docs/mllm_v0.6.pptx`, 이 문서의 7절. 결과: `results/hetero/k_model/`, `results/hetero/stage2b_report/fig_ieee_*.{pdf,png}`, nsys 두 디렉터리(위 제외 규칙 적용 후).

### 7.6 그쪽 Claude Code에게 주는 첫 지시 (6절을 대신함, 복사용)

> `docs/handoff_b300.md`를 7절까지 읽고, `docs/solver_need_experiments.md`의 "Nsight 진단"·"Nsight 진단 2" 절을 읽어 줘. 먼저 7.3의 0-a(공유 방식 기록)만 하고 결과를 보고해. 0-b 캡처와 4절 체크리스트는 내가 확인한 뒤 하나씩 진행한다. k 모델(R7)은 가설이므로 solver에 넣지 말고, 환경 상수는 바꾸되 판정 정의(`stage1_report.py`)와 분석 스크립트는 건드리지 마. GPU를 쓰는 실행은 시작 전에 나에게 확인받고, 결과는 관측과 제안을 구분해서 `docs/solver_need_experiments.md`에 이어서 기록해.
