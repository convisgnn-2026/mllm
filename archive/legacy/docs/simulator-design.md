# 새 시뮬레이터 설계 계획 — `hetsim`

> 2026-09-21. `formulation.md`의 문제를 푸는 도구. 기존 `cosolve.py`(시뮬레이터 둘을 CSV 표로 접착)를
> 대체하는 **단일 이산사건 클러스터 시뮬레이터 + 그 위의 solver·router**.

---

## 0. 왜 새로 만드나

| 기존 | 문제 |
|---|---|
| DiT `sim.py` + LLMServingSim 을 표로 접착 | 축(레버·카드·동거·시간)이 늘 때마다 표 칸이 곱으로 증가 |
| LLMServingSim 은 인스턴스 단위 | 클러스터·router·재구성을 표현 못 함, 1회 실행 수 분 |
| 성능 모델이 코드 밖(CSV)에 있음 | 논문 기여인 "모델"이 산출물에 없음 |

새 시뮬은 **모델이 코드 안에 있고**, solver 가 수천 구성을 초 단위로 채점할 만큼 빠르며, router 까지 한 루프에 있다.

---

## 1. 범위

```
입력   hardware.yaml (카드·링크) + tenants.yaml (모델·클래스·SLO) + 워크로드 트레이스
출력   요청별 타임스탬프 → SAR · goodput(요청/작업 가중) · CE · 에너지 · 전환 횟수
```

모델링하는 것: 카드 종류, 인스턴스 역할(E/P/D/PD/DiT-W), 연속 배칭, PD 간 KV 전송, KV 파킹, 동거 몫, 재구성 비가용, router 정책.
모델링하지 않는 것: 커널 수준, 네트워크 토폴로지(링크 대역폭 하나로 추상), KV 스트리밍, AO.

---

## 2. 패키지 구조

```
hetsim/
  spec/        hardware.py  tenants.py  levers.py     # 데이터클래스 + YAML 로더
  workload/    gen.py (감마 도착·클래스 믹스, bench/workload.py 규약)  trace.py (창별 λ)
  model/       llm.py   프리필·디코드 스텝 시간: F/M/D 해석식 × 카드 (β1_k, β0_k)
               vlm.py   인코더(이미지당) + llm.py
               dit.py   t_full_k × (148(1−b)/S + b),  카드별 t_full
               coloc.py 동거 몫 → 스텝 시간 배율
               kv.py    KV 크기, 파킹 적중, 전송 시간
  core/        event.py  card.py  instance.py (역할·배처·큐)  group.py  cluster.py
               router.py (그룹 선택 = solver 규칙, 그룹 내 = 정책)  reconfig.py
  policy/      router_rr.py  router_least.py  router_pred.py
               solver.py (열거 + 필터 + 목적)  switch.py (비용 인지 전환)
  score/       slo.py  goodput.py  cost.py (상각+전력)  report.py
  calib/       fit_beta.py (LLMServingSim·pool CSV 로 β 적합)
  tests/       회귀 게이트 (§5)
  cli.py
```

---

## 3. 핵심 설계 결정

**이벤트 입도.**
- LLM·VLM: **이터레이션 단위.** 디코드 스텝마다 이벤트. 배치 안 토큰 수·KV 총량으로 스텝 시간 계산. TBT·연속 배칭·PD 전송을 표현하려면 필요하다.
- DiT: **요청 단위.** 서비스 시간이 결정적(CV < 0.5%)이므로 시작·끝 이벤트 둘.
- VLM 인코더: 이미지당 고정 시간, 배치 없음(1차).

**스텝 시간 모델 (LLM-Pilot §IV-B 를 카드별로).**
```
T_prefill(B, S_in, k) = β1_k · max(F(B,S_in)/C_k, M/BW_k) + β0_k
T_decode(B, ΣKV, k)   = β1_k · (F_ffn(B)/C_k + M_kv(ΣKV)/BW_k) + β0_k
```
β 는 카드마다 한 번 적합. 4090 은 LLMServingSim 표로, 3090 은 데스크탑 실측으로, B300/PRO 6000 은 저장소 보고값으로.

**레버의 표현.**
| 레버 | 시뮬 안 표현 |
|---|---|
| PD | P 인스턴스 완료 → KV 바이트를 링크 큐에 넣음(대역폭 공유) → D 인스턴스 큐 |
| EPD | E 인스턴스(이미지당) → 임베딩 MB급 전송 → P |
| KV 파킹 | 선점 시 KV 를 호스트로 swap(PCIe 시간), 복귀 시 재계산 대신 swap-in. 프리픽스 적중률은 파라미터 |
| 동거 | 한 카드에 인스턴스 둘, 몫 (s_a, s_b) 로 각자 스텝 시간 ÷ s |
| 이종 | 인스턴스마다 카드 k, 모델이 k 의 (C, BW, M, β) 를 씀 |
| 재구성 | 역할이 바뀐 카드는 R 초 동안 어느 그룹에도 없음 |

**router 는 두 층.** 그룹 선택은 solver 가 낸 규칙 r_t(클래스 → 그룹). 그룹 안은 정책 플러그인(라운드로빈 = 베이스라인, 최소부하, 예측 완료 시각).

**solver 는 시뮬을 호출.** 후보 σ 를 열거 → 메모리·링크 필터 → 창 길이만큼 시뮬 → 목적값. 전환은 `switch.py` 의 비용 인지 규칙. 속도 목표: 4 GPU × 300 s 창 하나를 100 ms 안에.

**재현성.** 시드 고정, 여러 시드 병렬, 결과는 요청별 CSV 로 남기고 채점은 사후(`bench/score_slo.py` 규약).

---

## 4. 단계별 구축 (각 단계에 통과 게이트)

| 단계 | 만드는 것 | 게이트 |
|---|---|---|
| S1 | spec · workload · core 이벤트 루프 · DiT 모델 · 채점 | DiT 단일 테넌트가 `sim.py` 39개 구성의 makespan·SAR 을 1% 안에 재현 |
| S2 | LLM colo 인스턴스(연속 배칭) · β 적합 | `results/servingsim/pool/summary.csv` 의 colo 1~4장 SAR 을 5%p 안에 재현 |
| S3 | PD · 링크 · KV 파킹 | `sweep1/` (P:D × 파킹) 재현. PD 10 req/s 에서 0.79 근방 |
| S4 | VLM(EPD) · 동거 | `pool_vlm*/` 재현, 동거 몫 75/31 입력 시 합산 1.05× |
| S5 | solver 정수 분할 | `cosolve.py` 84칸 실현가능성 표 일치 |
| S6 | 시간축 전환 | `dynamic.py` 4트레이스의 정책 순위 재현(비용 인지 ≥ 반응형 ≥ 정적) |
| S7 | 이종 카드 축 | 우선 비율 가정(4090:3090 연산 2.3, 대역폭 1.08)으로 동작. 데스크탑 실측으로 β 교체 |
| S8 | router 골격 분리 | 시뮬의 router 모듈이 실제 요청 객체를 받는 인터페이스로 회사 배포 |

S1~S6 는 GPU 없이 진행. S7 부터 데스크탑.

---

## 5. 비용·에너지 채점 (`score/cost.py`)

```
c_k = MSRP_k / (수명 × 8760) + P_k × PUE × 전기요금
P_k = 실측 평균 전력(nvidia-smi 적분) 있으면 그것, 없으면 TDP × 이용률
CE  = 작업가중 goodput / Σ_k c_k × 사용시간
```
수명 3·5년, PUE 1.2·1.5 스윕. 전기요금은 사내 값.
관행 근거: Barroso·Hölzle *Datacenter as a Computer* 6장, Hamilton TCO 모델, Splitwise(ISCA'24), DynamoLLM(HPCA'25).

---

## 6. 기존 자산 처분

| 자산 | 처분 |
|---|---|
| LLMServingSim 2.0 | S2·S3 의 β 적합과 검증 오라클. 런타임 의존 없음 |
| `results/servingsim/*`, B300 7,130건 | 게이트 기준값 |
| `solver/sim.py`, `cosolve.py`, `dynamic.py` | 게이트 기준값. S6 통과 후 `legacy/` 로 이동 |
| `bench/` | 유지. S7 데스크탑 실측용, 워크로드 규약 공유 |


SAR/GPU-시간과 SAR/c_k에 대해

방향은 맞고, 차원만 정리하면 됩니다. SAR은 비율이라 GPU-시간으로 나누면 단위가 어색합니다. 같은 뜻을 차원이 맞게 쓰면 "SLO를 지킨 일의 양 ÷ 그 일에 쓴 자원"이고, 분자를 SLO 지킨 요청 수 또는 GPU-초로, 분모를 GPU-시간 또는 c_k로 두면 됩니다. c_k로 나눈 것이 정확히 LLM-Pilot의 CE입니다. 사내 만족도 프로파일링은 여기서 분자의 가중치 자리로 들어갑니다. 지금은 "SLO 안이면 1, 밖이면 0"인 이진 가중치인데, 이걸 "지연에 따른 만족도 곡선 u_c(latency)"로 바꾸면 더 엄밀한 goodput이 됩니다. score.py의 work 인자가 그 자리라서, 나중에 곡선만 끼우면 됩니다. 지금은 도입하지 않습니다.