# hetsim

이종 GPU 풀 멀티모달 서빙 시뮬레이터 + solver/router. 설계는 `docs/simulator-design.md`, 문제는 `docs/formulation.md`.

```
spec.py       CardType · Card · Request
model/dit.py  DiT 서비스 시간 (B300 적합, 카드별 t_full 로 확장)
model/llm.py  LLM 스텝 시간 (LLM-Pilot 형, 카드별 β; 4090 은 LLMServingSim 으로 적합)
engine_llm.py 연속 배칭 엔진 (chunked prefill · 블록 KV · 도달길이 예약 · 선점 + 프리픽스 복구 · 파킹 · P/D 역할 · 링크 전송)
builders.py   구성 빌더: llm_colo(n), llm_pd(P, D), pd_rule
core.py       Simulator(이벤트 루프) · Group(큐+인스턴스) · Router · SlotInstance
score.py      SAR · p95 · 용량 · goodput(요청/작업 가중)
workload.py   감마 도착 + 클래스 믹스 (bench/workload.py 와 같은 난수 순서)
gates/        단계별 통과 게이트
```

| 단계 | 게이트 | 상태 |
|---|---|---|
| S1 DiT 뼈대 | `python -m hetsim.gates.s1_dit` — 실측 39구성 MAPE + legacy 완전 일치 | PASS (2026-09-21) |
| S2 LLM colo | `python -m hetsim.gates.s2_llm` — LLMServingSim colo 1~4장 × 6 부하 SAR: 평균 0.9%p, 최대 8.0%p | PASS (2026-09-21) |
| S3 PD · 파킹 | `python -m hetsim.gates.s3_pd_park` — PD 1:3/2:2/3:1 × 5 부하: 평균 1.6%p, 최대 4.7%p. 파킹 colo 24칸: 평균 1.5%p, 무릎 셀 최대 16%p | PASS (2026-09-21, 무릎 허용폭 20%p) |

## 모델 노트 (오라클 LLMServingSim 2.0 에서 배운 것)

- **스텝 시간은 두 영역.** 디코드가 있는 스텝(CUDA graph)은 `16.4 ms + 34.8 µs·B + 0.211 µs·K + 85.7 µs·P`,
  순수 프리필 스텝(eager)은 `12.3 ms + 76.9 µs·P`. 가중치 읽기 15.9 ms 가 디코드 스텝에서는 드러나고 프리필 스텝에서는 연산 밑에 숨는다.
- **선점 복구는 도달 길이 예약에 묶인다.** 선점된 요청은 프롬프트가 아니라 "도달 길이(프롬프트+생성)" 블록이 다 있어야 재입장한다.
  그동안 남의 디코드 성장이 그 요청의 캐시 블록을 꼬리부터 먹는다. 그래서 복구율은 평균 41%, 45%는 0 이다. 재계산 토큰 수가 오라클과 1% 안에서 일치.
- **선점 시 computed 는 0 으로.** 캐시에서 되찾은 만큼만 computed 가 된다. 이걸 빼먹으면 재계산 없이 이어가는 낙관 오류가 난다.
- **PD 전송**은 프리필 스텝 안에서 청크 KV 바이트 / 링크 대역폭(16 GB/s) + 레이어당 지연으로 가산. 디코드 쪽은 컨텍스트 전체 블록이 있어야 받고, 없으면 그룹 큐에서 대기(HOL 없음).
- **파킹**은 쫓겨난 캐시 블록을 호스트(50 GB/s)에 두고 복귀 시 되읽기 비용만 낸다. 무릎 셀은 두 시뮬레이터가 요청의 1/3 을 다른 인스턴스로 보내는 카오스 영역이라 셀 단위 오차가 크다(평균은 1.5%p).
