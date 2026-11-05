# mllm — 고정 이종 GPU 풀의 멀티모달(LLM + DiT) 추론 서빙 연구

LLM-Pilot(CCGrid'26)을 온프레미스 이종 GPU 풀로 확장한다. 한 GPU를 여러 서비스가 나눌 때 생기는 **프로세스 간 간섭**을
측정·모델링하고, 공유 방식(시간 분할·MPS·MIG)·분할 기법(PD)·카드 배치를 수요와 SLO에 맞춰 고르는 **solver**의 필요성을 보인다.

## 지금 어디까지 왔나 (2026-11-05)

| 단계 | 결과 | 기록 |
|---|---|---|
| 4090+3090 (Stage 0–2b) | 공유 배치의 간섭 비율 k는 수요에 따라 커지고, 앵커 2개 선형으로 ±8% 예측. 분리 배치는 k=1 | `docs/solver_need_experiments.md` 앞부분 |
| Nsight 진단 | 간섭의 원인은 CUDA 컨텍스트 시간 분할(~2.3 ms). 커널은 안 느려지고 상대 차례만큼 기다림 (R6) | 같은 문서 "Nsight 진단" |
| B300 + RTX PRO 6000 포팅 | R6 재현. B300은 스텝이 슬라이스보다 짧아 "간격형". MPS는 경합(1.23×), MIG는 간섭 0(용량 대가). 같은 노드 NVLink 12 GB/s, 노드 간 25 GbE TCP 1.45 GB/s | 같은 문서 "B300 포팅" 절들, `docs/b300_playbook.md` |
| 다음 | 30B급 LLM + SD3.5-large로 재측정, MPS 부하 의존성, 결정 지도, 시뮬 외삽 | `docs/solver_need_experiments.md` 끝의 제안 |

**처음 보는 사람은** `docs/b300_playbook.md`(환경·명령·결과 위치) → `docs/handoff_b300.md`(배경과 체크리스트) → `docs/solver_need_experiments.md`(실험 기록 전체) 순서로 읽는다.

## 저장소 구조

```
bench/hetero/      실측 하네스·서버·분석·그림 코드 (목록과 역할: bench/hetero/README.md)
results/hetero/    실험 결과. 폴더 하나 = 실험 하나. 요약·그림·로그만 git에 있고 대용량 원본(.nsys-rep)은 노드에만
docs/              실험 기록(solver_need_experiments.md), 프로토콜(stage1_protocol.md), 포팅(handoff_b300.md, b300_playbook.md)
docs/notes/        미팅·메일·리뷰 기록, 이전 정리 이력
env/               venv 패키지 고정 목록
papers/            참고 논문(refs/)과 외부 코드(code/tetriserve)
presentations/     발표 자료
archive/legacy/    이전 MIG·시뮬레이터 연구 (코드·결과 보존, 현재 실험과 무관)
```

## 실행 요약

모두 저장소 루트에서, `~/venv/mllm`(`env/requirements-b300-freeze.txt`)로 실행한다. GPU는 UUID로 지정하고, 다른 사용자의 GPU는 건드리지 않는다.

| 하고 싶은 것 | 명령 |
|---|---|
| 공유 GPU 간섭 캡처(Nsight) | `PLACE=S4_B300 bench/hetero/profile_mixed_nsys.sh mixed\|llm\|dit` |
| 캡처 분석 | `STEP_MIN_MS=1 python3 bench/hetero/analyze_nsys_steps.py <dir>/mixed <dir>/llm <dir>/dit` |
| KV 전송 대역 (같은 노드 / 노드 간) | `bench/hetero/pd_xfer_probe_b300.sh <tag>` / `bench/hetero/pd_xfer_probe_mn.sh <tag>` |
| Stage 1 실험 1회 | `bench/hetero/run_mixed_modal.sh` (단일 노드), `run_mixed_modal_mn.sh` (멀티노드) |
| Stage 1 전체 | `python bench/hetero/run_stage1_pilot.py prepare \| dry-run \| run` |
| 채점·보고서 | `bench/hetero/stage1_report.py`, `stage2_report.py`, `stage2b_report.py` |

자세한 옵션과 MPS/MIG 설정은 `docs/b300_playbook.md`.
