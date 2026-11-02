# 이전 MIG·시뮬레이터 연구

현재 연구의 진입점은 [저장소 README](../../README.md)와 [10/13 미팅](../../docs/1013_meeting.md)이다. 이 폴더의 결과는 이전 연구 이력이며 현재 4090+3090 혼재 실측과 구분한다.

| 경로 | 내용 |
|---|---|
| solver/ | 이전 용량 모델·탐색·시뮬레이터·검증 |
| hetsim/ | 이전 이산사건 시뮬레이터와 gates |
| bench/ | MIG·DiT 실험 harness |
| scripts/ | 이전 장비 실험·시뮬레이터 실행 |
| slo_configs/ | 당시 SLO 탐색 설정 |
| figures/ | 당시 결과 그림 생성 코드 |
| results/ | 위 코드의 원시 데이터·분석 결과 |
| docs/ | 이전 설계·미팅·handoff·연구 이력 |

코드와 데이터의 상대 위치를 함께 보존했다. 여기에서 사용하는 `results/`, `solver/`, `bench/`, `slo_configs/`는 모두 이 폴더를 기준으로 한다.

```bash
cd archive/legacy
python solver/model.py
```

셸 실행 스크립트의 예전 `/home/jinwoo.kim27/vidbench` 진입 경로는 스크립트 기준 경로로 바꿨다. GPU UUID, Python 가상환경, 외부 LLMServingSim 위치는 당시 장비 설정이므로 현재 실행 환경에 맞춰 확인해야 한다. 이 정리 작업에서 GPU 실험은 실행하지 않았다.
