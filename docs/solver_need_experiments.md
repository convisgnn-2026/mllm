# 구성 선택 필요성 검증 프로토콜

## 연구의 출발점: LLM-PILOT 확장

사용자는 [LLM-PILOT 원 논문](../papers/refs/llm-pilot-ccgrid26.pdf)의 제1저자다. 이번 연구는 실제 멀티모달 AI 추론 플랫폼 기업에서 사용할 solver로 확장하고 같은 학회에 후속 논문을 제출하는 것을 목표로 한다(사용자 명시). 현재 GPU 간섭 실험은 이 확장의 동기·모델·제어 근거를 확보하는 단계다.

원 논문의 핵심은 워크로드·SLO·예산으로 후보 공간을 만들고, 보정된 분석 모델로 성능을 예측하며, SLO를 만족하는 offloading 전략과 VM 조합 중 비용 효율적인 구성을 선택하는 것이다. 원 논문은 shared AO의 큐 대기와 offloading 자원 contention도 다루므로 “기존 연구는 간섭을 전혀 고려하지 않았다”라고 설명하지 않는다.

이번에 검증할 새로운 질문은 **여러 LLM·VLM·DiT 서비스가 자원을 공유할 때, 각 전략을 개별적으로 평가해 고른 구성이 실제 혼합 실행에서도 유효한가? 공유 실행으로 달라지는 성능을 예측하고 배치·실행 허용을 함께 선택하면 더 나은 서비스 구성을 만들 수 있는가?** 이다. 실제 기업 워크로드·운영 SLO·비용 구조는 아직 확보한 것으로 간주하지 않는다.

| 계승하는 구성 | 확장하여 검증할 내용 |
|---|---|
| 워크로드·SLO·예산 입력 | 서비스별 도착률·단계 특성·개별 목표와 공유 자원 제약 |
| 분석 모델 + 실측 보정 | 함께 실행하는 단계 조합과 부하에 따라 달라지는 실행·대기시간 |
| 전략·VM 구성 선택 | 서비스별 전략, 단계 배치 및 검증된 실행 허용 장치의 공동 선택 |
| SLO 만족 조건의 비용 최적화 | 고정 자원에서 서비스별 요구를 얼마나 충족하는지 우선 검증하고, 자원 선택 범위가 있을 때 비용으로 확장 |

동일한 GPU 집합과 실행 시간의 비용은 고정이므로 현재 파일럿의 지연 감소를 곧바로 비용 절감이라 부르지 않는다. 모달리티별 token·image 처리량을 임의로 더하지 않는다. 초기에는 서비스별 성능 벡터·trade-off·요구 충족을 평가하고, 비용 실험에서는 같은 서비스별 수요와 품질·SLO를 만족시키는 데 필요한 자원 비용을 비교한다. 구체적인 SLO와 가치 가중치는 플랫폼 요구로 정한다.

최종 solver 필요성의 핵심 비교는 **동일 후보 공간·동일 자원에서 단독 성능에 기반한 구성 선택**과 **혼합 실행 간섭을 반영한 구성 선택**이다. 작은 후보 공간에서는 실측 최선 구성도 비교해서 예측 오차와 선택 손실을 분리한다. 간섭 관측만으로 solver의 개선을 주장하지 않으며, 고정된 간단한 정책보다 유용한 선택을 한다는 검증이 필요하다. 현재 고정 휴식·상황 제어 실험은 후보 장치의 탐색이고 최종 solver의 기여가 확정된 것은 아니다.


현재 판단 및 근거는 [10/13 미팅](notes/1013_meeting.md), 기존 데이터 분석은 [보고서](../results/hetero/solver_need/summary.md)를 참고한다.

## 재현 명령

```bash
python bench/hetero/analyze_solver_need.py
```

어느 디렉터리에서 실행해도 기본 입력·출력은 스크립트 위치를 기준으로 결정된다. 현재 입력은 code trace의 네 구성·여섯 부하 조건이다. 결과는 탐색적 채점이며 모델 예측이나 온라인 성능 추정이 아니다.

## 첫 GPU 실험: 구성 선택의 상반된 영역 재현

- 우선 부하: LLM 2 rps + DiT 0.3 rps. 현재 모델·정밀도·해상도·steps 유지.
- 후보: C_4090, C_3090, C_split, S4_3090. 단독 기준도 같은 시간·요청으로 측정.
- 요청 도착·크기는 미리 생성하여 고정한다. 모든 후보에 같은 시퀀스를 사용한다.
- 제안 시작값: warmup 60초, 본 관측 600초, 독립 seed 0–4. 파일럿으로 정상상태 도달 여부를 확인하고 필요하면 늘린다. 이는 서비스 기준이 아니다.
- warmup 부하와 큐 상태 정책을 명시한다. steady-state 측정과 cold-start 평가를 섞지 않는다.
- 두 loadgen은 준비가 완료된 뒤 공통 barrier에서 시작한다. 실제 송신·완료 절대 타임스탬프와 계획 도착을 남긴다.
- 관측 중 도착한 요청을 완료까지 추적한다. timeout·오류·거절은 별도 집계하고 attainment 분모에 포함한다. 미완료 censoring 규칙을 기록한다.
- 실제 overlapping 구간 및 interval별 도착률·완료율·backlog를 보고한다. 평균 완료율을 capacity로 단정하지 않는다.
- 실행 순서를 seed별로 섞고 장비 UUID·프로세스·메모리·클럭·환경 버전과 설정 manifest를 저장한다.
- 기존 runner의 광범위한 pkill 대신 이번 실행에서 시작한 PID만 종료하도록 수정한다.

기존 고정 N pilot 경로는 유지한다. 새 CONTROLLED_PLAN 모드에서는 controlled_mixed.py가 두 서비스를 공통 시계로 실행한다. CPU 모의 서버로 경로를 검증했지만 실제 GPU 서버와의 통합 실행은 아직 검증하지 못했다.

## 사후 채점

- 첫 탐색 사례 A: TTFT 0.5초, 평균 TPOT 20ms, DiT 10초.
- 첫 탐색 사례 B: TTFT 1초, 평균 TPOT 12ms, DiT 3초.
- 두 사례를 실제 서비스 요구로 주장하지 않는다. 전체 deadline sweep과 attainment 목표 90/95/99% 민감도를 함께 확인한다.
- 개별 요청 joint attainment, 서비스별 분포, seed별 결과와 불확실성을 보고한다. 연속 요청의 독립성을 가정해 단순 bootstrap으로 작은 오차막대를 만들지 않는다.
- 차이가 반복되지 않으면 경계 사례로 판정한다. threshold 통과 여부 외에 차이의 크기를 보고한다.

## Go / No-go

1. 다른 요구 영역에서 서로 다른 구성의 우위가 재현되면 정적 구성 선택기를 만든다.
2. 한 구성이 넓은 범위에서 우세하면 규칙을 기본선으로 채택하고 복잡한 solver를 보류한다.
3. 정적 필요성이 확인된 뒤 동일 운영 trace 내 수요 변화를 실측한다. 전환 비용까지 포함한 이득이 있을 때 온라인 제어로 확장한다.
4. placement-only, control-only, joint를 비교해 공동 제어의 추가 이득을 확인한다.
5. 후보가 증가할 때 전수 탐색과 lookup의 profiling·실행 비용을 측정하고 더 복잡한 알고리즘의 필요성을 판단한다.

## 10/13 구현·검증 상태

새 [controlled_mixed.py](../bench/hetero/controlled_mixed.py)는 길이 trace와 seed로 유한 시간의 요청 계획을 저장하고, 모든 구성에서 동일 계획을 재생한다. 프롬프트는 replay 시작 전에 로컬 tokenizer로 결정적으로 생성하며 실제 프롬프트 hash를 기록한다. 단독 LLM과 혼재 실행의 LLM 시퀀스도 동일하다. 실제 Azure 문장 재생이 아니라 trace의 입력·출력 길이를 반영한 합성 프롬프트다.

- 준비한 계획: results/hetero/controlled_plans/seed0.json ~ seed4.json.
- 조건: LLM 2 rps, DiT 0.3 rps, warmup 부하 60초 + 본 관측 600초.
- 둘 다 health check가 통과한 뒤 단일 프로세스·공통 clock에서 도착을 스케줄한다.
- warmup 동안 도착한 요청은 본 관측 채점에서 제외한다. warmup backlog는 이어진다. 이 길이가 정상상태를 보장하지는 않는다.
- 마지막 도착 후 관측 시간 끝까지 기다리고 추가 drain을 최대 300초 허용한다. 미완료는 drain_timeout으로 기록하고 분모에 포함한다.
- TTFT와 E2E는 계획 도착 기준이다. 실제 송신 지연도 기록하여 client 지연을 숨기지 않는다. TPOT는 completion usage 기준 평균이며 토큰별 ITL은 아직 기록하지 않는다.
- 출력: replay/{llm,dit}/requests.csv, manifest.json, backlog.csv. deployment.json과 GPU 모니터링은 상위 실행 폴더에 기록한다.
- backlog는 client 관측의 송신 후 미종료 요청 수다. 서버 내부 큐 길이와 동일하지 않다. 오류·timeout 종료도 종료 수에 포함한다.
- 기존 done 표시는 replay 절차 완료를 뜻하며 SLO 통과를 뜻하지 않는다. manifest의 errors와 서비스별 attainment를 확인한다.
- 서버는 별도 process group으로 시작하며 이번 실행의 group만 종료한다. 광범위한 pkill을 제거했다.

GPU 환경에서 실행하는 예시(저장소 루트 기준):

```bash
CONTROLLED_PLAN=results/hetero/controlled_plans/seed0.json \
  bash bench/hetero/run_mixed_modal.sh C 4090
CONTROLLED_PLAN=results/hetero/controlled_plans/seed0.json \
  bash bench/hetero/run_mixed_modal.sh S4 3090
```

모든 seed에서 실행하되 구성 순서는 섞는다. 기존 출력은 덮어쓰지 않는다. plan의 model과 서버 model이 다르면 시작을 거부한다. 단독 기준은 C none, S4 none, none 4090, none 3090을 같은 plan으로 실행한다. 기존 mixed_modal_summary.py는 이 새 출력 형식의 채점기로 사용하지 않는다.

검증 명령:

```bash
PYTHONDONTWRITEBYTECODE=1 ~/venv/jinwoo/bin/python -m unittest discover \
  -s bench/hetero/tests -v
bash -n bench/hetero/run_mixed_modal.sh
```

검증 완료: 계획 재현성·고정 관측 범위, 공통 clock 송신, warmup 제외, HTTP 오류·drain timeout 분모 포함, 출력 덮어쓰기 거부. 두 CPU 통합 테스트 통과. 이후 샌드박스 밖에서 GPU·로컬 tokenizer를 확인했고 실제 서버 통합 실행을 시작했다. 초기 PD health endpoint 오류를 수정했다. GPU 성능 반복 실험 완료 여부는 controlled_runs/batch_status.json과 comparison.md를 확인한다.


## 자동 실행·집계

[run_controlled_batch.py](../bench/hetero/run_controlled_batch.py)는 ①·④의 5개 seed를 순차 실행한다. 완료된 출력은 건너뛰고 미완료 출력이 있으면 중단해 확인을 요구한다. batch.lock으로 중복 batch 실행을 막는다. 최초 실행을 별도로 시작했을 때 --wait-pid로 종료를 기다릴 수 있다.

[compare_controlled.py](../bench/hetero/compare_controlled.py)는 완료된 manifest만 읽으며 동일 seed의 plan hash·prompt hash·model·관측 구간·분모를 대조한다. 불일치하면 비교를 거부한다. results/hetero/controlled_runs/comparison.md 및 comparison.json을 생성한다.

```bash
~/venv/jinwoo/bin/python bench/hetero/run_controlled_batch.py
python bench/hetero/compare_controlled.py
```

현재는 GPU 접근 가능한 실행 세션에서 batch가 진행 중이다. 다른 구성 실행을 동시에 시작하지 않는다. 진행 중 CSV 대신 requests.live.jsonl에 완료 요청이 즉시 저장되고, 최종 manifest는 drain 후 기록된다. TTFT/E2E는 계획 도착 기준이며 기존 pilot과 길이 분포가 다르므로 수치가 그대로 재현될 것으로 가정하지 않는다.


## 현재 연구 초점과 VLM 기능 검증 시작

사용자와 합의한 목적은 **혼재 환경에서 각 단계의 성능을 훼손하는 충돌·병목을 포착하고, 하나의 제어 장치로 불필요한 성능 악화를 줄이는 것**이다. 특정 SLO에서 하나의 배치만 통과하는 사례를 만드는 것은 목표가 아니다. 배치 trade-off와 제어에 의한 성능 경계 개선을 평가한다.

연구 대상 LLM·VLM·DiT는 모두 유지한다. 현재 제한된 GPU 환경의 다음 실험에서는 DiT 대신 VLM encoder를 이용해 분리·공유·제어 가능성을 확인한다. DiT 512² 단계 분리는 메모리 확장성 이점보다 단계 이동의 간섭·전송 순효과를 관찰하는 보조 실험으로 해석한다.

설치된 vLLM 0.28의 mm_encoder_only, EC producer/consumer 및 ECExampleConnector 구현을 확인했다. 로컬 VLM이 없어 공개 Qwen2-VL-2B-Instruct 모델 다운로드를 시작했다. 이후 3090 encoder → 4090 prefill/decode 분리 smoke를 순차 실행하도록 연결했다. 아직 실제 분리 추론 성공을 확정하지 않는다.

[실행 안내](../archive/legacy/vlm-2026-10/bench/vlm/README.md), [진행 상태](../archive/legacy/vlm-2026-10/results/vlm_encoder_check_status.json). 먼저 이미지 응답·encoder cache 전달·consumer 재인코딩 여부를 확인하고, 다음에 공유 간섭과 encoder 실행 허용 제어를 시험한다. 기능 검증 단계의 파일 전달 시간을 최적화 성능으로 해석하지 않는다.


### VLM 다운로드 장애 복구

최초 Qwen2-VL 다운로드는 Xet/CAS의 response body decoding 오류로 실패했다. 당시 GPU smoke는 시작되지 않았다. HF_HUB_DISABLE_XET=1로 일반 HTTP 전송을 재시도했고 다운로드 파일 크기가 증가하는 것을 확인했다. 실행 도구의 기본값에도 반영했다. 최종 GPU 분리 성공 여부는 상태 파일과 smoke 로그로 확인한다.


## LLM–VLM 공유 간섭·encoder 실행 허용 제어 파일럿

목표: encoder를 별도 GPU에 둔 VLM과 독립 LLM이 GPU를 공유할 때 추가 지연이 발생하는지, encoder 제출 동시성 제한으로 그 손해를 완화할 수 있는지 확인한다. 연구 대상은 LLM·VLM·DiT 모두 유지한다.

배치: 4090에 Qwen2-VL-2B의 prefill+decode, 3090에 Qwen2-VL encoder와 Llama-3.2-3B 전체. 모든 모델이 상주한다. 따라서 단독 기준은 다른 모델의 **메모리는 상주하지만 요청은 없는 기준**이다. 전용 GPU 단독 성능과 구분한다.

| 조건 | LLM req/s | VLM req/s | encoder 제출 제한 |
|---|---:|---:|---|
| llm_only | 1 | 0 | 없음 |
| vlm_only_low | 0 | 0.5 | 없음 |
| vlm_only_high | 0 | 2 | 없음 |
| mixed_low / mixed_high | 1 | 0.5 / 2 | 없음 |
| gated_low / gated_high | 1 | 0.5 / 2 | primer 요청 1개만 허용 |

첫 파일럿: seed 0, warmup 15초 + 측정 90초, 순서 shuffle, 최대 drain 120초. 정상상태나 통계적 유의성을 보장하는 반복 실험은 아니다. 시작은 이전 부하가 완전히 drain된 후이며 timeout이면 다음 조건을 실행하지 않는다.

LLM: 합성 입력 약 1024 tokens, 출력 256 tokens. VLM: 672²의 요청별 고유 합성 이미지, max_pixels 451584, 출력 64 tokens. 이미지는 encoder 부하 통제용으로 정확도 벤치마크가 아니다. 같은 서비스·같은 부하에서는 단독·공유·제어에 동일한 요청과 도착 계획을 적용한다. LLM arrival은 모든 LLM 조건에서 동일하다. 계획·payload hash를 저장한다.

제어는 공식 프록시의 encoder primer 호출에만 semaphore를 적용한다. VLM 언어 생성이 끝날 때까지 전체 요청을 잠그지 않는다. 모델 재배치나 커널 선점은 하지 않는다.

측정: 계획 도착 기준 TTFT/E2E, 요청 평균 TPOT, 스트리밍 chunk 간격 p90, 오류 수, encoder 허용 대기·primer 왕복시간, GPU utilization·메모리·전력. **chunk 간격은 token ITL이 아니며 primer 왕복시간은 순수 GPU encoder 시간이 아니다.** 파일 전달과 전처리가 포함되므로 단계별 커널 원인 분리는 후속 계측이 필요하다. 지연 요약은 성공 요청만 사용하고 오류 수를 함께 기록한다.

실행: archive/legacy/vlm-2026-10/bench/vlm/run_interference_pilot.py. 제어 프록시: archive/legacy/vlm-2026-10/bench/vlm/gated_proxy.py. 상태와 결과: archive/legacy/vlm-2026-10/results/vlm_interference_pilot/status.json, results.json, 조건별 summary.json·requests.live.jsonl. 전역 encoder_admission.jsonl은 요청 허용 대기를 기록한다.

판정: 무제어 공유의 추가 악화를 확인한 다음 고정 제한이 LLM 지연 감소와 VLM 추가 대기 사이에서 유용한 영역을 만드는지 본다. 효과가 없으면 그대로 보고한다. 현재 장치를 solver의 성능 개선으로 미리 주장하지 않는다.

## 다음 단계: encoder 실행 시점 제어 (seed 0)

질문: **encoder를 언제 실행할지 조절하면, LLM 응답 끊김을 줄이면서 VLM 대기 손해를 작게 유지할 수 있는가?** 반복 seed는 추가하지 않고, 기존 high 부하(LLM 1 req/s, VLM 2 req/s)의 같은 요청을 네 조건에 재생한다.

기존 파일럿은 7조건 완료, 측정 요청 오류 0건이다. high 부하에서 단독→공유 시 LLM 완료시간 p90은 4.07→4.63초, VLM 첫 응답 p90은 0.23→0.43초였다. 단순 직렬화에서는 각각 4.70초, 0.38초로 양쪽 손해를 함께 줄이지 못했다.

기존 기록을 시간축으로 대조하면 high 무제어에서 LLM 요청의 encoder primer 겹침 비율과 요청별 chunk 간격 p90의 Pearson 상관은 약 0.77이다. **탐색적 연관성**이며 인과관계나 GPU 커널 충돌의 증명이 아니다. primer에는 HTTP·전처리·파일 전달이 포함되고, 기존 기록에는 개별 chunk 시각이 없다. 같은 부하 변동이 양쪽 지연을 동시에 늘렸을 수도 있다. [분석 결과](../archive/legacy/vlm-2026-10/results/vlm_interference_pilot/overlap_analysis.json).

| 조건 | 직관적인 의미 | 동작 |
|---|---|---|
| free | 바로 실행 | encoder 동시성 제한 없음 |
| serialized | 하나씩 실행 | primer 1개만 진행 |
| fixed_pause | 항상 잠깐 양보 | 하나씩 실행 + 직전 primer 완료 후 최소 20ms 간격 |
| adaptive_pause | LLM이 느려지면 양보 | 하나씩 실행 + 최근 LLM chunk 간격이 25ms를 넘으면 잠시 대기 |

상황 제어는 클라이언트가 100ms마다 보내는 LLM chunk 간격의 최대값을 사용한다. 느린 간격 관측 후 100ms 동안 신호를 유효하게 보며, semaphore를 얻은 뒤 추가 정책 대기는 약 100ms로 제한한다(5ms polling). **큐 전체 대기시간을 100ms로 제한하는 것은 아니다.** 임계값은 파일럿 단독 chunk 간격 약 13ms와 high 공유 약 27ms를 참고한 최초 가설이며, 서비스 SLO가 아니다. 결과를 보고 같은 데이터에 임계값을 반복 맞추는 방식으로 효과를 주장하지 않는다.

warmup 15초 + 측정 90초, 최대 drain 120초, seed 0의 동일 도착·입력·출력, 조건 순서 shuffle. 모든 조건에서 동일한 피드백 계측을 실행한다. 직렬 기준을 포함해서 동시성 제한의 효과와 휴식의 추가 효과를 구분한다. 이 제어는 encoder 실행 전의 허용 판단이며 실행 중 커널 선점은 아니다.

새 기록에는 개별 chunk 시각, primer 실행 구간, 정책 대기, 피드백을 저장한다. 비교는 LLM 응답 간격·완료시간, VLM 첫 응답·완료시간, 오류·추가 대기를 함께 본다. 자유 실행보다 유용한 절충이 생기는지, 상황 제어가 고정 휴식보다 나은지 판단한다. 개선이 없으면 현재 제어 가설의 실패로 기록한다. 단일 seed만으로 통계적 유의성·일반적인 solver 필요성을 확정하지 않는다.

실행:
```bash
/home/jinwoo.kim27/venv/jinwoo/bin/python archive/legacy/vlm-2026-10/bench/vlm/run_interference_pilot.py \
  --control-study --seed 0 --out archive/legacy/vlm-2026-10/results/vlm_encoder_control_seed0
```

[진행 상태](../archive/legacy/vlm-2026-10/results/vlm_encoder_control_seed0/status.json). 완료 후 `results.json`과 조건별 `summary.json`을 비교한다. 기존 파일럿 결과는 보존했다.

### Seed 0 실행 결과: 구현은 정상, 상황 제어의 개선은 확인 못함

4조건 모두 완료했다. 조건마다 측정 LLM 74건, VLM 168건이며 측정 요청 오류는 0건, 피드백 전송 오류도 기록되지 않았다. 추가 seed는 실행하지 않았다.

| 조건 | LLM 완료시간 p90 | VLM 첫 응답 p90 |
|---|---:|---:|
| 바로 실행 | 4.71초 | 0.456초 |
| 하나씩 실행 | 5.37초 | 0.421초 |
| 고정 20ms 휴식 | 4.58초 | 0.380초 |
| LLM이 느려지면 양보 | 5.45초 | 0.555초 |

상황 제어는 실제 작동했다. warmup 포함 191개 primer 중 105개에서 1ms 넘는 정책 대기가 있었고, 총 허용 대기 p90은 약 287ms였다. 그러나 자유 실행보다 양쪽 지표가 나빠졌다. 고정 휴식은 이번 관측에서 양쪽 값이 낮지만, 단일 seed·순차 실행 결과여서 개선을 확정할 수 없다. 하나씩 실행 자체의 LLM 악화도 따로 관측됐다. 따라서 상황 제어의 악화를 전부 추가 휴식의 탓으로 단정하지 않는다.

개별 chunk 시간 대조에서 자유 실행의 primer 구간과 겹친 LLM 응답 간격 p90은 약 27.9ms, 겹치지 않은 간격은 12.3ms였다. 간섭과 부합하는 시간적 연관성이 더 구체화됐지만, primer는 순수 GPU 구간이 아니며 요청 chunk 표본도 독립 표본으로 보지 않는다.

판정: 현재의 **지연 관측 후 encoder 허용을 늦추는 규칙**은 효과가 입증되지 않았다. 지연 신호가 encoder 실행 이후에 도착하는지, LLM prefill 때문인지, primer의 CPU/파일 구간 때문인지 다음 분석에서 분리해야 한다. 바로 임계값을 바꿔 재실험하기보다 기록과 GPU 단계 계측을 통해 제어 시점·관측 신호의 적합성을 먼저 확인한다.

[결과](../archive/legacy/vlm-2026-10/results/vlm_encoder_control_seed0/results.json), [시간 겹침 분석](../archive/legacy/vlm-2026-10/results/vlm_encoder_control_seed0/overlap_analysis.json).

## 후속 실험: 지연 후 반응과 첫 응답 전 보호 비교

가설: chunk 지연을 보고 나서 encoder를 늦추는 방식은 이미 진행 중인 encoder를 중단하지 못한다. LLM 첫 응답 전 구간을 보호하는 신호가 더 유용한지 비교한다. 보호 대상은 **LLM 요청 전송 후 첫 content chunk 전까지**이며, 실제 GPU prefill 구간과 동일하지 않다(큐·네트워크 포함).

seed 0, warmup 15초 + 측정 60초, LLM 1 req/s·VLM 2 req/s, 동일 입력과 도착 계획, 순서 shuffle. 조건은 `serialized`(하나씩 실행), `reactive_gap`(25ms 초과 chunk 신호 후 양보), `prefill_guard`(첫 응답 대기 중 LLM이 있을 때 양보)이다. 두 제어의 추가 정책 대기는 약 100ms로 제한한다. 큐 전체 대기 상한은 아니다. 피드백은 모든 조건에서 100ms마다 동일하게 보낸다. 첫 응답 전 상태도 최대 300ms 지난 신호는 무시한다. 기존 결과보다 관측 길이가 짧으므로 이번 세 조건끼리 비교한다. 추가 seed는 없다.

계측: chunk 시각, 요청 전송→첫 응답 상태, 피드백 발신·수신 시각, primer 호출 시작·완료, 정책 대기. `analyze_signal_timing.py`와 `analyze_encoder_overlap.py`로 시간 관계를 분석한다. `nsys`·`ncu` 실행 파일은 현재 PATH에서 찾지 못했다. 이번 실험에서 GPU 커널 충돌의 직접 계측을 수행했다고 주장하지 않는다. 커널 원인 분리는 별도 profiler 단계가 필요하다.

실행:
```bash
/home/jinwoo.kim27/venv/jinwoo/bin/python archive/legacy/vlm-2026-10/bench/vlm/run_interference_pilot.py \
  --timing-study --duration 60 --seed 0 --out archive/legacy/vlm-2026-10/results/vlm_signal_timing_seed0
```

[상태](../archive/legacy/vlm-2026-10/results/vlm_signal_timing_seed0/status.json). 결과가 개선되지 않으면 그대로 기록하고 신호·제어 시점 가설을 수정한다.

### 첫 응답 전 보호 실험 완료 (seed 0)

3조건 모두 완료. 조건마다 측정 LLM 48건·VLM 108건, 측정 요청 오류 0건이다. 추가 seed는 실행하지 않았다. 지표는 성공 요청의 p90이다.

| 방식 | LLM 첫 응답 | LLM 완료 | LLM 요청별 chunk 간격 p90의 p90 | VLM 첫 응답 | VLM 완료 |
|---|---:|---:|---:|---:|---:|
| 하나씩 실행 | 0.240초 | 4.447초 | 27.25ms | 0.378초 | 1.287초 |
| 첫 응답 전 보호 | 0.220초 | 4.428초 | 27.43ms | 0.404초 | 1.323초 |
| 느려진 뒤 양보 | 0.262초 | 4.444초 | 27.57ms | 0.504초 | 1.423초 |

해석: 첫 응답 전 보호는 LLM 첫 응답 값이 조금 낮지만, 완료시간은 거의 같고 응답 간격은 개선되지 않았다. VLM에는 추가 지연이 생겼다. 지연 후 제어는 LLM 완료시간 개선 없이 VLM 첫 응답·완료시간을 늘렸다. 첫 응답 전 보호가 지연 후 제어보다 대기 손해를 덜 만들었지만, 유효한 간섭 완화 장치가 입증된 것은 아니다.

실제로 추가 정책 대기가 1ms를 넘은 primer는 측정 시작 기준 첫 응답 전 보호 11/108건, 지연 후 제어 58/108건이다. 정책 대기 p90은 각각 약 3ms와 101ms였다. 피드백 전송시간 p90은 약 1.4~1.6ms였지만, 별도로 100ms의 주기적 관측 지연이 존재한다. 신호의 신선도와 실행 전 대기만으로는 진행 중 encoder를 중단할 수 없는 점을 구분해야 한다.

세 조건 모두 primer와 겹친 LLM chunk 간격 p90은 약 27~28ms, 겹치지 않은 경우 약 12ms였다. 이 연관성은 남았으며 첫 응답 보호만으로 해소되지 않았다. 이를 근거로 **늦은 신호만이 실패 원인이라고 확정하지 않는다.** CPU·파일 전달과 GPU 커널 시간을 분리하지 않은 상태이고 단일 seed·조건별 순차 실행이므로 작은 차이는 우열의 증명이 아니다.

다음 우선순위: 임계값 변경보다 **encoder가 실행되는 동안 LLM decode가 받는 손해의 단계별 계측**이다. 현재 실행 전 대기 장치의 한계를 기록하고, 필요하면 GPU 실행 구간 계측을 별도 진단으로 수행한다. profiler 오버헤드가 있는 진단 결과를 위 성능 비교에 섞지 않는다.

[결과](../archive/legacy/vlm-2026-10/results/vlm_signal_timing_seed0/results.json), [신호 시각 분석](../archive/legacy/vlm-2026-10/results/vlm_signal_timing_seed0/signal_timing_analysis.json), [겹침 분석](../archive/legacy/vlm-2026-10/results/vlm_signal_timing_seed0/overlap_analysis.json).

## GPU 실행 구간 계측: 수집 완료

seed 0에서 LLM만 요청 / VLM만 요청 / 함께 요청의 3조건을 실행했다. warmup 15초 + 부하 관측 30초이며, 각 조건에서 warmup 후 5초에 encoder(8510), 독립 LLM(8540)의 PyTorch profiler를 시작하고 약 8초 후 중지했다. 시작·중지 호출과 trace 저장 때문에 조건별 실제 소요시간이 늘어날 수 있다. profiler가 성능을 교란하므로 이 실행의 지연 값으로 앞선 무계측 실험 대비 개선을 주장하지 않는다.

수집 상태는 complete(3/3), 측정 요청 오류 0건이다. encoder·LLM 각각 3개의 trace가 생성됐다. 요청이 없는 서비스의 trace는 GPU kernel 이벤트 0개이고, 요청이 있는 단독·공유 trace에는 GPU kernel 이벤트가 존재하는 것을 확인했다. 파일 목록과 이벤트 수는 `archive/legacy/vlm-2026-10/results/vlm_gpu_profile_seed0/trace_inventory.json`에 기록했다. **GPU trace 수집 완료와 원인 분석 완료는 구분한다.** 다음은 커널 실행시간·실행 간격과 클라이언트 응답 간격의 대조다.

실행: `run_interference_pilot.py --profile-study --duration 30 --seed 0 --out archive/legacy/vlm-2026-10/results/vlm_gpu_profile_seed0`.

용어: “encoder 하나씩”은 프록시가 encoder primer 호출의 동시 진행을 1개로 제한한다는 뜻이다. 이미지 A의 encoder 호출 응답 후 이미지 B를 허용한다. A의 언어 생성 완료를 기다리지 않으며 독립 LLM도 계속 실행한다. 이번 GPU 계측 실험은 제어 없이 원인을 관찰한다.

## GPU trace 분석 결과: GPU 실행 대기 증가를 관측

**현재 결론:** 공유 실행의 악화를 전처리·파일 전달만으로 설명하기는 어렵다. GPU에 커널을 제출한 뒤 실행이 시작되기까지의 긴 대기도 증가했다. 다만 이 대기는 자기 CUDA stream의 선행 작업 대기와 프로세스 간 GPU 스케줄링을 함께 포함하므로, GPU 자원 경쟁의 정확한 종류·기여율까지 확정하지 않는다.

### 비교 구간과 함정

두 profiler의 start 응답 중 늦은 시각 +1초부터, 첫 stop 호출 전까지의 공통 구간(조건당 약 7초)만 사용했다. LLM profiler 중지·trace 저장에 공유 조건에서 약 30초가 걸렸고 encoder profiler는 그동안 계속 동작했다. 전체 encoder trace와 LLM trace를 그대로 비교하면 안 된다. trace의 `baseTimeNanoseconds`와 이벤트 timestamp를 이용해 같은 시간축으로 맞췄다.

계측 대상은 3090에서 실행되는 독립 LLM과 VLM encoder다. 파일 전달과 4090의 VLM 언어 처리 구간은 이 GPU trace로 직접 분해하지 못한다. profiler의 실행 오버헤드가 있으므로 아래 값은 원인 진단용이며 기존 무계측 서비스 성능 값에 합치지 않는다.

| 관측 지표 | 해당 서비스만 요청 | 함께 요청 |
|---|---:|---:|
| LLM 커널 제출 API 반환→GPU 시작 대기 p90 | 0.118ms | 22.005ms |
| Encoder 커널 제출 API 반환→GPU 시작 대기 p90 | 45.060ms | 99.910ms |
| LLM 커널 제출 API 반환→GPU 시작 대기 중앙값 | 0.0278ms | 0.00114ms |
| LLM CPU execute 구간 p90, decode 2개인 동일 annotation | 15.615ms (147회) | 18.851ms (147회) |
| Encoder CPU execute 구간 p90, context 1개/597 tokens | 78.711ms (13회) | 152.617ms (11회) |

커널 대기는 CUDA launch API의 correlation ID로 GPU 이벤트를 연결해 계산했다. 비동기 launch는 API 반환 전에 GPU가 실행될 수도 있어 음수 값은 0으로 잘랐다. **모든 커널이 느려진 것은 아니다. 중앙값과 긴 대기의 변화가 다르다.** CPU execute는 host annotation의 경과시간이며 순수 GPU 연산시간이나 요청 전체 지연이 아니다.

### 연산 자체와 대기를 구분

커널 이름·grid·block·shared memory가 같고 양쪽에 30회 이상 존재하는 그룹을 비교했다. LLM의 공유 조건 누적 kernel 시간 상위 5개 그룹의 실행시간 중앙값 비율은 약 1.001~1.008배로 거의 같았다. 모든 커널이 같은 것은 아니며 그 밖의 그룹에는 증가가 있다. 같은 launch 설정도 텐서·batch·캐시 상태의 완전한 동일성을 보장하지 않는다.

공유 trace 내부에서 encoder의 실제 GPU kernel 구간과 겹친 LLM 제출 후 대기들의 p90은 약 29.3ms, 겹치지 않은 대기들은 약 0.109ms였다. 긴 구간일수록 겹칠 확률이 높다는 선택 효과가 있으므로 이를 독립적인 인과 증명이나 경쟁 기여율로 사용하지 않는다. 다만 앞선 primer HTTP 구간 대조보다 실제 GPU 실행에 가까운 시간 관계가 확인됐다.

LLM kernel busy 구간 사이 간격 p90은 단독 약 50.6µs, 공유 약 45.3µs였다. 따라서 “모든 커널 사이 간격이 커졌다”라는 설명도 맞지 않는다. 이번 분석의 핵심은 **제출된 작업의 긴 실행 대기와 일부 실행 구간 지연**이다.

### 연구 방향에 주는 의미

1. 실행 전 요청을 잠깐 미루는 장치는 이미 GPU에 제출된 작업의 실행 순서를 바꾸거나 선점하지 못한다. 앞선 단순 대기가 개선을 만들지 못한 결과와 양립하지만, 그것만이 실패 원인이라고 단정하지 않는다.
2. 다음 장치는 encoder 한 요청이 만드는 GPU 작업 구간의 길이·동시성·배치를 조절하거나 GPU 배치를 바꾸는 방향을 검토할 근거가 있다. 먼저 어떤 구간을 실제로 제어 가능한지 확인해야 한다.
3. 전처리·파일 전달이 무관하다고 결론 내리지는 않는다. 그 기여율을 수치로 말하려면 별도 span 및 대조 실험이 필요하다. 특히 decoder 파일 읽기까지 현재 trace에 포함됐다고 해석하지 않는다.

재현: `analyze_gpu_profile.py archive/legacy/vlm-2026-10/results/vlm_gpu_profile_seed0`, `plot_gpu_profile_summary.py archive/legacy/vlm-2026-10/results/vlm_gpu_profile_seed0`.

[상세 수치](../archive/legacy/vlm-2026-10/results/vlm_gpu_profile_seed0/gpu_analysis.json), [요약 그림](../archive/legacy/vlm-2026-10/results/vlm_gpu_profile_seed0/gpu_wait_summary.png). 단일 seed·짧은 공통 구간·profiler 오버헤드가 있는 탐색적 진단이며 추가 GPU 실험은 실행하지 않았다.

## Solver 후보 배치 비교: encoder를 어느 GPU에 둘 것인가

GPU 실행 대기 증가 관측을 구성 선택 문제로 연결한다. 2개 GPU·모델·입출력·도착 계획·seed 0을 고정하고 encoder 배치만 바꾼다.

| 후보 | 3090 | 4090 |
|---|---|---|
| with_llm: LLM과 encoder 공유 | 독립 LLM + VLM encoder | VLM prefill/decode |
| with_vlm: VLM 단계들을 같은 GPU에 배치 | 독립 LLM | VLM encoder + VLM prefill/decode |

with_vlm도 encoder/언어 처리 프로세스와 파일 전달 connector를 유지한다. 통합 VLM 서버와 동일한 구현이 아니다. 전송 방식을 바꾸지 않은 상태에서 배치 후보를 비교한다. 각 프로세스 메모리 비율은 기존과 동일(encoder 0.1, 각 언어 모델 0.4)이며 GPU 수·종류는 고정이다. 같은 GPU에서도 프로세스별 KV cache 확보량은 환경에 따라 달라질 수 있다.

각 배치에서 LLM만 요청, VLM만 요청, 함께 요청의 3조건을 실행한다. 모델은 모든 조건에 상주한다. 총 6조건, warmup 15초 + 측정 90초, 최대 drain 120초, profiler 없음. 신규 배치인 with_vlm의 실행 가능성을 먼저 확인하고 with_llm을 이어서 실행한다. 조건 순서는 기존 seed 기반 shuffle을 사용한다. 배치 블록 순서는 고정이므로 시간 경과 영향을 배제했다고 주장하지 않는다. 추가 seed 반복은 없다.

판정: 각 배치의 혼합 성능과 자기 배치 단독 기준 대비 악화율을 함께 본다. encoder GPU가 3090→4090으로 바뀌므로 절대 성능 차이를 전부 간섭 감소라고 해석하지 않는다. with_vlm에서는 encoder와 VLM 언어 처리 사이 간섭이 새로 생길 수 있다. 후보 중 한 배치가 모든 지표에서 우세하면 그 결과를 그대로 기록하며, 현 조건에서 복잡한 solver 필요성을 입증한 것으로 주장하지 않는다. 같은 후보 공간에서 단독 성능 기반 선택과 혼합 실행을 반영한 선택이 달라지는지는 다음 판단 대상이다.

실행: `archive/legacy/vlm-2026-10/bench/vlm/run_placement_study.py`. [전체 상태](../archive/legacy/vlm-2026-10/results/vlm_placement_seed0/status.json)와 배치 하위 `status.json`에 진행 상황을 기록한다. 완료 시 동일 payload/계획을 검증하고 `comparison.json`에 단독 대비 악화율을 자동 저장한다. 기존 결과는 보존한다.

### 배치 비교 결과 확인 (2026-10-14)

2배치 × 3조건 완료, 측정 요청 오류 0건. 같은 payload와 도착 계획 검증 통과. 아래는 이번 무계측 실험끼리의 p90 비교다.

| 혼합 실행 지표 | encoder와 독립 LLM이 3090 공유 | encoder를 VLM 언어 처리와 함께 4090에 배치 |
|---|---:|---:|
| LLM 첫 응답 | 0.316초 | 0.177초 |
| LLM 답변 완료 | 4.979초 | 3.776초 |
| LLM 요청별 chunk 간격 p90의 p90 | 28.18ms | 13.15ms |
| VLM 첫 응답 | 0.523초 | 0.193초 |
| VLM 답변 완료 | 1.442초 | 1.097초 |

각 배치의 자기 단독 기준 대비 악화: LLM 완료시간 +41.9%→+4.5%, VLM 첫 응답 +126.3%→+2.3%. 단독 LLM 완료시간은 각각 3.510초, 3.613초로 변경 배치가 원래 더 빠른 것은 아니었다. VLM 단독 첫 응답은 0.231초→0.188초로 encoder GPU 이동 자체의 이득도 있다. 따라서 혼합 실행의 절대 개선 전체를 간섭 감소에 귀속하지 않는다.

해석: 같은 GPU 2개에서 encoder 배치를 바꾸자 LLM 완료시간 약 24%, VLM 첫 응답 약 63% 감소가 관측됐다. 단계 배치가 유용한 제어 후보라는 근거다. 두 서비스의 주요 지표가 동시에 좋아져 이번 부하에서는 배치 간 trade-off나 복잡한 solver 필요성이 입증된 것은 아니다. 단일 seed·배치별 순차 실행·다른 GPU 속도라는 한계가 남는다. 다음 후보 검증은 혼합 비율/단계 부하 변화에서도 이 배치가 계속 우세한지, 혹은 선호 배치가 달라지는지 확인하는 것이다. 선호가 바뀌지 않으면 단순 고정 배치가 충분하다는 결과를 받아들인다.

[원시 비교 결과](../archive/legacy/vlm-2026-10/results/vlm_placement_seed0/comparison.json).

## 부하 변화에 따른 배치 선호 검증 (2026-10-14 시작)

질문: 기존 1 LLM req/s + 2 VLM req/s에서 우세했던 with_vlm 배치가 다른 혼합 부하에서도 우세한가? 선호가 바뀌면 그 조건과 trade-off를 관찰하고, 바뀌지 않으면 이번 범위에서 고정 배치가 충분하다는 결과를 기록한다.

| 조건 | LLM req/s | VLM req/s |
|---|---:|---:|
| reference_mix: 기존 비율 | 1 | 2 |
| llm_heavy: LLM 요청 증가 | 4 | 0.5 |
| vlm_heavy: VLM 요청 증가 | 0.25 | 4 |
| both_heavy: 양쪽 요청 증가 | 4 | 4 |

위 부하는 Poisson 계획의 목표 도착률이며 실제 관측 건수는 달라진다. 두 배치에서 동일 seed 0·계획·payload를 사용한다. 입력 길이, 이미지 크기, 출력 길이, GPU 2개, 메모리 비율, eager 실행, 파일 connector는 유지한다. 모든 encoder는 무제어로 실행한다. 조건마다 warmup 15초 + 측정 60초, drain 최대 120초이며 총 8조건이다. 고부하에서 drain timeout이면 다음 조건 진행을 중단하고 실패 상태를 남긴다. 단순 완료를 성공률 100%로 해석하지 않고 오류도 확인한다.

이번에는 with_llm→with_vlm 순서이며 배치 안의 조건은 seed 기반 shuffle이다. 순서 효과가 완전히 제거된 설계는 아니고 반복 seed는 추가하지 않는다. 각 부하의 두 배치끼리 LLM/VLM 첫 응답·완료시간·응답 간격·오류를 비교한다. 별도 부하별 단독 기준은 이번 실행에 포함하지 않아 이전 단독 수치로 악화율을 산출하지 않는다. reference_mix도 이번 60초 실행끼리 비교한다. p90은 성공 요청 기준이며 작은 표본, 특히 vlm_heavy의 LLM 결과는 탐색적으로 해석한다.

실행: `archive/legacy/vlm-2026-10/bench/vlm/run_placement_study.py --load-sweep`.
[상태](../archive/legacy/vlm-2026-10/results/vlm_placement_load_sweep_seed0/status.json). 배치별 하위 `status.json`에서 현재 조건을 볼 수 있고, 완료 후 `comparison.json`에 결과와 동일 계획/payload 검증 결과를 바탕으로 한 비교 입력이 저장된다. 이 실행은 비용 절감이나 solver의 최적성을 미리 주장하지 않는다.

### 부하 변화 실험 결과: 주요 지표의 배치 선호 역전 없음

8조건 완료, 측정 요청 오류 0건, 두 배치의 payload·도착 계획 검증 통과. seed 0 한 번의 결과다. 아래 수치는 성공 요청 p90이며 같은 60초 관측 조건끼리 비교한다.

| 부하 (LLM/VLM req/s) | LLM 완료: encoder 3090 공유 → 4090 배치 | VLM 첫 응답: encoder 3090 공유 → 4090 배치 |
|---|---:|---:|
| 기존 비율 (1/2) | 4.395 → 3.568초 | 0.423 → 0.193초 |
| LLM 중심 (4/0.5) | 18.886 → 17.778초 | 0.384 → 0.149초 |
| VLM 중심 (0.25/4) | 4.572 → 3.273초 | 0.519 → 0.229초 |
| 양쪽 고부하 (4/4) | 40.477 → 20.298초 | 0.720 → 0.231초 |

VLM 완료시간도 네 조건 모두 4090 배치가 낮았다. LLM 첫 응답 역시 네 조건 모두 낮았다. 따라서 주요 첫 응답·완료 지표에서 선호 배치 역전은 관측되지 않았다. 다만 모든 지표의 절대 우위는 아니다. VLM 요청 평균 TPOT p90은 VLM 중심에서 14.84→15.36ms, 양쪽 고부하에서 15.13→15.29ms로 소폭 증가했고, VLM chunk 간격 지표도 변경 배치에서 소폭 증가했다. 단일 seed의 작은 차이를 안정된 trade-off라고 주장하지 않는다.

LLM 4 req/s 조건은 변경 배치에서도 첫 응답 p90이 약 11.2~13.7초다. 배치 변경만으로 높은 부하의 지연이 해결되지 않았다. 유한 시간 부하·drain 결과이므로 정상상태 처리 용량을 추정하거나 이 지연 전체를 encoder 간섭으로 귀속하지 않는다. 이번 실행에는 각 부하의 단독 기준이 없다.

현재 판단: 이 모델 크기·입출력 길이·GPU 조합·부하 범위에서는 VLM 단계들을 4090에 두고 독립 LLM을 3090에 두는 고정 배치가 주요 지표에 유리하다. 요청률 변화만으로 적응형 배치 solver의 필요성은 입증되지 않았다. 기존 관측인 공유 실행 손해와 배치에 따른 완화는 유지된다. 다음 연구 후보는 실제 플랫폼에 해당하는 모델/입출력/단계별 부하 구성을 선정해 선택 공간을 넓히는 것이며, 역전을 만들기 위한 임의 조건 탐색과 구분한다.

[결과](../archive/legacy/vlm-2026-10/results/vlm_placement_load_sweep_seed0/comparison.json). LLM 표본 수는 기존 비율 48건, LLM 중심·양쪽 고부하 242건, VLM 중심 15건으로 특히 마지막 조건의 LLM p90은 탐색적이다.

## Stage 0 기능 smoke 완료 (2026-10-15)

[인계 문서](notes/claude_research_handoff_20261015.md)의 Stage 0을 실행했다. SLO·비용 판정은 하지 않았다. 아래 수치는 기능 확인용이며 결론에 쓰지 않는다.

**하네스 변경.** `run_mixed_modal.sh`에 `REPLAY_ONLY=llm|dit`를 추가했다. 두 서버를 모두 올린 채 한 서비스 요청만 재생하는 **상주형 단독(resident solo)** 경로다. 기존 `none`은 상대 모델을 띄우지 않으므로 다른 조건이다. `run_controlled_batch.py`는 후보 목록(`LLM_DIT[:only]`), plan 파일, 출력 경로, drain timeout, 실패 시 계속 진행을 인자로 받는다. 기본 인자는 10/13 batch의 작업 순서를 재현한다. 보고서는 `stage0_report.py`가 만든다. 기존 CPU 테스트 2건 통과.

**조건.** plan `controlled_plans/smoke_seed900.json`(seed 900, 탐색·검증 seed와 분리). warmup 15초 + 관측 45초, LLM 2 rps + DiT 0.3 rps, LLM 125건(측정 89)·DiT 17건(측정 13), in+out 최대 7444 토큰. 후보 4개 × replay {both, llm만, dit만} = 12 run, 순서 shuffle, drain 최대 120초. 시작 전 두 GPU 모두 compute 프로세스 없음을 확인했다.

**결과.** 12/12 완료, 측정 요청 오류 0, send_lag p99 ≤ 10 ms. 같은 replay 종류 안에서 plan hash·prompt hash·CSV 열·분모가 모두 일치했다. 로그의 EngineDeadError 30줄은 모두 `done` 시각 이후에 찍힌 종료 잡음이다(runner가 EngineCore를 먼저 종료). [보고서](../results/hetero/stage0_smoke/stage0_summary.md), [JSON](../results/hetero/stage0_smoke/stage0_summary.json).

| 후보 | 4090 | 3090 | replay 중 최대 메모리 MiB (4090 / 3090) | 비고 |
|---|---|---|---|---|
| S4_4090 (1 GPU) | LLM + DiT | 유휴 | 21,802 / 18 | 24,564 중 약 2.7 GB 여유 |
| S3_3090 (1 GPU) | 유휴 | LLM + DiT | 253 / 21,297 | 24,576 중 약 3.2 GB 여유 |
| S4_3090 | LLM | DiT | 13,398 / 8,271 | 기존 측정 구성 |
| S3_4090 | DiT | LLM | 8,655 / 13,043 | 이번에 처음 기동 |

1 GPU 후보 둘 모두 로드와 요청 처리에 성공했다. 여유 메모리가 3 GB 안쪽이라 Stage 1의 긴 관측에서 vLLM KV 할당은 util 0.5로 고정되지만 DiT 활성화 메모리 변동은 확인 대상이다. 전력은 1 GPU 후보에서 해당 카드가 정격 근처(4090 448 W, 3090 370 W)까지 올라갔다. 관찰 기록이며 비교 결론은 아니다.

**3 서비스 동시 상주 추정.** VLM 실험의 실측(encoder util 0.1 + 언어 util 0.4 = 한 카드 최대 12,104 MiB)과 이번 LLM+DiT 1 GPU 실측(약 21,800 MiB)을 합치면, 한 카드에 LLM+DiT, 다른 카드에 VLM encoder+언어를 두는 배치는 현재 메모리 비율을 바꾸지 않고도 들어간다. LLM과 VLM 언어 모델을 같은 카드에 두는 배치는 util 합이 0.9 이상이라 비율 조정 없이는 불가하다. 추가 장비 없이 세 서비스 상주가 가능한 배치가 최소 하나 있다는 뜻이며, 성능은 미측정이다.

**Stage 1 진행 가부.** 진행 가능. 네 후보 모두 Stage 1 후보 공간에 넣는다. P/D 분리(C_*)와 DiT 단계 분리(C_split)는 pilot 실측이 있으나 상주형 단독 경로와 계측 호환성은 이번에 확인하지 않았으므로 Stage 1 전에 같은 smoke를 거친다. Stage 1에서 사전 고정할 항목: SLO 격자(TPOT 정의 명시), 가격 격자(상각·임대 병기, 유휴 카드 과금 포함·제외), 수요점, 관측 길이, 안정성 보고 방식(복수 관측창의 도착·완료율과 최종 in-flight).

### Stage 0 후속: KV cache 용량 불일치와 고정 (2026-10-15)

Codex 검토에서 같은 후보의 단독·혼합 실행 KV 용량이 달랐음을 확인했다(S3_4090: LLM만 37,776 토큰, 혼합 50,272 토큰). 12개 실행 전체를 대조하면 세 값(37,776 / 49,888 / 50,272)이 나왔다.

- **원인.** `gpu-memory-utilization 0.5`는 총 예산만 고정한다. vLLM은 profiling 시점의 weights + non-torch + activation 피크를 빼고 남는 메모리를 KV로 잡는다. 문제의 실행은 torch.compile 캐시 미스가 있었고(compilation 9.86 s, 다른 실행 0.11 s) 그동안 non-torch 6.99 GiB·activation 0.75 GiB로 늘어(정상 6.28·0.13) KV가 1.33 GiB 줄었다. 4090과 3090 사이 49,888/50,272 차이도 같은 산정 방식의 결과다. 캐시 상태에 따라 달라지므로 비결정적이다.
- **조치.** `run_mixed_modal.sh`의 vLLM 인자에 `--kv-cache-memory-bytes`(환경변수 `KV_BYTES`, 기본 4 GiB = 4,294,967,296)를 추가했다. `KV_BYTES=0`이면 이전 방식. 4 GiB는 컴파일 피크가 있어도 util 0.5 예산 안에 들어가는 보수적인 값이다. `deployment.json`에 KV 바이트와 서버별 실제 KV 토큰 수를 기록하고, `stage0_report.py`가 모든 실행의 KV 토큰 동일 여부를 검사한다. 첫 확인: C_split의 prefill·decode 서버 모두 37,440 토큰.
- **영향.** Stage 0의 기능 통과 판정은 유효하다. 이전 pilot·통제 재측정(10/13)은 util 기반 KV였으므로 Stage 1 수치와 직접 비교하지 않는다.
- **추가 smoke 실행 중.** 기본 후보 4개(both) + 분할 후보 C_4090·C_3090·C_split × {both, llm만, dit만} = 13 run, 출력 `results/hetero/stage0_smoke_kvfix/`. 분할 후보의 상주형 단독 경로와 KV 동일성을 확인한 뒤 Stage 1 후보 공간을 확정한다.

## Stage 1 프로토콜 초안 — 실행 전 고정 대상 (2026-10-15)

Codex 검토 반영: 평가 2(같은 비용에서 SLO 충족 처리능력)부터 측정한다. Stage 1은 탐색 파일럿이며, 발견한 경계는 Stage 2에서 새 seed·별도 실행으로 검증한다. 아래 값은 실행 전에 고정하고 격자 전체를 보고한다. run 수·시간은 가설이다.

**질문.** 고정 GPU 2장(또는 1장)에서 기준 수요 mix의 비율을 유지하며 공통 배율 α를 올릴 때, 후보별로 두 서비스가 모두 SLO를 만족하는 최대 α는 어디까지인가. 후보 간 α_max가 다르면 같은 자원의 SLO 충족 처리능력이 다르다는 뜻이다.

**후보 공간.** Stage 0 통과 후보. 기본 4개(S4_4090, S3_3090, S4_3090, S3_4090)는 확정. 분할 3개(C_4090, C_3090, C_split)는 kvfix smoke 결과에 따라 포함 여부를 정한다. KV 예산 4 GiB 고정, LLM util 0.5, DiT 512² 28 step batch 1, T5 제외. 모든 후보에서 두 모델이 상주한다.

**수요.** 기준 mix λ0 = (LLM 2 rps, DiT 0.3 rps), code trace 길이. 배율 α ∈ {0.5, 1.0, 1.5, 2.0}. α별 plan은 seed 901(탐색용)로 새로 생성하며, 같은 α에서는 모든 후보가 같은 plan·prompt를 재생한다. Stage 2 검증 seed는 902 이상에서 따로 고정한다.

**관측.** warmup 60 s 제외, 관측 600 s, drain 최대 300 s. α=0.5에서 DiT 측정 표본은 약 90건, LLM 약 600건이다. DiT 표본이 60건 미만이 되는 조건은 두지 않는다. 미완료·오류·drain_timeout은 실패로 분모에 포함하고 종류별로 따로 보존한다.

**SLO 정의(용어 고정).**
- LLM TTFT: 계획 도착 시각부터 첫 content chunk까지. client 송신 지연 포함(send_lag 별도 기록).
- LLM TPOT: 요청별 평균 decode 간격 = (end − first) / (completion_tokens − 1). **token 간 지연의 p95가 아니다.** chunk 시각은 아직 기록하지 않으므로 tail ITL은 보고하지 않는다.
- LLM 충족: 한 요청이 TTFT와 TPOT 조건을 **동시에** 만족. 서비스 충족률 = 만족 요청 / 측정 요청(오류 포함).
- DiT E2E: 계획 도착부터 이미지 완료까지. 충족률 분모 동일.
- 격자: TTFT {0.3, 0.5, 1, 2} s × TPOT {12, 20, 30, 50} ms × DiT E2E {3, 5, 10, 20} s × 목표 충족률 {90, 95, 99}%. 기존 탐색 격자와 같다. 대표 셀은 사후에 고르지 않고 격자 전체를 heatmap·표로 보고한다. 설명에 쓰는 셀은 "예시"로 표기한다.

**안정성 보고.** 관측 600 s를 3개 창(200 s)으로 나눠 창별 도착 수·완료 수·창 끝 in-flight를 서비스별로 보고한다. 기울기 단일 임계값으로 불가를 판정하지 않는다. 마지막 창의 in-flight가 첫 창보다 크고 drain_timeout이 있으면 "포화 의심"으로 표기하고, Stage 2에서 긴 관측으로 확인한다.

**지표.** 후보별 α_max(격자 셀마다) = 두 서비스가 모두 목표 충족률을 넘는 최대 α(시험한 격자 위의 값이며 실제 상한은 "≥"로 읽는다). 보조: 서비스별 충족률 벡터, 지연 p50/p90(성공 요청), 오류·미완료 수.

**비용·자원.** 1차: 후보가 점유한 GPU 종류·장수와 카드별 전력 적분 Wh(gpu.csv). 2차: 가격 비율 r = p_3090 / p_4090 ∈ {0.5, 0.75, 0.94, 1.0}에 대한 시간당 비용 = p_4090 × (n_4090 + r·n_3090). 0.94는 출시가 상각 비율(1499/1599), 그 외는 임대 시세 범위를 덮는 민감도 값이며 날짜 기록된 시세표를 확보하면 교체한다. 시나리오: (a) 상각 4년 + 전력 $0.12/kWh × PUE 1.3, (b) 임대 시세. 유휴 카드 과금 **포함·제외 둘 다** 보고한다. 포함 조건에서는 1 GPU 후보의 총비용이 2 GPU 후보와 같다.

**비교 산출물.** α_max / 시간당 비용을 후보·가격 비율·SLO 셀별로 제시. 이는 LLM-PILOT CE의 고정 자원 형태이며 token/s·image/s 합산이 아니다. 비용 절감(평가 1)은 1 GPU 후보가 어떤 α에서 통과하는지로 같은 데이터에서 읽는다.

**실행 규모(가설).** 혼합: 후보 7 × α 4 = 28 run, 약 13 분/run → 약 6 h. 상주형 단독 프로파일(단독 기반 선택기용): LLM 배치 3종(4090, 3090, PD) + DiT 배치 3종(4090, 3090, split) × α 4 = 24 run → 약 5 h. 혼합을 먼저 실행하고 단독은 이어서 한다. 순서는 α별로 후보를 shuffle한다.

**중단·보류 조건(최초 가정값, 확정 기준 아님).** 설정 실패 또는 측정 요청 오류율이 눈에 띄게 높은 run은 원인 확인 후 재실행. 같은 후보·α에서 drain_timeout 2회면 그 이상 α는 실행하지 않는다.

**아직 하지 않는 것.** 온라인 전환, 실행 허용 제어, 세 서비스 동시 실행. 세 서비스 동시 상주는 Stage 0의 메모리 합산 추정만 있으므로 "가능성이 있는 배치"로 두고, VLM 하네스를 포함한 실제 동시 로드·요청 smoke를 별도로 한다.

### KV 고정 후 smoke 결과 (2026-10-15): 7개 후보 Stage 1 진입 가능

13/13 완료, 측정 요청 오류 0, send_lag p99 ≤ 7 ms, 계측 일치 검사 통과. **모든 실행·모든 서버에서 KV 37,440 토큰으로 동일**(PD의 prefill·decode 서버 포함). 분할 후보 세 개의 상주형 단독 경로(LLM만·DiT만)도 동작했다. [보고서](../results/hetero/stage0_smoke_kvfix/stage0_summary.md).

| 후보 | replay 중 최대 메모리 MiB (4090 / 3090) |
|---|---|
| S4_4090 (1 GPU) | 20,456 / 18 |
| S3_3090 (1 GPU) | 253 / 19,897 |
| S4_3090 | 12,055 / 8,271 |
| S3_4090 | 8,655 / 11,643 |
| C_4090 (PD + DiT 4090) | 20,458 / 11,645 |
| C_3090 (PD + DiT 3090) | 12,053 / 19,899 |
| C_split (PD + Denoise 4090, enc·VAE 3090) | 17,596 / 14,929 |

KV를 4 GiB로 고정하면서 1 GPU 후보의 여유가 약 4 GB로 늘었다(이전 2.7–3.2 GB). Stage 1 후보 공간은 위 7개로 확정한다. 세 서비스 동시 상주 smoke는 별도 항목으로 남긴다.

### Stage 1 실행 전 최종 검토 (2026-10-15)

사용자가 Claude와 논의하고 실험을 진행하도록 승인했다. 확정 조건과 실행 코드는 [Stage 1 프로토콜](stage1_protocol.md), 기계 판독 조건은 `results/hetero/stage1_pilot/protocol.json`을 따른다. 위 초안에서 다음 사항을 정정한다.

- 혼합 7후보 × 4개 α, 총 28회부터 수행한다. 24개 단독 프로파일을 자동으로 이어서 실행하지 않는다. 다른 서비스의 상주 배치가 다른 단독 실행은 같은 기준으로 합칠 수 없으므로 후보별 resident solo를 후속 설계한다.
- α_max는 시험한 격자 중 관측상 통과한 최대값이다. 실제 용량에 등호·상한·신뢰한계로 표시하지 않는다. 모든 α의 pass/fail/미측정과 비단조를 함께 보고한다. 동일 자원 직접 비교는 같은 GPU 종류·장수 후보끼리, 다른 구성은 명시한 가격 모델로 비교한다.
- 가격 비율은 모두 연구 가정이다. 실제 임대 시세 범위를 확보했다는 뜻이 아니다. 향후 시세는 별도 분석 버전으로 추가한다.
- 오류·timeout은 유효한 실패 결과로 보존하며 성능 결과에 따라 자동 재실행·α 생략·확장을 하지 않는다. 설정·메타데이터·요청 무결성 오류는 중단 또는 무효로 구분한다.
- 새 strict smoke gate가 13/13 완료·오류 0·필수 서버 KV 37,440 토큰·바이트·계측 일치를 확인했다. 컴파일 캐시 미스는 기존 KV 차이와 동반 관측된 유력 원인으로 기록하고, 별도 재현 없이 단독 원인으로 확정하지 않는다.
- 같은 후보 내 실제 요청 ID·도착·측정 구간·CSV 분모·출력 길이 검증, 출력별/전역 실행 lock, 실행 중 하네스 drift 검사를 추가했다. 기존 결과·실행 파일은 보존했다.

**Stage 1 시작 확인.** 2026-10-15 12:57 KST, strict smoke gate 및 CPU 검증 12건 통과 후 28개 혼합 실행을 시작했다. 첫 실행 S3_3090·α=0.5, 프로토콜 hash `981282a73e6f…`. [진행 상태](../results/hetero/stage1_pilot/batch_status.json), [중간 보고서](../results/hetero/stage1_pilot/stage1_summary.md), [확정 프로토콜](stage1_protocol.md). 아직 완료 결과가 아니며 독립 검증 seed는 실행하지 않았다.

## Stage 1 결과 (2026-10-15, seed 901, 28/28 완료) — 관측과 후속 제안의 구분

28 run 모두 done·valid, infra 실패 0, 플랜·프롬프트 해시·CSV 열·분모 parity 4개 α 전부 통과, 모든 서버 KV 37,440 토큰. 12:57 시작, 19:20 종료. 보고서 [stage1_summary.md](../results/hetero/stage1_pilot/stage1_summary.md), 셀별 결과 `results/hetero/stage1_pilot/stage1_cells.csv`. 단일 시드 탐색 결과이며 선택기 검증·솔버 개선 주장은 하지 않는다. 프로토콜·코드·실행은 이 절을 쓰면서 변경하지 않았다.

### 관측 1: 격자 위에서 통과한 셀 수 (192셀 기준)

| 후보 | α=0.5 | α=1.0 | α=1.5 | α=2.0 | α_max>0 셀 |
|---|---|---|---|---|---|
| C_4090 (PD + DiT 4090) | 76 | 44 | 0 | 0 | 76 |
| C_split (PD + DiT 양분) | 68 | 39 | 0 | 0 | 68 |
| S4_3090 (LLM 4090, DiT 3090) | 47 | 0 | 0 | 0 | 47 |
| S4_4090 (1 GPU, 4090) | 21 | 0 | 0 | 0 | 21 |
| S3_4090 (LLM 3090, DiT 4090) | 16 | 0 | 0 | 0 | 16 |
| S3_3090 (1 GPU, 3090) | 0 | 0 | 0 | 0 | 0 |
| C_3090 (PD + DiT 3090) | 0 | 0 | 0 | 0 | 0 |

α=1.5 이상에서는 어느 후보도 어느 셀도 통과하지 못했다. 비단조 셀 0. 192셀 중 98셀은 어떤 후보·α에서도 통과 없음(TTFT 0.3 s 전 셀, TPOT 12 ms의 대부분, 99% 목표의 대부분). 즉 α_max는 모든 후보에서 0.5 또는 1.0에 절단(censored)되어 있고, 실제 용량 경계는 S4_*·S3_4090은 (0.5, 1.0), PD 후보는 (1.0, 1.5) 사이 어딘가에 있다는 것만 안다.

### 관측 2: run 수준 p90 (성공 요청)과 backlog

| 후보 | α | TTFT p90 s | TPOT p90 ms | DiT E2E p90 s | backlog@600 s LLM/DiT | drain timeout LLM/DiT |
|---|---|---|---|---|---|---|
| S4_3090 | 0.5 / 1.0 / 1.5 / 2.0 | 0.31 / 0.34 / 0.41 / 0.46 | 10.0 / 14.9 / 22.4 / 35.0 | 6.6 / 31.4 / 227 / 392 | 0/0 · 0/4 · 2/75 · 2/172 | 0 · 0 · 0 · 0/74 |
| S4_4090 | 〃 | 0.43 / 0.66 / 4.34 / 20.6 | 23.0 / 53.6 / 157 / 177 | 4.1 / 8.6 / 74 / 251 | 0/0 · 1/4 · 2/30 · 64/141 | 0 |
| C_4090 | 〃 | 0.76 / 0.94 / 2.32 / 9.37 | 10.2 / 10.3 / 10.7 / 12.4 | 2.3 / 3.8 / 22.6 / 188 | 0/0 · 0/3 · 2/1 · 7/113 | 0 |
| C_split | 〃 | 0.76 / 0.95 / 2.00 / 6.36 | 10.3 / 10.6 / 11.3 / 12.3 | 2.5 / 4.0 / 22.5 / 178 | 0/0 · 0/3 · 2/1 · 7/106 | 0 |
| S3_4090 | 〃 | 0.79 / 1.00 / 4.24 / 22.3 | 26.7 / 67.1 / 166 / 186 | 2.0 / 2.6 / 3.8 / 7.4 | 0/0 · 1/2 · 2/1 · 70/3 | 0 |
| C_3090 | 〃 | 2.36 / 3.56 / 7.91 / 33.0 | 26.0 / 29.8 / 33.4 / 35.1 | 11.9 / 205 / 431 / 562 | 0/0 · 19/50 · 32/161 · 38/262 | 0 · 0 · 0/63 · 0/167 |
| S3_3090 | 〃 | 2.00 / 41.4 / 281 / 423 | 173 / 423 / 414 / 412 | 23.9 / 359 / 590 / 669 | 0/0 · 35/88 · 683/188 · 1328/285 | 0 · 0 · 179/140 · 818/237 |

- **병목은 DiT 카드의 위치다.** DiT가 3090에 있는 세 후보(S4_3090, C_3090, S3_3090)는 α=1.0부터 DiT 윈도별 in-flight가 단조 증가한다(예: S4_3090 α=1.0 w0→w2 11→4→4로 유지되나 α=1.5에서 40→55→75). DiT가 4090에 단독으로 있는 S3_4090은 α=2.0(0.6 img/s)에서도 DiT p90 7.4 s, backlog 3이지만 LLM(3090 단독)이 TTFT 22 s로 포화한다. DiT가 4090을 prefill(PD) 또는 LLM 전체와 공유하면 α=1.5~2.0에서 DiT backlog가 쌓인다(C_4090 113, S4_4090 141).
- **PD 후보는 TPOT를 전 α에서 10~12 ms로 유지한다.** decode 엔진이 3090에 단독으로 있어 간섭이 없기 때문으로 보인다(resident-solo 비교가 없어 간섭 항으로 확정하지는 않음). 대가는 TTFT로, α=0.5에서 이미 0.76 s(S4_3090 0.31 s의 2.4배)이고 α=1.5에서 2.0~2.3 s로 올라간다.
- **α=1.0에서 셀을 통과한 후보는 PD 두 개뿐이다.** 비분할 후보 중 DiT를 4090에 두는 유일한 배치(S3_4090)는 LLM이 3090에서 TPOT 67 ms로 실패하고, LLM을 4090에 두는 배치(S4_3090, S4_4090)는 DiT 또는 TPOT에서 실패한다. PD는 prefill을 3090에서 떼어내고 DiT에 4090을 주면서 decode를 격리하는 조합을 후보 공간에서 유일하게 제공한다. **단 PD 후보는 KV 8 GiB(엔진 2개)로, 이 차이가 분할 효과와 섞여 있다.** 이 시드에서 "분할이 α=1.0 셀을 유일하게 통과한다"는 관측이고, "분할의 용량 우위"는 아직 주장할 수 없다.
- **C_split과 C_4090은 사실상 같다**(통과 셀 68 vs 76, 모든 p90이 ±10% 안). DiT 단계 분할의 효과는 이 설정에서 관측되지 않았다.
- **1 GPU.** S4_4090은 α=0.5에서 21셀 통과(대략 TTFT ≥ 1.0 s·TPOT ≥ 30 ms·DiT ≥ 5 s·90%, TPOT 50 ms면 95%). 이 중 6셀은 어느 2 GPU 후보도 α 1.0에 못 미쳐 S4_4090이 비용 1.00(p_4090 단위, 점유 과금)으로 가장 싼 통과 후보다. 나머지 15셀은 PD 후보가 α=1.0을 통과하므로 α_max/비용이 r=0.75에서 1.0/1.75=0.57 vs 0.5/1.0=0.50으로 PD가 앞서지만, S4_4090의 α_max가 0.5에 절단되어 있어 결론을 내릴 수 없다. S3_3090은 α=0.5에서도 0셀이고 α=1.0에서 포화한다. 3090 단독 배치는 이 수요 범위에서 비용 절감 후보가 아니다.
- **유휴 카드 과금 포함** 시 모든 후보가 같은 비용이므로 비교는 α_max만으로 하며, 그 경우 α=1.0 셀은 PD, α=0.5 셀은 S4_3090(TTFT 0.5 s 셀 23개 단독 1위) 또는 PD가 1위다.

### 관측 3: 유효성 관련

- **플랜 아티팩트 1건.** α ≥ 1.0 플랜 3개에서 모든 후보가 LLM 오류를 정확히 1건 기록했고, 전부 같은 요청이다(idx 829, 입력 10 토큰, 출력 23 토큰 계획, 계획 시각 414.17 s, 오류 "Missing first token or completion usage"). 후보와 무관하게 재현되므로 서버 장애가 아니라 해당 프롬프트가 내용 토큰 없이 종료되는 플랜 고유 현상으로 분류한다. 분모에 1/1152~1/2336으로 포함되어 결과 판정에 영향은 없다. α=0.5 플랜에는 없다.
- **saturation suspect 열의 한계.** 현재 규칙은 "마지막 윈도 in-flight > 첫 윈도 AND drain timeout 존재"이므로, 300 s 드레인 안에 빠진 run은 backlog가 단조 증가해도 False로 표시된다(S3_3090·C_3090 α=1.0, S4_4090·C_4090·C_split α=2.0, S4_3090 α=1.5 등). 이 절의 해석은 backlog@600 s와 윈도별 in-flight 열을 직접 사용했다. 규칙 변경은 하지 않았고 후속 제안에 둔다.
- **중단 조건 미발동.** "같은 후보·α에서 drain_timeout 2회" 조건은 같은 후보·α를 반복하지 않는 설계라 발동하지 않았고, 성능 결과에 따른 자동 생략도 하지 않았다(프로토콜 그대로).
- **에너지.** 관측 600 s 동안 카드별 Wh(4090/3090): 1 GPU S4_4090 37.7→73.8 Wh(α 0.5→2.0), S3_3090 56.7→59.4 Wh(α 전 구간 거의 상수 = 포화), PD 후보 약 30/34 → 70/48 Wh. 유휴 카드는 4~5 Wh. 비용 뷰에 쓰기 전 전력 가격 가정은 프로토콜의 시나리오를 따른다.

### 다섯 가지 토론 질문에 대한 현 시점 답 (관측 기반, 잠정)

1. 분할 vs 서비스 배치 용량: PD 두 후보만 α=1.0 셀을 통과했다. 용량 우위라고 하려면 (0.5, 1.0)·(1.0, 1.5) 구간의 α 세분화와 KV 8 GiB 대조가 필요하다.
2. 어디서 이기고 무엇을 내주나: PD는 TPOT(10 ms 유지)와 DiT(4090 배정)에서 이기고 TTFT(2.4배)를 내준다. S4_3090은 TTFT에서 이기고 DiT를 내준다.
3. 유효성: 28/28 valid, parity 통과, 플랜 아티팩트 1건 식별, saturation 플래그는 보수적이라 backlog 열로 보완.
4. 1 GPU 절감 조건: 4090 단독은 α=0.5·완화 SLO(TTFT ≥ 1 s, TPOT ≥ 30 ms, DiT ≥ 5 s, 90~95%)에서만 통과하며 그중 6셀에서 유일한 최저 비용 후보. 3090 단독은 없음.
5. 새 시드로 확인할 경계: S4_4090의 TTFT 0.5 s(p90 0.425)·DiT 5 s(p90 4.10), C_*의 DiT 3 s(p90 2.3~2.5), C_*의 TTFT 1.0 s(α=1.0 p90 0.94~0.95), S4_4090·S3_4090·C_3090의 TPOT 30 ms(23~27 ms).

### 후속 실험 제안 (실행하지 않음, Codex 검토 대상)

1. **α 세분화.** α ∈ {0.6, 0.75, 0.85} (S4_*·S3_4090 경계)와 {1.1, 1.25, 1.4} (PD 경계)를 같은 seed 901로 추가해 절단을 풀기. 전체 격자 대신 통과 셀이 바뀌는 후보만 돌려도 된다(≈ 4후보 × 3 + 2후보 × 3 = 18 run).
2. **KV 대조.** 단일 엔진 후보(S4_3090, S4_4090)에 KV_BYTES=8 GiB로 α=1.0 1 run씩 추가해 PD의 α=1.0 통과가 KV 용량 때문인지 확인. 메모리 여유(S4_4090 약 4 GB)상 S4_4090은 불가할 수 있으므로 S4_3090·S3_4090만 가능할 수 있음을 사전에 기록.
3. **새 시드 검증.** seed 902, 903으로 α=0.5·1.0만(14 run/seed) 반복해 5번 항목의 경계 셀 판정이 유지되는지 확인.
4. **resident-solo 프로파일(Stage 2).** 후보별 REPLAY_ONLY=llm|dit로 α=0.5·1.0을 실행해 PD의 TPOT 유지와 S3_*의 TPOT 상승을 간섭 항으로 정량화.
5. **보고서 규칙.** saturation suspect를 "in-flight 단조 증가 AND (backlog@600 s > 임계 OR drain timeout)"로 바꾸는 안을 제안. 프로토콜 변경이므로 Codex 합의 후 다음 Stage부터 적용하고, Stage 1 결과는 그대로 둔다.
6. **플랜 점검.** idx 829 프롬프트의 응답(즉시 EOS 여부)을 서버 로그로 확인하고, 이후 플랜 생성 시 출력 길이 0인 요청을 제외할지 결정.

## Stage 1 후속: KV 대조와 α 세분화 (2026-10-25 21:58 ~ 10-26 01:34, seed 901, 18/18 완료)

사용자 지시("실행하고 그림 그려서 보여줘")로 Stage 1 결과 절의 후속 제안 1·2번을 실행했다. 파일럿 러너·프로토콜·결과는 그대로 두고, 래퍼 `bench/hetero/run_stage1_refine.py`가 기존 러너의 플랜 생성·검증·strict KV/parity·실행 루프·보고서를 그대로 호출하며 배치별 프로토콜을 따로 동결했다. 18 run 모두 done·valid, infra 실패 0, parity 통과. 플랜 아티팩트(idx 829) 1건은 α ≥ 0.75 플랜에도 동일하게 존재한다. 그림: `results/hetero/stage1_pilot/figures/fig_stage1_capacity.png`, `fig_stage1_decision.png`, `fig_stage1_kv_control.png` (스크립트 `bench/hetero/plot_stage1_map.py`, 파일럿·세분화·KV 대조 디렉터리를 합쳐 그린다).

### KV 대조 (α=1.0, 같은 플랜) — KV 크기는 결과를 바꾸지 않는다

| run | KV 합계 | TTFT p90 s | TPOT p90 ms | DiT E2E p90 s |
|---|---|---|---|---|
| C_4090 파일럿 | 8 GiB (4+4) | 0.939 | 10.3 | 3.78 |
| C_4090 대조 (`stage1_kv2g`) | 4 GiB (2+2) | 0.994 | 10.3 | 3.78 |
| S4_3090 파일럿 | 4 GiB | 0.340 | 14.9 | 31.4 |
| S4_3090 대조 (`stage1_kv8g`) | 8 GiB | 0.337 | 14.8 | 26.7 |
| S3_4090 파일럿 | 4 GiB | 0.996 | 67.1 | 2.62 |
| S3_4090 대조 (`stage1_kv8g`) | 8 GiB | 0.996 | 66.2 | 2.62 |

PD의 KV를 단일 엔진과 같은 4 GiB로 줄여도 TPOT·DiT는 동일하고 TTFT만 0.06 s 올랐다. 단일 엔진에 8 GiB를 줘도 지연이 바뀌지 않았다(S4_3090의 DiT 26.7 vs 31.4 s는 3090 DiT 포화 구간의 run 간 변동으로 본다. 별도 시드 없이는 KV 효과로 읽지 않는다). **Stage 1 결과 절의 "PD 후보는 KV 8 GiB 혼재" 유보는 해제한다.** PD의 α=1.0 통과는 KV 용량이 아니라 배치(decode 격리 + DiT 4090) 효과다. KV 토큰 수는 2 GiB 18,720 / 8 GiB 74,896으로 예측값과 일치했다(블록 1,835,008 B 가정).

### α 세분화 (KV 4 GiB) — 절단이 풀린 뒤의 용량 경계

| 후보 | α | TTFT p90 s | TPOT p90 ms | DiT E2E p90 s | 통과 셀 |
|---|---|---|---|---|---|
| S4_3090 | 0.6 / 0.75 / 0.85 | 0.31 / 0.31 / 0.31 | 10.4 / 12.2 / 12.9 | 7.9 / 9.9 / 15.8 | 52 / 31 / 20 |
| S4_4090 | 0.6 / 0.75 / 0.85 | 0.46 / 0.51 / 0.54 | 25.9 / 33.8 / 38.4 | 4.6 / 5.2 / 5.9 | 16 / 4 / 4 |
| S3_4090 | 0.6 / 0.75 / 0.85 | 0.87 / 0.88 / 0.88 | 32.3 / 39.0 / 43.7 | 2.2 / 2.5 / 2.5 | 8 / 4 / 4 |
| C_4090 | 1.1 / 1.25 / 1.4 | 1.03 / 1.22 / 1.91 | 10.4 / 10.5 / 10.6 | 4.6 / 6.1 / 12.7 | 28 / 24 / 4 |
| C_split | 1.1 / 1.25 / 1.4 | 1.03 / 1.22 / 1.62 | 10.8 / 10.9 / 11.2 | 4.8 / 6.6 / 13.3 | 26 / 20 / 3 |

관측.
- **경계 위치.** S4_4090은 TTFT 0.5 s·TPOT 30 ms·DiT 5 s를 모두 α 0.6~0.75 사이에서 넘는다(세 지표가 동시에 한계에 닿는 1 GPU 특성). S4_3090은 LLM 쪽은 0.85까지 여유(TTFT 0.31 s, TPOT 13 ms)이고 DiT 10 s를 0.75~0.85 사이에서 넘는다. PD는 TTFT 1.0 s를 1.0~1.1 사이, DiT 5 s를 1.1~1.25 사이, DiT 10 s를 1.25~1.4 사이에서 넘고 TPOT는 1.4까지 10.6 ms로 평탄하다.
- **통과 셀 수는 α에 대해 단조 감소**하며, 비단조 셀은 S4_3090의 DiT 10 s·99% 5개뿐이다(α 0.5 실패·0.6 통과·0.75 실패). 92~134 표본에서 99% 목표는 실패 1건으로 뒤집히므로 보고서 경고대로 표본 잡음으로 본다.
- **결정 지도의 동률 해소.** 격자 α 4개일 때는 통과 셀 97개 중 66개가 동률이었다. 10개 α를 합치면 고유 승자 43셀(S4_3090 25, C_4090 15, S3_4090 2, S4_4090 1), 동률 54셀 중 50셀은 C_4090과 C_split의 동률이다(두 후보는 모든 α에서 p90 차이 10% 이내로, 사실상 같은 구성). 95셀은 여전히 어느 후보도 통과하지 않는다(TTFT 0.3 s 전부, 99% 목표 대부분).
- **승자 구조.** TTFT ≤ 0.5 s 행은 S4_3090(DiT 10 s 이상에서 α 0.6~0.85), TTFT ≥ 1 s 행은 PD(α 1.0~1.4), 그 사이 TTFT 1 s·TPOT 20~50 ms·DiT ≥ 10 s 구간은 S4_3090이 PD를 α 0.85 vs 0.5로 이긴다(PD가 TTFT 1 s를 95% 이상으로 맞추지 못하기 때문). 비용 보정(r=0.75, 점유 카드)에서는 1 GPU 4090이 TTFT 1~2 s·TPOT 30~50 ms·DiT ≥ 10 s 셀 14개에서 승자가 된다(α 0.6~0.85를 비용 1.0으로 달성).
- **주장하지 않는 것.** 단일 시드이므로 경계값의 신뢰구간은 없다. 세분화 α에서도 상주형 단독 run이 없어 간섭 항은 분리되지 않았다.

### 다음 단계

Stage 1 결과 절의 후속 제안 3~6번이 남아 있다. 순서 제안: (3) seed 902·903에서 경계 셀 재검증(α 0.5·0.6·0.75·1.0·1.1·1.25, 승자 후보 4개 = 48 run, 약 11 h) → (4) 상주형 단독 프로파일로 선택기 비교(Stage 2). 그 전에 Codex 검토: KV 유보 해제의 근거 충분성, 비단조 5셀 처리, 결정 지도의 동률 표기 방식.

## Seed 검증과 Stage 2 resident-solo 시작 (2026-10-27)

사용자 지시("계획대로 진행")로 위 "다음 단계"의 (3)·(4)를 순차 체인으로 시작했다. 결과 전이며 아래는 실행 조건 기록이다.

- **Seed 검증** (`results/hetero/stage1_seed902`, `stage1_seed903`, 10:17 시작): 승자 후보 4개(S4_4090, S4_3090, S3_4090, C_4090) × α {0.5, 0.6, 0.75, 1.0, 1.1, 1.25} × seed 2 = 48 run. KV 4 GiB, 파일럿과 같은 규칙, `run_stage1_refine.py`에 `--seed` 인자만 추가했다. C_split은 C_4090과 모든 α에서 p90 차이 10% 이내라 제외했다. 플랜 검증 통과. seed 902 α=0.5의 DiT 측정 요청은 65건으로 최소 기준 60건에 가깝다(같은 평균 도착률에서 포아송 변동). 비교 스크립트 `bench/hetero/plot_stage1_seeds.py`는 후보별로 세 시드 모두에서 시험한 α만 써서 α_max와 셀별 승자 일치를 센다.
- **Stage 2 resident-solo** (`results/hetero/stage2_solo_llm`, `stage2_solo_dit`, seed 검증 종료 후 자동 시작): 7 후보 × α {0.5, 1.0} × {LLM만, DiT만} = 28 run, seed 901 플랜 그대로. 새 래퍼 `bench/hetero/run_stage2_solo.py`가 기존 러너를 쓰되 실행 시점에만 REPLAY_ONLY를 넣고, 무결성 검사를 재생 종류의 요청으로 한정한다. solo prompt 해시 규칙과 무결성 검사는 Stage 0 solo smoke 결과로 일치를 확인했다.
- **선택기 비교** (`bench/hetero/stage2_report.py`): solo 그대로(간섭 없음 가정), solo+k(혼합/solo p90 비율로 지연 보정, seed 901 안에서는 다른 α에서 맞춘 k, 새 시드에서는 같은 α의 seed-901 k), seed-901 혼합 측정, 고정 배치 2종(C_4090, S4_3090), 오라클. 평가 두 가지: 셀별 최대 α 선택(오라클 일치·regret·과대 약속)과 α별 최소 비용 선택(혼합에서 SLO 위반·과지불·놓침). 혼합 run을 solo 자리에 넣은 자기 시험에서 예측이 혼합과 완전히 일치했다(fp = fn = 0, regret 0).

## Seed 검증 결과 (2026-10-27, seed 902·903 48/48 완료)

두 배치 모두 complete 24/24, 인프라 실패 0. 산출물: `results/hetero/stage1_validation/` (`fig_seed_latency`, `fig_seed_winners`, `seed_validation.md/.json`). 후보별로 세 시드 모두에서 시험한 α만 비교했다(S 후보 0.5·0.6·0.75·1.0, C_4090 0.5·1.0·1.1·1.25. seed 901에서 C_4090을 0.6·0.75로 돌리지 않아 공통 격자가 다르다).

| 후보 | α | p90 TTFT s (901/902/903) | p90 TPOT ms | p90 DiT E2E s |
|---|---|---|---|---|
| S4_4090 | 0.75 / 1.0 | 0.51·0.51·0.58 / 0.66·0.77·1.00 | 34·32·43 / 54·64·88 | 5.2·4.5·3.7 / 8.6·6.4·14.9 |
| S4_3090 | 0.75 / 1.0 | 0.31·0.34·0.35 / 0.34·0.34·0.38 | 12·13·14 / 15·16·17 | 9.9·8.0·7.0 / 31.4·12.2·32.4 |
| S3_4090 | 0.75 / 1.0 | 0.88·0.90·1.08 / 1.00·1.12·1.35 | 39·51·86 / 67·97·127 | 2.5·2.3·2.3 / 2.6·2.6·3.3 |
| C_4090 | 1.0 / 1.25 | 0.94·0.99·1.21 / 1.22·1.30·1.41 | 10·10·11 / 11·11·11 | 3.8·4.0·5.9 / 6.1·5.0·10.7 |

관측.
- **구조는 유지된다.** 세 시드 모두에서 같은 후보가 같은 지표로 먼저 한계에 닿는다. S4_3090은 LLM이 α 1.25까지 TTFT ≤ 0.40 s·TPOT ≤ 24 ms이고 DiT가 먼저 포화된다. C_4090은 TPOT 10~11 ms로 평탄하고 TTFT와 DiT가 한계를 정한다. S3_4090은 DiT ≤ 3.8 s이고 LLM TPOT이 한계다. S4_4090은 세 지표가 함께 오른다.
- **경계 α는 격자 한 칸 정도 움직인다.** 예: C_4090의 TTFT 1 s 경계는 seed 901·902에서 α 1.0~1.1 사이, seed 903에서는 1.0 이전. S4_3090의 DiT 10 s 경계는 0.75~0.85(901), 0.75~1.0(902·903). S3_4090의 TPOT은 시드 간 차이가 가장 크다(α 0.75에서 39 / 51 / 86 ms).
- **시드 차이의 원인 후보는 실현 부하.** 같은 공칭 α에서도 600 s 창의 측정 요청 수가 다르다. α 1.0에서 LLM 요청 1152 / 1217 / 1294건, 출력 토큰 합 30.4k / 35.7k / 38.6k, DiT 요청 166 / 151 / 179건. seed 903이 LLM 기준 12%(토큰 27%) 더 무거워서 지연도 가장 높다. 확인하려면 실현 도착률로 α를 다시 매긴 비교가 필요하다(제안, 미실시).
- **셀 판정의 일치.** 후보×α×셀 판정 3072개 중 218개(7%)가 시드에 따라 통과/실패가 바뀐다. 어느 시드에서든 승자가 있는 105셀 중 64셀은 세 시드의 승자가 같다(비용 보정 r=0.75: 67셀). 불일치 41셀의 분류는 다음과 같다.
  - 16셀: 한 시드에서는 통과 후보가 없다(가능 영역 경계).
  - 14셀: 단독 승자와 그 후보를 포함한 동률. 대부분 C_4090의 공통 격자에 0.6·0.75가 없어서 생긴 동률이다.
  - 11셀: 서로 다른 단독 승자.
- 불일치는 90% 목표(46셀 중 23)와 DiT 3·5 s, TTFT 1~2 s 열에 몰린다. α 0.5 근처에서 C_4090과 S 후보가 겹치는 구간이다. TTFT ≤ 0.5 s × DiT ≥ 10 s 영역(S4_3090)과 TTFT 2 s × DiT ≥ 10 s 영역(C_4090)은 세 시드 모두 같다.
- **주장하지 않는 것.** 시드 3개로는 경계 α의 구간 추정이 거칠다. 동률 14셀은 격자 차이로 생긴 것이라 실제 동률로 보지 않는다.

## Stage 2 결과: resident-solo와 선택기 비교 (2026-10-28, 28/28 완료)

두 배치 모두 complete 14/14, 인프라 실패 0이다. 산출물은 `results/hetero/stage2_report/` (`stage2_summary.md/.json`, `fig_stage2_interference`, `fig_stage2_selectors`)에 있다. 평가 α는 solo 프로파일이 있는 0.5·1.0 두 개뿐이다. 새 시드 평가도 이 두 α에서만 했다. 비용은 r=0.75, 점유 카드 기준이다.

**간섭 (혼합 p90 / solo p90, 같은 배치·같은 플랜, seed 901)**

| 배치 유형 | 후보 | TTFT | TPOT | DiT E2E |
|---|---|---|---|---|
| 서비스가 다른 카드 | S4_3090, S3_4090 | 0.99–1.02 | 0.99–1.01 | 1.00–1.07 |
| PD, DiT가 prefill과 4090 공유 | C_4090, C_split | 1.10–1.26 | 1.00–1.03 | 1.16–1.48 |
| PD, DiT가 decode와 3090 공유 | C_3090 | 3.5–4.7 | 2.6–2.9 | 1.8–6.9 |
| 한 카드에 둘 다 | S4_4090 | 1.3–2.0 | 2.3–3.6 | 2.0–3.3 |
| 한 카드에 둘 다 | S3_3090 | 2.5–41.7 | 6.4–6.5 | 3.6–14.7 |

관측.
- **간섭은 배치에 따라 0에서 수십 배까지 달라진다.** 서비스를 다른 카드에 두면 간섭이 없다(비율 1.0 ± 0.07). PD에서 DiT가 prefill 카드를 공유하면 TTFT와 DiT만 1.1–1.5배 늘고, decode가 다른 카드에 있으니 TPOT은 그대로다. DiT가 decode 카드를 공유하면(C_3090) TPOT까지 2.6–2.9배 늘어난다. 한 카드 배치는 α가 커질수록 비율도 커진다(S4_4090 TPOT 2.3 → 3.6, S3_3090 TTFT 2.5 → 41.7).
- **간섭은 α에 대해 일정하지 않다.** 간섭이 있는 모든 후보에서 α 1.0의 비율이 α 0.5보다 크다. 그래서 상수 하나로 보정하면 α를 벗어나 외삽할 때 틀린다.

**선택기 비교**

| 평가 | 선택기 | 최대 α 오라클 일치 | regret Σα | 최소 비용: SLO 위반 / 놓침 / 과지불 (경우 수) | false-feasible |
|---|---|---|---|---|---|
| seed 901 | solo | 62/119 | 34.0 | 93 / 0 / 0 (138) | 70 |
| seed 901 | solo+k (다른 α의 k) | 80/94 | 7.0 | 6 / 18 / 15 (138) | 16 |
| seed 901 | 고정 C_4090 | 76/94 | 9.0 | 18 / 0 / 18 (138) | 246 |
| seed 901 | 고정 S4_3090 | 23/94 | 45.5 | 91 / 0 / 14 (138) | 246 |
| seed 902 | solo | 68/123 | 28.5 | 70 / 4 / 0 (156) | 56 |
| seed 902 | solo+k (같은 α의 seed-901 k) | 97/105 | 4.0 | 0 / 17 / 25 (156) | 0 |
| seed 902 | 혼합 seed 901 | 96/105 | 4.5 | 0 / 18 / 25 (156) | 0 |
| seed 902 | 고정 C_4090 | 74/105 | 18.0 | 36 / 0 / 39 (156) | 228 |
| seed 903 | solo | 80/119 | 21.5 | 60 / 0 / 0 (106) | 102 |
| seed 903 | solo+k (같은 α의 seed-901 k) | 93/96 | 1.5 | 5 / 1 / 2 (106) | 34 |
| seed 903 | 혼합 seed 901 | 92/95 | 1.5 | 5 / 1 / 2 (106) | 33 |
| seed 903 | 고정 C_4090 | 76/90 | 7.0 | 14 / 0 / 22 (106) | 278 |

관측.
- **solo 프로파일만 쓰면 안전하지 않다.** 오류가 모두 같은 방향(과대 약속)이다. 최소 비용 선택에서 SLO 위반이 seed 901 93/138, seed 902 70/156, seed 903 60/106이다. 혼합에서 아무도 통과하지 못하는 경우에도 배치를 내놓는다(false-feasible 56–102). 위반은 공유 배치(S4_4090, C_split, C_3090)를 싸거나 충분하다고 잘못 판단한 데서 나온다.
- **간섭 보정을 하면 대부분 회복된다.** seed 901에서 보정 계수 k를 다른 α에서 가져오면(α 두 개라 0.5↔1.0 외삽) 위반은 93 → 6으로 줄지만 놓침 18이 생긴다. α 1.0의 k를 0.5에 쓰면 간섭을 과대 추정해 가능한 배치를 놓치고(fn 쪽), 반대로 쓰면 과소 추정해 위반한다(fp 쪽). 앞 관측대로 k가 α에 따라 커지기 때문이다.
- **새 시드에서 같은 α의 k는 seed-901 혼합 측정과 거의 같은 결과를 낸다.** 이 비교가 보여주는 것은 seed 901에서 얻은 프로파일과 간섭 크기가 다른 도착열에도 옮겨간다는 점이다. k가 α 사이로 일반화된다는 근거는 아니다. seed 903의 위반 5건과 false-feasible 34는 seed 903이 실현 부하가 더 무거운 데서 온다(Seed 검증 결과 절 참고).
- **고정 배치는 어느 쪽도 오라클에 못 미친다.** 고정 C_4090은 최대 α 일치가 74–76셀이고, 비용 최소화에서는 위반 14–36에 과지불 18–39가 붙는다. 고정 S4_3090은 23–38셀이다. 셀마다 다른 배치를 골라야 하는 경우가 실제로 있다는 근거다.
- **주장하지 않는 것.** 이 결과를 solver 개선으로 주장하지 않는다. α 두 개, 후보 7개(새 시드는 4개), 단일 모델 쌍이다. solo+k는 혼합을 한 번 돌려야 k를 얻으므로 "혼합 측정 없이"라는 주장은 할 수 없다. 남는 문제는 간섭 크기를 혼합 측정 없이 예측할 수 있느냐다.

### 다음 단계 제안 (미실시)
1. **α 의존 간섭 모델.** 간섭 비율을 공유 카드의 실현 부하(예: DiT 점유율 × LLM 토큰률)의 함수로 맞추고, seed 901의 세분화 α(0.6·0.75·0.85·1.1·1.25)에서 solo를 몇 개 더 돌려 외삽 오차를 본다. 대상은 간섭이 있는 S4_4090, C_4090 정도로 줄인다.
2. **실현 도착률로 α 재지정.** 시드 간 차이가 실현 부하로 설명되는지 확인한다. 추가 실행은 필요 없다.
3. **배치 유형 규칙.** "서비스가 다른 카드면 k=1"은 이번 데이터로 성립한다. 공유 배치에만 측정이 필요하다는 식으로 프로파일 비용을 줄일 수 있는지 정리한다.
4. 이후 계획에 있던 수요 비율 축과 VLM 통합.

## Stage 2b: 간섭 모델 외삽 검증용 solo 추가 (2026-11-02 시작)

사용자 지시("Solver를 썼을 때와 안 썼을 때 비교", "적당히 실험해봐")로 아래를 시작했다. 목적은 정답(같은 α의 혼합 결과)을 보지 않는 solver를 정의하는 것이다. 지금의 solo+k는 같은 α의 혼합 p90으로 k를 구하므로 공정한 비교가 아니다.

- **실행** (`results/hetero/stage2_solo_llm_b`, `stage2_solo_dit_b`, 체인 `results/hetero/stage2_b/chain.sh`, 08:39 시작): 간섭이 있으면서 결정 지도에서 이기는 두 배치 S4_4090(α 0.6·0.75·0.85·1.5)과 C_4090(α 1.1·1.25·1.4·1.5)의 resident-solo. 2 × 4 × {LLM만, DiT만} = 16 run, 약 4.5 h. seed 901 플랜 그대로이고 혼합 실행은 추가하지 않는다(해당 α의 혼합은 파일럿·세분화 배치에 있다). 프로토콜 cd8c20a8fbb5(LLM), 740c93f76087(DiT). 다른 카드 배치(S4_3090, S3_4090)는 k=1이 확인돼 제외, S3_3090·C_3090은 어느 셀에서도 이기지 않아 제외.
- **비교 설계.** solver 없음: 고정 S4_4090, 고정 S4_3090, 고정 C_4090, solo만(간섭 무시). solver 있음: solo + k(α), k는 배치당 앵커 α {0.5, 1.0}의 혼합/solo 비율로만 맞추고 다른 α로 외삽. 다른 카드 배치는 k=1. 상한은 오라클. 평가는 seed 902·903의 α 0.6·0.75·1.1·1.25(앵커 α와 다르고 k를 맞춘 seed와도 다름). 지표는 기존 stage2_report와 같다(최대 α 일치·regret, 최소 비용의 위반·과지불·놓침).
- **먼저 볼 것.** 배치당 α 6개의 k가 α에 대해 어떤 모양인지. 단조 증가이고 두 앵커로 외삽이 되면 "혼합 앵커 2회"가 결론이 되고, 안 되면 앵커 개수가 결론이 된다.
- **선택 사항.** solver와 기본값의 선택이 갈리는 시나리오 5~6개를 seed 904에서 실제로 배포해 비교. k(α) 모양을 보고 결정.

## Stage 2b 결과: 앵커 2개 k 외삽과 solver 유무 비교 (2026-11-02, 16/16 완료)

두 배치 complete 8/8, 인프라 실패 0 (08:39~11:48). 산출물 `results/hetero/stage2b_report/` (`stage2b_summary.md/.json`, `fig_stage2b_k`, `fig_stage2b_selectors`, `fig_stage2b_selectors_explained`). 분석 `bench/hetero/stage2b_report.py`, 발표용 그림 `bench/hetero/plot_stage2_explained.py --stage2b`.

**k(α) 측정 (seed 901, 혼합 p90 / solo p90)**

| 후보 | α | k TTFT | k TPOT | k DiT |
|---|---|---|---|---|
| S4_4090 | 0.5 / 0.6 / 0.75 / 0.85 / 1.0 / 1.5 | 1.34 / 1.47 / 1.62 / 1.71 / 1.97 / 10.5 | 2.31 / 2.49 / 2.80 / 2.99 / 3.62 / 7.29 | 2.02 / 2.16 / 2.15 / 2.33 / 3.29 / 20.8 |
| C_4090 | 0.5 / 1.0 / 1.1 / 1.25 / 1.4 / 1.5 | 1.10 / 1.26 / 1.30 / 1.44 / 2.16 / 2.53 | 1.00 (전 구간) | 1.16 / 1.45 / 1.67 / 2.08 / 3.89 / 6.33 |

관측.
- **앵커 사이·근처에서는 선형으로 충분하다.** 앵커 α {0.5, 1.0}의 k로 그은 직선이 S4_4090 α 0.6~0.85의 k를 TTFT·TPOT ±8%, DiT +5~25%(과대 추정 = 보수적) 안에서 맞힌다. C_4090 α 1.1은 0%/−9%, 1.25는 −7%/−23%(과소 추정).
- **포화 무릎을 지나면 어떤 2점 모델도 맞지 않는다.** C_4090 α 1.4 이후 k가 급증(TTFT 1.44→2.16, DiT 2.08→3.89)하고 선형 모델은 36~78% 과소 추정한다. S4_4090 α 1.5는 k 7~21로, 1.0까지의 추세와 무관하다. 이 구간은 이미 거의 모든 셀이 실패하는 영역이지만, solver가 여기까지 외삽하면 과대 약속이 된다. 유효 범위는 앵커 상한의 약 1.25배까지다.
- C_4090의 TPOT k는 모든 α에서 1.00이다. decode가 다른 카드에 있어서 DiT 공유의 영향이 TTFT와 DiT에만 간다.

**seed 901 held-out α에서 셀 예측 (공유 후보, 192셀 × α 4개 = 768판정)**

| 예측기 | S4_4090 fp / fn | C_4090 fp / fn |
|---|---|---|
| solo만 | 261 / 0 | 156 / 0 |
| solo + k 선형(앵커 2) | 2 / 0 | 52 / 0 |
| solo + k 상수(가까운 앵커) | 18 / 0 | 56 / 0 |
| solo + k 같은 α 측정 | 4 / 0 | 0 / 0 |

C_4090의 fp 52는 전부 α 1.4·1.5(무릎 이후)에서 나온다.

**solver 유무 비교 (seed 902+903 합산, held-out α 0.6·0.75·1.1·1.25, 후보 4개)**

| 선택기 | 최대 α 일치 | 최소 비용 383건: ok / 위반 / 과지불 / 놓침 | false-feasible |
|---|---|---|---|
| solo만 (간섭 무시) | 153/224 | 142 / 237 / 0 / 4 | 329 |
| **solver = solo + k 선형(앵커 2)** | **140/180** | **272 / 36 / 10 / 75** | 74 |
| solo + k 같은 α 측정 (k 상한) | 138/180 | 266 / 40 / 10 / 77 | 62 |
| seed 901 혼합 재사용 | 140/180 | 272 / 32 / 10 / 79 | 62 |
| 오라클 | 168/168 | 383 / 0 / 0 / 0 | 0 |
| 고정 S4_4090 | 9/168 | 45 / 338 / 0 / 0 | 1153 |
| 고정 C_4090 | 125/168 | 310 / 73 / 39 / 0 | 1153 |
| 고정 S4_3090 | 54/168 | 156 / 227 / 30 / 0 | 1153 |

관측.
- **앵커 2개로 외삽한 k는 같은 α에서 잰 k와 같은 성능을 낸다**(272 vs 266 ok, 36 vs 40 위반). 또 seed 901 혼합 결과를 통째로 재사용한 것과도 같다(272/32). 즉 α 0.6~1.25 범위에서는 배치당 혼합 2회 + solo 프로파일이 "모든 α에서 혼합"과 같은 정보를 준다. 남는 오차는 k가 아니라 seed 차이와 프로파일 격자에서 온다.
- **solver vs 최선의 고정 배치(C_4090).** 위반 36 vs 73(절반), 과지불 10 vs 39. 대신 solver는 75건(20%)에서 "가능한 배치 없음"이라 답한다. 놓침 75건의 내역: 실제로는 C_4090이 가능했던 경우 53건(모두 α 0.6·0.75. seed 901에 C_4090 solo가 α 1.0 아래로 없어 보수적 규칙이 α 1.0 프로파일을 썼기 때문), S4_3090이 가능했던 경우 17건(seed 901에서 실패한 경계 셀이 seed 902·903에서 통과. 시드 차이).
- **위반 36건은 전부 α 0.6·0.75에서, 선택된 후보는 S4_3090 16, S4_4090 14, S3_4090 6이다.** S4_3090·S3_4090은 k=1로 seed 901 혼합을 그대로 쓰므로 이 위반은 k 오차가 아니라 시드 차이(seed 903의 실현 부하가 12% 큼)다. α 1.1·1.25에서는 위반 0.
- **고정 배치는 "안 된다"고 말할 수 없다.** 실제로 어느 배치도 못 지키는 (셀, α) 1153건에서 고정 배치는 전부 배포한다. solver는 74건(6%)만 잘못 배포한다.
- **solo만 쓰는 선택기는 여전히 위반 62%**로, 간섭 모델 없이는 쓸 수 없음을 재확인했다.

**주장할 수 있는 것과 없는 것**
- 할 수 있는 것: 이 모델 쌍·이 풀에서, 공유 배치는 혼합 앵커 2회(α 0.5·1.0)와 solo 프로파일, 분리 배치는 solo 프로파일만으로, 처음 보는 수요(α 0.6~1.25)와 도착열에서 배치를 고를 때 위반 9%, 과지불 3%, 놓침 20%였다. 고정 배치 중 최선은 위반 19%, 과지불 10%였다.
- 할 수 없는 것: 앵커 상한의 1.25배를 넘는 외삽(무릎 이후). 다른 수요 비율(LLM:DiT), 다른 모델 쌍, 3개 이상 서비스. 놓침 20%는 프로파일 격자 보강 전의 수치다.

### 다음 단계 제안 (미실시)
1. **C_4090 solo α 0.6·0.75 추가** (2 α × 2 kind = 4 run, 약 1 h). 놓침 75건 중 53건이 이 격자 공백에서 나오므로, 채우면 solver의 놓침이 약 6%로 내려갈 것으로 예상한다. 예상이지 결과가 아니다.
2. **무릎 감지 규칙.** 앵커 상한 × 1.25를 넘는 α는 "외삽 불가"로 표시하고 앵커를 추가하게 한다. 코드 변경 없이 보고서 규칙으로 먼저 둔다.
3. **실현 부하로 α 재지정.** 위반 36건이 시드 차이에서 오므로, k를 공칭 α가 아니라 실현 부하(공유 카드의 토큰률·DiT 요청률)의 함수로 두면 줄어드는지 본다. 추가 실행 없음.
4. 수요 비율 축, VLM 통합은 그대로 후순위.

## 탐색: 단독 프로파일만으로 k를 계산할 수 있는가 (2026-11-02, GPU 실행 없음)

사용자 질문("예측은 수학적으로 하면 되잖아")에 대한 1차 시험. 스크립트 `bench/hetero/analyze_k_model.py`, 산출물 `results/hetero/k_model/` (`k_model.md/.json`, `fig_k_model.png`). 기존 resident-solo 실행의 1 Hz nvidia-smi 기록과 요청 레코드만 썼다.

**설정.** 공유 배치 5개(S4_4090, S3_3090, C_4090, C_3090, C_split) × α(0.5·1.0, 세분화 포함)에서 실측 k를 정답으로 두고, 단독 실행에서만 얻는 특징으로 k를 예측한다. 특징은 지표별 단계가 도는 GPU에서 (1) 상대 서비스의 점유율 u_other, (2) 자기 점유율 u_own, (3) 단계 길이 비 r = L_other/L_own(decode 스텝 = 단독 TPOT p50, DiT 스텝 = 단독 E2E p50/28, prefill = 단독 TTFT p50). TTFT는 PD에서 prefill GPU와 decode GPU를 함께 본다(첫 토큰은 KV 전송 뒤 decoder가 낸다). 모델은 지표별 1~2 파라미터: LIN k = 1 + a·u, PS k = 1/(1 − a·u), PSG = PS + b·u·r. 평가는 배치 하나를 빼고 맞춘 뒤 그 배치를 예측(leave-one-placement-out), α ≤ 1.25만 적합하고 그 위는 외삽 점검.

**결과 (39점, 배치 5개).**

| 모델 | 중앙값 \|상대오차\| | p90 | 25% 이내 | 지표별 중앙값 TTFT / TPOT / DiT |
|---|---|---|---|---|
| LIN | 37% | 94% | 13/39 | 31% / 23% / 45% |
| PS | 24% | 73% | 20/39 | 30% / 50% / 8% |
| PSG | 23% | 71% | 22/39 | 23% / 50% / 9% |

관측.
- **구조 규칙은 성립한다.** 공유가 없는 단계·지표 쌍 21개의 k는 0.99~1.07이다(C_3090의 TTFT만 예외였는데, 이는 PD의 첫 토큰이 decode GPU를 거치기 때문이며 매핑을 고치자 설명됐다).
- **DiT E2E는 점유율만으로 ±10% 안에 맞는다.** PS, a = 1.45. 포화 전 15점 중 13점이 ±12% 이내. DiT 스텝이 길어서 시간 분할(time-sharing) 가정이 맞는 것으로 본다.
- **공존 배치(S4_4090)의 TTFT도 ±12% 안에 맞는다.** PD 배치의 TTFT는 20~57% 과대 예측한다. prefill은 긴 커널이라 같은 u에서도 덜 느려지고, TTFT에 KV 전송처럼 간섭과 무관한 항이 섞여 있다.
- **TPOT은 맞지 않는다.** 공존 배치에서 실측 2.3~3.6을 1.2~1.6으로 50% 과소 예측한다. decode 스텝(10 ms)이 DiT 스텝(70 ms)보다 훨씬 짧아 스텝마다 DiT 커널 뒤에서 기다리는 효과를 평균 점유율이 담지 못한다. 단계 길이 비 r을 더한 PSG도 LOO에서는 개선이 없었다. 공존 단일 엔진(S4_4090: u 0.19~0.34, k 2.3~3.6)과 PD decoder(C_3090: u 0.45~0.83, k 2.6~2.9)에서 u와 k의 관계가 서로 달라 배치 간에 전이되지 않는다. 공존 단일 엔진에서는 decode가 자기 prefill과도 경쟁하는데 이 항이 모델에 없다.
- **포화는 어떤 모델도 못 맞힌다.** u_own + u_other가 1에 가까워지는 점(S3_3090 α 1.0, C_3090 α 1.0, S4_4090 α 1.5)에서 실측 k 7~42를 2~5로 예측하거나 극으로 발산한다.
- **앵커 1개 + PS 꼴 외삽**(α 0.5에서 a를 맞추고 u(α)로 외삽)은 TTFT에서는 2-앵커 선형과 비슷하지만(±8%), TPOT·DiT에서는 α 1.0에서 +150% 이상 벗어난다. 2-앵커 선형이 현재로서는 가장 안정적이다.

**판단.** nvidia-smi의 평균 점유율 수준 특징으로는 DiT와 공존 TTFT만 solver 등급(±10%)이고, TPOT·PD TTFT·포화는 안 된다. "측정 0회"는 지금 데이터로는 주장할 수 없다. 측정 2회(2-앵커 선형)가 현 시점의 결론이다.

**다음 후보 (미실시).**
1. 더 세밀한 단독 프로파일: 커널 단위 타임라인(Nsight Systems 또는 CUPTI)으로 각 단계의 스텝 길이 분포와 SM 점유를 재고, 두 서비스의 커널 열을 겹쳐 대기 시간을 계산하는 시뮬레이션형 모델. 측정은 서비스당 1회이므로 "0회"에 해당한다. GPU 실행 필요(짧음).
2. 포화 항: 혼합 시 총 부하 ρ = u_own·k + u_other가 1에 닿는 지점을 무릎으로 두고, 그 전까지만 예측하고 그 뒤는 "불가"로 답하는 규칙. 기존 데이터로 검증 가능.
3. 하이브리드: 구조 규칙(공유 없음 → k=1, decode 분리 → TPOT k=1)으로 측정 대상을 줄이고, 공유된 단계에만 앵커를 쓴다. 현재 구현이 이미 이 형태다.

## Nsight 진단: 공유 GPU에서 TPOT이 왜 2.8배가 되는가 (2026-11-02, C_3090 α 1.0, 캡처 3회)

**목적.** 분할 기법의 장점이 흐려지는 원인이 연산 자체의 둔화(SM/대역폭 경쟁)인지, 프로세스 간 GPU 시간 분할인지, 큐잉인지를 커널 단위로 확인한다. 대상은 C_3090(P/D 분할, DiT를 decode GPU인 3090에 둠), α 1.0, seed 901 플랜. 프로토콜 실행이 아니라 진단이며, 재생을 캡처 뒤 끊었으므로 요청 기록은 부분값이다.

**설정.** `bench/hetero/profile_mixed_nsys.sh {mixed|llm|dit}`. 서버·포트·KV 4 GiB·플랜은 run_mixed_modal.sh와 같다. 3090의 두 프로세스(decode vLLM :8200, dit_server :8300)를 하나의 `nsys launch` 세션(--trace=cuda, --cuda-graph-trace=graph)으로 띄우고, 재생 시작 90 s 뒤 90 s를 캡처한다. GPU 메트릭 샘플링은 root 없이 막혀 있어(RmProfilingAdminOnly=1) `nvidia-smi dmon -s um`을 대역폭 대용으로 같이 기록했다. 분석은 `bench/hetero/analyze_nsys.py`(sqlite export), 타임라인은 `bench/hetero/plot_nsys_timeline.py`. 결과는 `results/hetero/nsys_c3090/{mixed,llm,dit}/`. 그래프 추적 기록은 커널 기록 창보다 넓게(재생 시작부터) 남아 있어 커널 창으로 잘라서 계산했다.

**재생 기록 sanity(부분값).** TPOT p50 LLM 단독 9.4 → 혼합 25.4 ms, denoise p50 DiT 단독 2.85 → 혼합 4.17 s. 프로토콜 실행값(9.4 → 26.6 ms, 2.87 → 3.89 s)과 같은 방향·크기이므로 진단 조건이 프로토콜 조건을 재현한다.

**결과 (90 s 창).**

| 항목 | LLM 단독 | DiT 단독 | 혼합 |
|---|---|---|---|
| decode 스텝(CUDA graph ≥ 2 ms) p50/p90 | 8.54 / 9.47 ms | – | 18.29 / 19.34 ms (2.14×) |
| 스텝 사이 간격 p50 | 0.98 ms | – | 5.63 ms |
| decode 비그래프 커널 p50 비 (혼합/단독) | – | – | 1.0~1.2 (gemm 1.01, sampler 1.19) |
| decode 배치 크기 (vLLM Running) | 0~2 | – | 0~4 |
| decode가 GPU를 점유한 시간 비율 | 0.376 | – | 0.496 |
| DiT 커널 수 / 90 s | – | 2,063,436 | 1,397,696 (0.68×) |
| DiT 커널 p50 (elementwise / gemm 128×128) | – | 18.1 / 148 µs | 18.0 / 146 µs (동일) |
| DiT 커널 평균 | – | 42.1 µs | 61.6 µs (1.46×) |
| DiT 커널 중 1 ms 초과 비율 / 그 시간 비중 | – | 0.03% / 1% | 0.77% / 31% |
| DiT 커널이 실행 중으로 기록된 시간 비율 | – | 0.965 | 0.957 |
| decode 점유 시간 중 DiT 커널도 기록된 비율 | – | – | 0.944 (42.1 s / 44.6 s) |
| 두 프로세스 합집합 점유 비율 | – | – | 0.984 |
| dmon GPU1 sm% / mem% 평균 | 40 / 36 (decode 구간 mem 97) | 100 / 42 | 100 / 54 (p90 63) |

관측.
- **실행 중인 커널은 느려지지 않는다.** DiT 커널의 p50은 혼합과 단독이 같고(18.0 vs 18.1 µs, gemm 146 vs 148 µs), decode의 비그래프 커널도 1.0~1.2배다. SM이나 DRAM 대역폭을 동시에 나눠 써서 느려지는 양상이 아니다(dmon mem%도 54%로 포화가 아니다. LLM 단독의 decode 구간은 mem 97%인데, 이는 decode 스텝 자체가 대역폭 한계로 도는 것이며 혼합에서는 시간이 분할되어 평균이 내려간다).
- **길이가 늘어나는 것은 "다른 프로세스가 GPU를 가져간 구간"이 끼어든 커널·그래프뿐이다.** DiT 커널의 0.77%만 1 ms를 넘지만 그 시간이 DiT 레인의 31%를 차지한다. 늘어난 길이(혼합 길이 − 단독 p50)의 분포는 0.4~0.6 ms와 2.3~2.6 ms의 두 봉우리로, decode 쪽 슬라이스 길이에 대응한다. decode 그래프도 같은 일을 당해 8.5 ms 작업이 18.3 ms 벽시계가 된다(2.14×). 두 프로세스 기록의 합집합 점유가 0.984로 1을 넘지 않는 것과도 맞는다: GPU는 두 컨텍스트를 ms 단위로 번갈아 실행하고, 각자의 기록에는 상대 차례가 대기 시간으로 포함된다.
- **배치 크기는 원인이 아니다.** 두 경우 모두 Running 0~4개이고, decode 스텝 분포가 좁다(p50 18.3, p90 19.3). 커널 단위 비교에서도 attention 커널(splitkv 21 → 36 µs)만 배치 차이로 조금 길어졌을 뿐 절대량은 15 µs다.
- **vLLM 내부 큐잉은 원인이 아니다.** Waiting 0~4, KV 사용률 최대 46%. 큐잉은 DiT 쪽에서 결과로 나타난다: DiT는 단독으로도 GPU 점유 0.965(dmon sm 100%)라 여유가 없고, 공유로 커널 처리량이 0.68배가 되면 denoise가 1.36~1.46배 길어지고 대기열이 쌓인다(프로토콜: 큐 7.7 → 170 s).
- **TPOT 2.8배의 분해.** 스텝 8.5 → 18.3 ms(+9.8 ms, DiT에 넘어간 슬라이스)와 스텝 사이 간격 1.0 → 5.6 ms(+4.6 ms, 샘플링·동기화 커널과 CPU 측 대기도 DiT 슬라이스 뒤로 밀림). 합 9.5 → 23.9 ms, 실측 TPOT p50 9.4 → 25.4 ms.

**판단(관측 수준).** 공유 GPU에서 분할 기법의 장점이 사라지는 1차 원인은 **프로세스(CUDA 컨텍스트) 간 GPU 시간 분할**이다. 연산 효율 저하(커널 둔화)는 측정되지 않았고, 큐잉은 DiT가 단독으로도 포화 근처였기 때문에 생기는 2차 증폭이다. 따라서 P/D 분할의 TPOT 이득은 decode GPU를 다른 프로세스와 나누지 않을 때만 유지된다(C_4090 TPOT k 1.00, C_3090 2.8). 공존 단일 엔진(S4_4090)은 같은 구조에 prefill까지 더해지므로 같은 기제로 설명될 것으로 보지만 이번 캡처 대상은 아니었다.

**다음 후보(미실시).**
1. 같은 캡처를 S4_4090(공존 단일 엔진)에 1회 해서 "prefill + decode + DiT 세 종류 커널의 시간 분할"이 TPOT 2.5배·TTFT 1.9배를 같은 식으로 설명하는지 확인(약 15분).
2. 시간 분할 모델로 k를 계산: 공유 GPU에서 상대가 항상 준비 상태(점유 ≥ 0.95)면 decode 스텝 ≈ 2× 단독, 간격 증가는 별도 항. 기존 Stage 1·2 자료로 검증 가능하며 GPU 실행 불필요.
3. 기제가 시간 분할이므로 MPS(한 컨텍스트로 묶어 공간 분할) 또는 MIG(B300)에서는 간섭 모양이 달라진다. B300 포팅 시 간섭 모델을 재측정해야 하는 근거로 handoff 문서에 반영.
4. `--cuda-graph-trace=node`로 그래프 내부 커널까지 보는 캡처는 위 결론을 바꾸지 않을 것으로 보여 보류.

## Nsight 진단 2: S4_4090(단일 엔진 + DiT, 둘 다 4090) 같은 캡처 (2026-11-02 18:01~18:13, 캡처 3회)

**설정.** `PLACE=S4_4090 bench/hetero/profile_mixed_nsys.sh {mixed|llm|dit}`. vLLM 단일 엔진(:8100, prefill+decode)과 dit_server(:8300)를 모두 4090에 두고 한 nsys 세션으로 캡처. 나머지는 진단 1과 같다. 결과 `results/hetero/nsys_s4_4090/`. DiT 단독은 4090에서 여유가 있어(점유 0.53) 커널 창이 81 s로 조금 짧다.

**재생 기록 sanity(부분값).** TPOT p50 LLM 단독 7.9 → 혼합 22.1 ms, TTFT p50 0.096 → 0.240 s, denoise p50 1.25 → 2.20 s. 프로토콜(TPOT p90 7.8 → 19.7, TTFT 0.102 → 0.192, denoise 1.24 → 2.16)과 같은 방향·크기.

| 항목 | LLM 단독 | DiT 단독 | 혼합 |
|---|---|---|---|
| decode 스텝(graph ≥ 2 ms) p50/p90 | 6.85 / 7.43 ms | – | 14.11 / 15.02 ms (2.06×) |
| 스텝 사이 간격 p50 | 0.90 ms | – | 5.43 ms |
| prefill 큰 gemm 커널 p50 (64×128 sliced) | 574 µs | – | 1093 µs (1.90×); p90 1290 → 3646 |
| 짧은 vLLM 커널(rms_norm, silu 등) p50 비 | – | – | 0.9~1.1 |
| vLLM Running | 0~4 | – | 0~5 |
| DiT 커널 수 / 창 | – | 2,328,900 | 2,364,314 (같음) |
| DiT 커널 p50/p90/p99 | – | 7 / 62 / 240 µs | 7 / 62 / 245 µs (같음) |
| DiT 커널 평균 | – | 18.4 µs | 29.6 µs (1.61×) |
| 1 ms 넘는 DiT 커널 비율 / 시간 비중 | – | 0.00% / 0% | 0.47% / 39% |
| DiT 레인 빈틈(커널 사이 0.3 ms 초과) 수 / 합 | – | 125 / 0.3 s | 2,052 / 4.8 s (그중 4.6 s가 vLLM 실행 구간 안) |
| 그 빈틈 길이 p25~p90 | – | – | 2.24~2.27 ms |
| DiT 점유 / decode 점유 / 합집합 | – / 0.443 / – | 0.526 / – / – | 0.776 / 0.644 / 0.885 |
| dmon GPU0 sm% / mem% 평균 | 42 / 33 | 44 / 17 | 89 / 47 (p90 67) |

관측.
- **같은 기제가 4090·단일 엔진에서도 그대로 나온다.** decode 스텝 2.06×(C_3090 2.14×), 간격 +4.5 ms(C_3090 +4.6 ms), DiT 커널 중앙값 불변, 소수 커널만 ms 단위로 늘어남.
- **시간 슬라이스 길이는 약 2.3 ms로 측정된다.** DiT 커널 사이의 빈틈 중 0.3 ms를 넘는 것의 길이가 p25~p90 모두 2.24~2.27 ms(C_3090에서도 2.27~2.29 ms)이고, 그 시간의 96%가 vLLM 커널·그래프 실행 구간 안에 있다. 즉 GPU가 DiT에서 vLLM으로 넘어가는 차례 하나가 약 2.3 ms다. 커널 안에 끼어든 경우의 늘어난 길이 분포(0.4~0.6 ms, 2.3~2.9 ms)도 같은 값을 가리킨다.
- **prefill도 같은 방식으로 늘어난다.** 단일 엔진은 prefill을 그래프 밖 eager 커널로 돌리는데, 0.5 ms급 gemm은 p50이 1.9배, p90은 2.8배가 된다. 짧은 커널(수십 µs)은 그대로다. 이것이 TTFT 1.9~2.5배의 원인이며, 타임라인에서 115 ms짜리 prefill 구간 내내 DiT 커널이 번갈아 끼어 있는 것이 보인다.
- **DiT 쪽 차이: 4090에서는 처리량은 유지되고 지연만 는다.** DiT 단독 점유가 0.53이라 여유가 있어 커널 수(처리량)는 같고 denoise 지연만 1.76배가 된다. 3090(점유 0.97)에서는 처리량이 0.68배로 떨어져 큐가 쌓였다. 공유의 결과가 "지연 증가"로 끝나느냐 "포화"로 가느냐는 공유 전 단독 점유가 1에 얼마나 가까운지로 갈린다.
- **배치 크기·KV·vLLM 큐잉은 여기서도 원인이 아니다.** Running 0~5, Waiting 0~1, KV ≤ 43%.

**판단(관측 수준).** 두 배치, 두 GPU에서 동일하게, 공유 GPU의 간섭은 프로세스 간 ~2.3 ms 단위 시간 분할로 설명된다. 커널 둔화나 대역폭 포화는 어느 캡처에서도 측정되지 않았다. 이 동작은 MPS나 MIG 없이 두 프로세스를 한 GPU에 올렸을 때 CUDA의 기본 동작이며(NVIDIA MPS 문서: 서로 다른 컨텍스트의 작업은 동시에 실행될 수 없고 time-slice로 번갈아 실행), 우리 설정(compute mode Default, MPS 없음, MIG 없음)은 그 기본 상태다. 따라서 Stage 1·2의 k는 "MPS 없는 공유"에 대한 값이다.

**다음 후보(미실시).** MPS를 켠 공유 1회 비교(사용자 보류, 나중에 별도 진행). 시간 분할 모델로 k 계산: 공유 GPU에서 상대 점유 u_other가 높을수록 자기 스텝이 (1 + 상대가 차례를 가져가는 비율)배가 되고, 상대가 항상 준비 상태면 2배에 수렴. 기존 자료로 검증 가능.

## k를 시간 분할 기제에서 계산하기: 적합 매개변수 0개 (2026-11-03, GPU 실행 없음)

**목적.** Nsight 진단으로 확인한 기제(프로세스 간 ~2.3 ms 시간 분할, 커널은 느려지지 않음, 상대가 준비돼 있으면 스텝이 2배)를 그대로 식으로 옮겨, 단독 프로파일만으로 k를 계산할 수 있는지 본다. 비교 대상은 `k_model.md`와 같은 46점(공유 배치 5개 × α × 지표, k = 혼합 p90 / 단독 p90). 스크립트 `bench/hetero/analyze_k_timeshare.py`, 결과 `results/hetero/k_model/k_timeshare.{md,json,png}`.

**모델(TS3).** 입력은 단독 실행의 nvidia-smi 점유율 u_me(내 단계), u_ot(상대), 단독 TPOT p50, 그리고 Nsight에서 한 번 잰 상수 두 개(슬라이스 2.3 ms, 스텝 사이 손실 2슬라이스).
- 스텝 배율 k_step = 1 + p. p는 "내 스텝 동안 상대가 일을 갖고 있을 확률".
- p의 규칙: 상대의 활동 덩어리가 내 스텝보다 훨씬 길면 p = 1(k 측정이 p90이므로 상대와 겹친 요청이 p90을 결정한다). decode 스텝(7~27 ms) 대 DiT 이미지(1~3 s)가 여기 해당하고, DiT 이미지 대 decode를 가진 LLM(요청 내내 활동)도 해당한다. 상대의 덩어리가 짧으면(DiT 이미지 대 prefill만 있는 GPU, 덩어리 ~TTFT) p = 되먹임으로 부풀린 평균 점유 b(a ← u_me(1+b), b ← u_ot(1+a) 반복).
- TPOT k = ((1+p)(T−1 ms) + 1 ms + 2·2.3 ms·p) / T. TTFT·DiT k = 1 + p(TTFT는 p = b만 사용; TTFT에는 큐·KV 전송 등 연산 외 성분이 커서 거칠다).
- 포화 판정: 부풀린 점유 합 a + b ≥ 1이면 "두 프로세스가 GPU 하나 이상을 요구" → 예측 불신.

**결과.**

| 범위 | n | 중앙값 \|상대오차\| | p90 | 25% 이내 | 지표별 중앙값 TTFT / TPOT / DiT |
|---|---|---|---|---|---|
| 비포화(a+b < 1) | 16 | 9% | 15% | 15/16 | 12% / 7% / 11% |
| 포화(a+b ≥ 1) | 30 | 33% | 81% | 10/30 | 28% / 32% / 61% |
| 전체 | 46 | 19% | 71% | 25/46 | 26% / 18% / 17% |

참고: 매개변수를 맞춘 기존 모델(LIN/PS/PSG, 배치 하나 빼고 맞춤, α ≤ 1.25)은 중앙값 23~37%, 25% 이내 13~22/39였다.

관측.
- **비포화 구역에서는 측정 없이 ±15%.** S4_4090 α 0.5·0.6의 TPOT 2.31/2.49 → 2.47/2.47, DiT 2.02/2.16 → 2.00; C_4090·C_split의 DiT 1.16~1.48 → 1.13~1.31; C_3090 α 0.5 TPOT 2.57 → 2.39. 유일한 예외는 C_3090 α 0.5 TTFT(3.47 → 1.60): PD의 TTFT는 decode GPU에서 첫 토큰을 내므로 DiT 큐에 묶이는데 모델에 없다.
- **포화 경계(a+b 1.0~1.05)까지는 완만히 벗어난다.** S4_4090 α 0.75·0.85: −7~−17%. a+b가 1.2를 넘으면 큐잉이 지배해 2~40배가 되고 모델은 2를 넘지 못한다. 이 구역은 "불가"로 답하는 것이 맞다.
- **이전에 안 맞던 TPOT이 맞는 이유.** 평균 점유율(0.2~0.45)로는 1.3~1.6이 나와 50% 과소였는데, p90의 요청은 DiT 이미지와 겹친 요청이라 상대 준비 확률이 1이다. 덩어리 길이 비교로 p를 정하는 것이 핵심이고, 이는 Nsight 타임라인(decode 14 ms 상자가 1.2 s짜리 DiT 이미지 안에 들어 있음)에서 직접 보인다.

**유보.** 46점을 보고 TS0→TS1→TS2→TS3으로 규칙을 고른 것이므로 선택 편향이 있다. 규칙 자체(덩어리 길이 비교, 되먹임 점유, 포화 = 요구 합 ≥ 1)는 기제에서 나왔지만 "맞는 규칙"을 데이터를 보고 골랐다. 검증이 되려면 (1) 이 46점이 아닌 데이터(seed 902·903 혼합 실행)에서 k를 재서 맞추고, (2) 선택기 수준에서 TS3 k로 solver를 돌려 oracle 대비 결과(ok/violation/overpay/missed)를 2-앵커 선형과 비교해야 한다. 둘 다 GPU 실행 없이 가능하다. 그 전까지는 "측정 0회 k"를 주장하지 않는다.

**다음 후보(미실시).** (1) stage2b_report에 k 모드 `timeshare`를 추가해 선택기 수준 평가. (2) seed 902·903 혼합 실행으로 held-out k 비교. (3) PD TTFT 항: decode GPU의 DiT 큐 영향을 TTFT에 더하는 식(첫 토큰 = prefill + KV 전송 + decode GPU의 첫 스텝).

## B300 포팅 0-a: 공유 방식 기록 (2026-11-03, GPU 실행 없음)

**목적.** handoff 7.3의 0-a. R6(프로세스 간 시간 분할)의 전제인 "compute mode Default · MIG off · MPS 없음"이 새 풀에서 성립하는지 기록한다.

**관측.**

| 노드 | GPU | compute mode | MIG | MPS | 드라이버 / CUDA | 비고 |
|---|---|---|---|---|---|---|
| B300 `vanta-gpu-worker-study01` (10.201.4.151) | B300 SXM6 ×8, 275 GB | 8장 모두 Default | GPU 7만 Enabled(1g.67gb 1 + 1g.34gb 5, 타 사용자) | 없음 | 595.71.05 / 13.2 | 비어 있는 카드 1·3·4·6. 0·2·5는 타 사용자(vLLM 등) |
| RTX PRO 6000 `vanta-dev-gpu-worker-infer01` (10.201.4.101) | PRO 6000 Blackwell SE ×4, 96 GB | 4장 모두 Default | 처음 4장 Enabled. 0·1은 인스턴스 없이 모드만 켜져 CUDA 불가(`cuInit`→100). 사용자가 `nvidia-smi -i 0,1 -mig 0`으로 해제 → 0·1 Disabled, `cuInit 0 / ctxCreate 0` 확인. 2·3은 Enabled 유지(3번 2g.48gb 위 타 사용자 vLLM) | 없음 | 595.71.05 | 사용 가능 카드 0·1 |

**노드 간 링크.** 같은 서브넷 1홉. B300 쪽 bond0 = 100 GbE ×2 active-backup, PRO 6000 쪽 bond0 = **25 GbE ×2 active-backup(Broadcom BCM57414, RoCE 가능)**. PRO 6000 노드에는 IB 카드가 물리적으로 없음(lspci에 Mellanox 없음). 따라서 노드 간 상한 25 Gbps ≈ 2.9 GB/s(9월 iperf3 실측 23.5 Gbps). RTT 0.07~0.13 ms. 이전 4090↔3090 PD 전송이 UCX TCP 루프백 2.4~2.6 GB/s였으므로(`results/hetero/pd_xfer/README.md`) 노드 간 PD 분할은 같은 급의 링크 위에서 성립한다. B300 노드의 IB(800G ×8, 100G ×4)는 B300 간 전용.

**판단(관측 수준).** B300 1·3·4·6과 PRO 6000 0·1은 R6 전제(Default, MIG off, MPS 없음)를 만족한다. 규칙 B는 0-b 캡처로 확인한 뒤 적용한다. MIG 분리 실험(0-c)은 B300 쪽에서 sudo로 인스턴스를 만들어 할 수 있다(PRO 6000 쪽은 비밀번호 sudo).

**환경.** 인터넷 열림(PyPI·HF 200). B300 venv `~/venv/jinwoo`: torch 2.13+cu130(sm_100/120), diffusers 0.40, **vLLM·NIXL 없음**. PRO 6000 노드: venv 없음, 시스템 python3에 torch 없음. 모델 캐시 양쪽 비어 있음.

**제안(미실시).** (1) 별도 venv `~/venv/mllm`에 vLLM+NIXL 설치(양쪽 노드), 3B·SD3.5-medium 다운로드. (2) `run_mixed_modal.sh`의 멀티노드 복사본: 카드 2장 → (호스트, UUID), decode vLLM·dit_server 원격 기동, `UCX_NET_DEVICES=bond0`. (3) Stage 0: pd_smoke(B300 2장) → pd_xfer_probe(B300→PRO 6000) → 단독 포화점 → λ0.

## B300 포팅 Stage 0-1: 단일 카드 스모크 (2026-11-03 실행, 2026-11-05 정리)

실행: B300 GPU1(`GPU-1b5a091d`) 한 장에 vLLM(Llama-3.2-3B-Instruct, `--gpu-memory-utilization 0.5 --kv-cache-memory-bytes 4GiB`) + dit_server(SD3.5-medium 512², 28 steps) 동거. 스크립트는 세션 임시 `smoke.sh`(저장소 밖). 결과 `results/hetero/stage0_b300/smoke2/`.

### 관측

| 항목 | B300 | 이전(4090) |
|---|---|---|
| attention backend | **FLASHINFER** (vLLM 0.28.0 자동 선택, 후보 FLASHINFER/FLASH_ATTN/TRITON_ATTN/FLEX_ATTENTION) | FLASH_ATTN |
| GPU KV cache size | **37,440 tokens** | 37,440 |
| 기동 시간 (vLLM·DiT 모두 READY) | 108 s (FlashInfer JIT 포함) | — |
| GPU1 메모리 (두 서버 상주) | 22,136 MiB (dit 8.8 GB + vLLM 13.4 GB) | — |
| DiT solo 1장 (encode/denoise/vae/total) | 0.027 / 1.04 / 0.03 / **1.09 s** (첫 요청 1.35 s, 워밍업) | ~10 s 급 |
| LLM 64 tok 응답 | 정상 3/3 | — |
| 혼합 버스트 4 LLM(48 tok) + 2 DiT 동시 | LLM 0.089~0.100 s, DiT 1.12 s / 2.22 s (직렬) | — |

- KV 토큰 수가 이전 서버와 **정확히 일치**(37,440). `--kv-cache-memory-bytes` 고정이 카드 종류와 무관하게 KV 용량을 맞춰 준다는 것이 확인됨. Stage 1 의 "KV 4 GiB 고정" 조건은 B300·PRO6000 에서도 그대로 쓸 수 있다.
- DiT 1건이 1.1 s 라 B300 에서는 4090 대비 ~9× 빠르다. 같은 λ 로는 DiT 가 포화하지 않으므로 λ0 는 solo 포화에서 다시 뽑아야 한다(Stage 0-3).
- 혼합 버스트에서 LLM 응답은 DiT 와 동시임에도 0.1 s 안쪽. 요청 수가 작아 간섭 판정은 못 한다(관측 범위 밖). 간섭은 Stage 2 에서 본다.
- 실행 결함: 스모크 스크립트 끝의 `wait` 가 서버 프로세스까지 기다려 요청 완료 후 ~41 h 동안 서버가 GPU1 에 남아 있었다(타 사용자 영향 없음, GPU1 은 우리 전용). 2026-11-05 수동 종료, 스크립트는 curl PID 만 기다리도록 수정. 저장소에 넣을 때 반영.

### 제안

- FlashInfer 를 Blackwell 정상 경로로 채택하고 run 마다 백엔드를 deployment.json 에 기록(`run_mixed_modal_mn.sh` 에 추가 예정). 4090 결과(FLASH_ATTN)와의 차이는 "카드 차이"에 백엔드 차이가 섞인다는 점을 보고서에 명시.
- 다음: Stage 0-2 `pd_smoke` (B300 두 장: GPU1·GPU3) → `pd_xfer_probe` B300→PRO6000(25 GbE). GPU 사용이므로 시작 전 확인 필요.

## B300 포팅 0-b: 한 장 공유(S4_B300) Nsight 캡처 1차 (2026-11-05 10:30~10:46)

실행: `PLACE=S4_B300 bench/hetero/profile_mixed_nsys.sh mixed|llm|dit`. B300 GPU1 한 장에 vLLM(3B, FLASHINFER, TRTLLM prefill 자동 선택) + dit_server(SD3.5-medium). nsys 2025.6.3(`/opt/nvidia/nsight-systems/2025.6.3`, 시스템 기본 `nsys`는 CUDA 13.1 설치 불완전으로 동작 안 함). 플랜 alpha1_seed901, 캡처 90 s. 출력 `results/hetero/nsys_s4_b300/{mixed,llm,dit}/`.

### 관측

- **mixed·llm 두 모드에서 vLLM 엔진이 재생 시작 약 8 s 뒤 멈췄다.** 완료된 LLM 요청은 mixed 15건, llm 20건(idx 0~20 중 19·21 제외, 전부 out_len ≤ 12). 멈춘 뒤 엔진 로그는 `Running: 2 reqs, Waiting: 0, generation 0.0 tok/s`가 캡처 종료까지 반복, dmon sm% 0. API 서버에는 요청이 계속 도착했다(`POST /v1/completions` llm 344건, mixed 412건)이지만 엔진에 전달되지 않았고, 종료 시 EngineCore 가 SIGTERM 에 반응하지 않아 강제 종료됐다. → EngineCore 프로세스가 호스트 쪽에서 교착. 두 모드가 같은 지점에서 멈춰 결정적이다.
- 기각된 가설: FlashInfer 추론 중 JIT 컴파일(`flashinfer_jit.log`에 추론 중 컴파일 없음, cached_ops 변경 없음). 긴 프롬프트(idx 7, 6,156 토큰은 정상 완료). KV 부족(usage 4%).
- 남은 후보: (a) nsys 2025.6.3 주입 + `--cuda-graph-trace=graph` 와 vLLM CUDA graph/FlashInfer(trtllm-gen cubin) 조합의 교착. 이전 서버는 nsys 2026.5.1 + FLASH_ATTN 에서 문제 없었다. (b) FlashInfer/TRTLLM 경로 자체의 긴 디코드(out_len 63·208) 교착. 스모크(nsys 없음)에서는 48·64 토큰 디코드가 정상이었다는 점에서 (a)가 더 유력하지만 미확정.
- **dit 모드(DiT 단독)는 정상.** 84건, e2e p50 1.54 s, p90 3.39 s, 오류 0. `analyze_nsys.py`는 이 캡처에서 실패한다(vLLM이 유휴라 CUDA graph 트레이스 표가 없음. 이전 서버는 유휴 vLLM도 표가 있었음). 스크립트는 건드리지 않고 sqlite를 직접 집계한 값(DiT 프로세스만): 커널 2,532,431개, GPU busy 18.4 s / 90.2 s(점유 0.20), 커널 µs p50 4.64 · p90 13.4 · p99 29.5 · max 1,882; dmon sm% 평균 21.6. 비교: 4090 DiT 단독은 점유 0.53, 커널 p50 6.56 µs, sm% 44.5. 같은 λ(α 1.0)에서 B300의 DiT 단독 점유가 0.20이라 공유해도 포화 구간에는 못 들어간다. λ0 재산정의 근거.
- 따라서 mixed·llm 캡처는 **R6 판정에 쓸 수 없다**(캡처 창 안에 디코드 스텝이 없음).

### 제안

1. 원인 분리 실행(GPU1, 약 5분, 확인 후): 같은 서버·같은 플랜을 **nsys 없이** llm-only 로 60 s 재생. 통과하면 (a) 확정 → nsys 옵션을 바꿔(`--cuda-graph-trace=node`, 또는 `-t cuda` 만) 재캡처. 실패하면 (b) → `VLLM_ATTENTION_BACKEND=FLASH_ATTN` 또는 `--enforce-eager` 로 비교.
2. 이전 서버와 같은 nsys 2026.5.1 을 홈에 설치할 수 있으면 버전 변수부터 제거.
3. 관련 보고: nsys 로 감싼 vLLM 이 멈추는 사례(NVIDIA 포럼), RTX PRO 6000 CUDA graph 호스트 락 사례. 같은 증상인지는 불명.

### 0-b 멈춤 원인 분리 (2026-11-05 10:55~11:07, GPU1, LLM만 60~75 s 재생, 수집 미시작)

| 시험 | 조건 | 결과 |
|---|---|---|
| nsys 없음 | 같은 서버·플랜 | 정상. 102건 완료(멈췌던 idx 19·21 포함), 오류 0, 최대 313 토큰 디코드 |
| A | `nsys launch --trace=cuda,nvtx --cuda-graph-trace=graph` (이전 서버 옵션) | **멈춤 재현.** 20건 후 Running 2·생성 0, sm% 0.4 |
| B | `--trace=cuda,nvtx --cuda-graph-trace=node` | 정상. 74건, sm% 6.3 |
| C | `--trace=nvtx` | 정상. 74건 |

**관측.** 멈춤은 nsys 2025.6.3의 `--cuda-graph-trace=graph` 와 vLLM(FlashInfer, CUDA graph) 조합에서만 난다. 수집을 켜지 않아도 주입만으로 난다. FlashInfer/TRTLLM 경로 자체는 문제 없다(가설 (b) 기각). 기록 `results/hetero/stage0_b300/diag_llm_nonsys`, `diag_nsys_{A,B,C}`.
**조치.** `profile_mixed_nsys.sh` 의 `*_B300` 배치는 `--cuda-graph-trace=node` 를 쓴다(`GRAPH_TRACE_B300` 로 바꿀 수 있음). 4090/3090 분기는 graph 유지. 멈춘 1차 캡처는 삭제(2026-11-05 정리; 로그 요지는 위 관측에 있음). node 모드에서는 graph 실행이 하나의 구간이 아니라 노드(커널) 단위로 기록되므로, `analyze_nsys.py`(graph 표 전제)는 그대로 두고 decode 스텝은 별도 집계로 복원한다(제안: 같은 프로세스 커널 사이 간격 < 1 ms 를 한 스텝으로 묶음).

### 0-b nsys 구성 확정 (2026-11-05 11:07~11:30)

- node 모드 재캡처(nsys 2025.6.3, `--cuda-graph-trace=node`): LLM 재생은 멈추지 않았으나(mixed 147건·오류 0) **EngineCore 커널이 수집되지 않음**(진단: "CUDA profiling might have not been started correctly"). DiT 프로세스만 기록됨. 결과는 R6 판정에 못 쓰므로 삭제(2026-11-05 정리).
- 추가 시험(GPU1, LLM만 60 s, 20 s 캡처):

| 시험 | nsys | 옵션 | 결과 |
|---|---|---|---|
| D | 2025.6.3 | node + `--trace-fork-before-exec=true` | 멈춤 없음, EngineCore 커널 369,649개(graph 노드 340,536) 수집 |
| E | **2026.5.1** (사용자가 NVIDIA 공개 rpm을 `~/tools/nsys`에 풀어 설치) | **graph (이전 서버와 동일)** | 멈춤 없음, EngineCore graph 실행 90,730건 수집, GRAPH_TRACE 표 생성 |

- **결정.** B300 캡처는 nsys 2026.5.1 + graph 모드로 한다. 이전 서버와 nsys 버전·옵션·분석기(`analyze_nsys.py`)가 모두 같아져 비교에 변수가 없다. `profile_mixed_nsys.sh`의 `*_B300` 기본값을 그렇게 바꿈. node 모드용 집계 스크립트는 graph 모드로 확정되어 삭제(2026-11-05 정리).
- 11:31 S4_B300 세 모드 본 캡처 시작(`results/hetero/nsys_s4_b300/`).

## B300 포팅 0-b 결과: 한 장 공유(S4_B300) Nsight 캡처 (2026-11-05 11:31~11:48, nsys 2026.5.1 graph 모드)

실행: `PLACE=S4_B300 bench/hetero/profile_mixed_nsys.sh mixed|llm|dit`. B300 GPU1 한 장, vLLM(Llama-3.2-3B, FLASHINFER) + dit_server(SD3.5-medium 512² 28 steps), 플랜 alpha1_seed901(이전 서버와 같은 요청률), 캡처 90 s. 멈춤·오류 없음(mixed LLM 424건·DiT 79건, llm 346건, dit 86건). 결과 `results/hetero/nsys_s4_b300/{mixed,llm,dit}/` (`nsys_summary.json`, `timeline_*.png`). 비교 대상은 "Nsight 진단 2"의 S4_4090.

### 관측

**분석기 문턱.** `analyze_nsys.py`는 decode 스텝을 "graph 실행 ≥ 2 ms"로 정의한다(4090 스텝 6.8 ms 기준). B300 단독 decode graph 길이 분포는 0.1 ms 미만(보조 launch 1,247개)과 1.5~3 ms(4,127개, p50 1.83 ms)의 두 덩어리라 2 ms 문턱은 스텝의 95%를 버린다(219/4,127). 분석기는 건드리지 않고, 문턱만 환경변수로 받는 사본(세션 스크래치 `analysis/analyze_nsys_thr.py`, `STEP_MIN_MS=1`)으로 같은 지표를 다시 냈다. 아래 decode 수치는 1 ms 문턱 값이며, 2 ms 문턱 값은 각 디렉터리의 `nsys_summary.json`에 그대로 있다.

| 지표 | B300 단독 | B300 혼합 | 비율 | 4090 단독 → 혼합 (비율) |
|---|---|---|---|---|
| decode 스텝 p50 / p90 ms | 1.84 / 1.93 | 1.85 / 1.94 | **1.00** | 6.85 → 14.1 (2.06) |
| 스텝 사이 간격 p50 ms | 0.21 | 1.68 (DiT 있을 때 1.72, 없을 때 0.22) | **8.0** | 0.90 → 5.43 (6.0) |
| 스텝+간격 (토큰당 시간) ms | 2.05 | 3.53 | **1.72** | 7.75 → 19.5 (2.5) |
| 요청 TPOT p50 / p90 ms (캡처 중 기록) | 1.96 / 2.32 | 2.84 / 4.65 | **1.45 / 2.0** | 9.4 → 25.4 (2.7, C_3090) |
| 요청 TTFT p50 / p90 ms | 28.9 / 58.6 | 32.3 / 76.1 | 1.12 / 1.30 | — |
| DiT 커널 p50 / p99 µs | 4.6 / 29.5 | 4.6 / 29.5 | **1.00** | 6.5 / 240 → 동일 |
| DiT ≥ 1 ms 커널 수 | 37 | 535 (p10/p50/p90 = 2.34/2.46/2.72 ms) | — | 늘어난 길이 2.3~2.6 ms |
| DiT denoise p50 / e2e p90 s | 1.36 / 3.61 | 1.33 / 3.35 | **1.0** | 지연 1.76× |
| 점유(busy frac) decode / DiT / 합집합 | 0.11 / 0.20 | 0.12 / 0.21 / 0.31 | — | 0.64 / 0.78 / 0.89 |
| dmon sm% | 13 / 22 | 39 | — | — / 89 |

- **R6의 기제는 B300에서도 같다.** (1) 커널은 느려지지 않는다(DiT 커널 p50·p99 동일, decode 스텝 길이 동일). (2) 상대 프로세스가 끼어든 DiT 커널만 늘어나고, 늘어난 길이가 2.2~2.5 ms에 몰린다(4090/3090의 2.3 ms 슬라이스와 같은 크기). (3) 공유 방식은 0-a에서 확인한 기본 시간 분할(Default·MIG off·MPS 없음).
- **드러나는 모양은 다르다.** 4090에서는 decode 스텝(6.8 ms)이 슬라이스(2.3 ms)보다 길어 스텝 중간이 잘려 스텝이 2배가 됐다. B300에서는 decode 스텝(1.84 ms)이 슬라이스보다 짧아 스텝은 한 차례 안에 끝나고, 비용은 전부 **스텝 사이 간격**(0.21 → 1.72 ms, DiT 슬라이스 하나를 기다리는 시간)으로 나타난다. 그 결과 토큰당 시간 1.72×, 요청 TPOT p50 1.45×·p90 2.0×로, "스텝 2배" 공식은 B300에 그대로 옮기면 과대 추정이다.
- **DiT 쪽은 요청 수준에서 간섭이 안 보인다.** DiT 점유 0.20 + decode 0.11로 GPU가 70% 비어 있어, 늘어난 커널 535개(합 ~1.3 s/90 s)가 denoise 시간(1.3 s/요청)에 묻힌다. R6-b(결과는 단독 점유 합으로 갈린다)와 일치하며, 이번 요청률은 "조금 느려지는" 구간이다(사용자 판단 2026-11-05: 포화 구간은 이번 목적이 아님).
- 혼합 sm% 39 ≈ 단독 합 35. 대역폭·클럭 요인은 이번에도 없다(mem% 10).
- 그림 `results/hetero/nsys_s4_b300/mixed/fig_dit_kernel_lengths.png`: DiT 커널 길이 히스토그램을 단독(위)·혼합(아래) 두 패널로. 2.2~3.0 ms 구간이 단독 0개, 혼합 464개.
- 그림 `results/hetero/nsys_s4_b300/mixed/fig_elongated_kernels.png`: (a) 0.3 ms 이상 DiT 커널 길이 히스토그램, 단독(333개)과 혼합(785개). 단독의 봉우리(0.4·0.55·0.95·1.85 ms)는 혼합에도 그대로 있고, 혼합에만 2.2~3.0 ms 덩어리(약 450개)가 추가된다. 늘어난 양 = 2.2~3.0 ms − 수십 µs ≈ decode 차례 하나. (b) 그중 하나(2.48 ms)를 확대: 커널이 멈춘 사이에 vLLM decode graph 1.87 ms가 들어가 있다.
- 그림 `results/hetero/nsys_s4_b300/mixed/timeline_mixed_busy.png`(캡처 +1.6 s, 30 ms 창): decode graph(~1.9 ms) → DiT 커널 묶음(~1~1.5 ms) → decode graph … 가 겹침 없이 번갈아 가며, 주기 약 4 ms. 단독(`llm/timeline_llm.png`)에서는 graph 사이 간격이 0.2 ms다. 기본 창(+30 s) 그림 `timeline_mixed.png`는 prefill 구간이라 참고용.

### 제안

1. **k 모델의 입력에 "스텝 길이 ÷ 슬라이스 길이"를 넣어야 한다.** 스텝 < 슬라이스면 간섭은 "상대 슬라이스 대기(간격)"로, 스텝 > 슬라이스면 "스텝 안 절단"으로 나타난다. R7(k = 1 + p)은 이 구분이 없으므로 B300 값으로 검증하기 전에는 solver에 넣지 않는다(기존 결정 유지). R2의 2-앵커 선형은 이 모양 차이와 무관하게 쓸 수 있다.
2. **`analyze_nsys.py`의 스텝 문턱 2 ms는 카드별 상수다.** 판정 정의를 바꾸지 않기 위해 사본으로 처리했지만, 포팅을 확정할 때는 `STEP_MIN_MS` 인자(기본 2)를 저장소 분석기에 넣는 것이 맞다. 사용자 확인 후.
3. 다음: C_B300(PD 분할 + decode 카드 공유, GPU1·GPU3) 같은 캡처 → decode 전용 엔진에서도 "간격형" 간섭인지. 그 뒤 Stage 0-2(pd_smoke, 노드 간 pd_xfer_probe).

## B300 포팅 0-b 결과 2: PD 분할 + decode 카드 공유(C_B300) Nsight 캡처 (2026-11-05 11:55~12:12)

실행: `PLACE=C_B300 bench/hetero/profile_mixed_nsys.sh mixed|llm|dit`. prefill vLLM = GPU3 단독(추적 안 함), decode vLLM + dit_server = GPU1 공유, KV 전송 NixlConnector(UCX tcp, 같은 노드). nsys 2026.5.1 graph 모드. 멈춤·오류 없음(mixed LLM 454·DiT 84, llm 377, dit 84). 결과 `results/hetero/nsys_c_b300/`. 두 프로세스가 GPU1에 있었음을 sqlite의 deviceId로 확인(dit_server는 자기 가시 목록의 cuda:1 = GPU1, decode는 cuda:0 = GPU1).

### 관측

| 지표 (1 ms 문턱) | C_B300 단독 → 혼합 | S4_B300 단독 → 혼합 (참고) |
|---|---|---|
| decode 스텝 p50 ms | 1.84 → 1.85 (1.00×) | 1.84 → 1.85 |
| 스텝 사이 간격 p50 ms | 0.27 → 1.63 (DiT 있을 때 1.69, 없을 때 0.32) | 0.21 → 1.68 |
| 토큰당 시간 ms | 2.11 → 3.48 (1.65×) | 2.05 → 3.53 (1.72×) |
| 요청 TPOT p50 / p90 ms | 1.96 / 2.32 → 2.59 / 4.32 (1.32× / 1.86×) | 1.45× / 2.0× |
| 요청 TTFT p50 / p90 ms | 350 / 1,051 → 526 / 1,540 | 29 / 59 → 32 / 76 |
| DiT 커널 p50 µs | 4.6 → 4.7 | 4.6 → 4.6 |
| DiT 2.2~3.0 ms 커널 수 | 0 → 30 (p50 2.48 ms) | 0 → 464 |
| 점유 decode / DiT / 합집합 | 0.09 / 0.20 → 0.29 | 0.11 / 0.20 → 0.31 |

- **decode 전용 엔진에서도 간섭은 "간격형"이다.** 스텝 길이는 그대로이고 DiT가 있을 때 스텝 사이 간격이 1.7 ms로 늘어난다. S4_B300과 같은 모양·같은 크기. 이전 C_3090(decode 스텝 8.5 → 18.3 ms, 2.8×)과 다른 것은 카드 종류가 아니라 스텝 길이(1.84 ms)가 슬라이스(~2.3 ms)보다 짧기 때문이라는 해석이 두 배치에서 유지된다.
- **PD의 TTFT는 KV 전송이 지배한다.** 단독에서도 TTFT p50 350 ms·p90 1.05 s이고 vLLM 로그의 KV 전송 평균 0.7~1.1 s, P90 1.3~2 s. 3B 모델 1k 토큰 KV는 ~110 MB라 2.4 GB/s면 50 ms여야 하는데 10배 이상 느리다. 간섭 판정과 무관한 별도 문제(UCX tcp 경로·세그먼트 설정 또는 B300 NIXL 경로)로, Stage 0-2 `pd_xfer_probe`에서 다룬다.
- **DiT 요청 지연은 이 캡처로는 판정할 수 없다.** 네 캡처 모두 denoise 시간이 한 실행 안에서 0.65~1.5 s로 흔들린다(첫 구간 빠르고 중간 느림). 원인은 GPU가 아니다. DiT 커널 p50은 전 실행 4.6 µs로 같고, DiT 단독에서 요청당 GPU 시간 0.49 s vs 벽시계 0.91 s(GPU 몫 0.54, 요청당 커널 67k개). 즉 512² SD3.5-medium은 B300에서 **CPU 측 커널 launch에 절반이 묶여** 있고, nsys 주입(launch마다 오버헤드)과 노드의 다른 사용자 CPU 부하(load 9~12, 타 사용자 python 100%)에 따라 벽시계가 움직인다. S4의 "DiT 1.36 → 1.33 변화 없음"과 C의 "0.83 → 1.31"은 둘 다 이 잡음 안이다. R6 판정의 DiT 항목은 커널 수준(p50 동일, 끼어든 커널만 ~2.3 ms 연장)으로만 쓴다.
- 정리 중 발견: 중단된 진단 스크립트가 남긴 `nvidia-smi dmon`(CPU 100%) 프로세스를 12:15에 종료. 캡처 중 dmon 자체가 CPU 하나를 점유하므로 DiT launch 지연에 영향이 있을 수 있다(미정량).

### 제안

1. **R6 포팅 판정: 성립.** 두 배치(S4_B300, C_B300)에서 (1) 커널 둔화 없음, (2) 끼어든 커널의 연장 ≈ 2.3 ms 슬라이스, (3) 공유 방식 기본 시간 분할. 규칙 A/B는 B300에 그대로 쓴다. 단 "decode 스텝 2배"는 스텝 > 슬라이스일 때의 특수형이고, 일반형은 "토큰당 시간 = 스텝 + 상대 차례 대기"다.
2. B300에서 DiT 간섭을 요청 수준으로 보려면 DiT를 GPU에 묶인 크기로 키워야 한다(1024² 또는 배치 ≥ 4, `--max-batch`). 그 전까지 DiT k는 nsys 없는 Stage 1/2 측정(이전 서버 방식)으로만 잰다.
3. 다음: Stage 0-2(pd_smoke 두 장 → pd_xfer_probe 노드 간)에서 KV 전송 속도 문제를 먼저 본다.

## B300 포팅 Stage 0-2a: 같은 노드 KV 전송 프로브 (2026-11-05 12:24~12:33, GPU3 → GPU1)

`bench/hetero/pd_xfer_probe_b300.sh`(이전 `pd_xfer_probe.sh`의 B300판: 3B 모델, 2048 토큰 프롬프트 = KV 235 MB, `--gpu-memory-utilization 0.5`). 순차 8건 / 동시 16건, decode 서버의 "KV Transfer metrics" 기준. 출력 `results/hetero/pd_xfer_b300/<tag>/`.

### 관측

| 태그 | UCX 설정 | 선택된 경로(로그) | 순차 8건 평균 / p90 | 실효 | 동시 16건 평균 / p90 |
|---|---|---|---|---|---|
| tcp_tuned_all | 하네스 그대로(`cuda_copy,tcp,self`, 4m seg, 64m buf, 4 lanes, `NET_DEVICES=all`) | `tcp/vxlan.calico` | 162 / 219 ms | 1.45 GB/s | 900 / 1,438 ms |
| tcp_tuned_lo | 위 + `NET_DEVICES=lo` | `tcp/lo` | 158 / 219 ms | 1.49 GB/s | 838 / 1,355 ms |
| tcp_tuned_bond0 | 위 + `NET_DEVICES=bond0` | `tcp/bond0` | 159 / 222 ms | 1.48 GB/s | 730 / 1,233 ms |
| **default** | UCX 환경변수 없음(NIXL 기본) | `device(cuda_ipc/cuda)` + `rc_mlx5`(IB 레인) | **20 / 42 ms** | **~12 GB/s** | **24 / 49 ms** |

- C_B300 캡처에서 본 TTFT 0.35~1.5 s와 vLLM 로그의 187 MB/s는 vLLM 설정 문제가 아니다. 커넥터·레이아웃·디스크립터 수는 이전 서버와 같다. 원인은 하네스의 UCX 환경변수가 같은 노드에서도 **호스트 경유 TCP**를 강제한 것이고, 그 경로의 상한이 이 노드에서는 1.45 GB/s다(이전 서버 2.4 GB/s). 인터페이스 선택(`all`이 Calico VXLAN을 고름)은 속도에 영향이 없었다(`lo`·`bond0`와 동일).
- 동시 16건의 700~900 ms는 직렬화가 아니라 1.45 GB/s를 16개가 나눠 쓴 결과다. B300은 prefill이 빨라 전송 16개가 거의 동시에 시작한다(이전 서버는 prefill이 느려 자연히 시차가 났다).
- 같은 노드에서는 NIXL 기본 경로로 두면 20 ms(235 MB)다. 로그상 데이터 레인은 `cuda_ipc`(NVLink), 제어·RMA는 `rc_mlx5`(B300 노드의 IB NIC mlx5_0/1/6/7). "accelerated IB support not found" 경고에도 IB 경로는 열린다. 요청당 KV 전송이 TTFT에 더하는 시간이 수십 ms로 내려간다.

### 조치·제안

1. **같은 노드 PD(C_B300 등)는 UCX 환경변수를 주지 않는다.** `profile_mixed_nsys.sh`·`run_mixed_modal_mn.sh`의 UCX_* 는 노드 간 배치(B300 → PRO6000, 25 GbE)에서만 넣는다. 멀티노드 하네스에서 `UCX_NET_DEVICES=bond0`(B300) / 해당 NIC(PRO6000)로 명시한다.
2. C_B300 캡처(간섭 판정)는 영향 없음. decode 스텝·간격 지표는 전송 경로와 무관하고, TTFT 항만 이 설정 때문에 부풀었다. Stage 1에서 C_B300을 쓸 때는 1번을 적용한 뒤 TTFT를 다시 잰다.
3. 노드 간 전송(`pd_xfer_probe` B300 → PRO6000)은 아직 안 했다. 25 GbE 이론 3.1 GB/s, 호스트 경유 TCP가 이 노드에서 1.45 GB/s이므로 노드 간은 그 이하가 될 가능성이 크다. 다음 단계에서 측정(PRO6000 GPU0 사용, 확인 필요).

## B300 공유 방식 비교 1: MPS 아래 한 장 공유(S4_B300, MPS) Nsight 캡처 (2026-11-05 13:58~)

설정: B300 GPU1 전용 MPS 제어 데몬을 사용자 권한으로 띄움(`CUDA_VISIBLE_DEVICES=GPU1 CUDA_MPS_PIPE_DIRECTORY=~/mps_gpu1/pipe nvidia-cuda-mps-control -d`, root 불필요, 다른 GPU에 영향 없음). vLLM과 dit_server를 그 파이프의 클라이언트로 띄움(`profile_mixed_nsys.sh`에 `MPS_PIPE`, `MPS_PCT_LLM/DIT` 추가). 비율 제한 없음(기본 100/100). 나머지는 S4_B300과 동일(플랜 alpha1_seed901, nsys 2026.5.1 graph). 출력 `results/hetero/nsys_s4_b300_mps/`.
주의: MPS 모드에서는 nsys가 하드웨어 트레이스를 못 쓰고 소프트웨어 계측 트레이스로 떨어진다(진단 메시지). 그래서 절대값(스텝 1.84 → 2.01 ms)이 시간 분할 캡처와 다르고, 비교는 반드시 같은 MPS 조건의 단독과 한다.

### 관측 (llm 단독 vs mixed, 1 ms 문턱)

| 지표 | MPS 단독 → 혼합 | 시간 분할 단독 → 혼합 (S4_B300) |
|---|---|---|
| decode 스텝 p50 / p90 ms | 2.01 / 2.11 → 2.49 / 2.61 (**1.24×**) | 1.84 → 1.85 (1.00×) |
| 스텝 사이 간격 p50 ms | 0.21 → 0.25 (**변화 없음**) | 0.21 → 1.68 (8×) |
| 토큰당 시간 ms | 2.22 → 2.74 (**1.23×**) | 2.05 → 3.53 (1.72×) |
| 요청 TPOT p50 / p90 ms | 2.10 / 2.46 → 2.40 / 3.22 (**1.14× / 1.31×**) | 1.96 / 2.32 → 2.84 / 4.65 (1.45× / 2.0×) |
| 요청 TTFT p50 / p90 ms | 29.0 / 62.1 → 28.9 / 62.9 (1.0×) | 28.9 / 58.6 → 32.3 / 76.1 |
| 두 프로세스 동시 실행 시간 / decode busy | 0.38 | 0.17 |
| DiT 커널 p50 / p90 / p99 µs | 5 / 17 / 34 (혼합) | 4.6 / 13.4 / 29.5 |

- **MPS는 간섭의 형태를 바꾼다.** 시간 분할에서는 스텝은 그대로이고 "상대 차례 대기"가 간격으로 붙었다. MPS에서는 간격이 사라지고(단독과 같음) 두 프로세스의 커널이 실제로 겹쳐 돌며(동시 실행 38%), 그 대신 스텝 자체가 1.24× 길어진다(SM·대역 경합). DiT 커널 p90·p99도 13→17, 29→34 µs로 늘어 DiT 쪽도 경합을 받는다.
- **크기는 절반 이하로 줄지만 0이 아니다.** 토큰당 시간 1.72× → 1.23×, TPOT p90 2.0× → 1.31×. 이 요청률(α 1.0, 두 프로세스 점유 합 0.31)에서도 k≈1.15~1.3이 남는다.
- DiT 단독(dit 모드) 1차는 노드 간 프로브의 포트 충돌로 오염되어 삭제, 14:23 깨끗이 재실행. **DiT 쪽 MPS 효과(커널 수준, 단독 → 혼합):** 커널 p50 5.1 → 5.5 µs(+8%), p90 14.6 → 17.0(+16%), p99 29.6 → 34.4(+16%). 시간 분할에서 464개였던 2.2~3.0 ms "멈춘 커널"은 6개로 사라진다. 즉 MPS에서 DiT는 멈추지 않는 대신 모든 커널이 조금씩 느려진다(SM 공유). 요청 수준 denoise(1.17 → 1.32 s)는 CPU launch 잡음 범위라 판정에 쓰지 않는다.

### 제안
- MPS 비율 스윕(`MPS_PCT_LLM/MPS_PCT_DIT` = 70/30, 30/70)을 같은 플랜으로 돌려 "비율이 LLM k와 DiT k를 반대로 움직이는지" 확인. 그 뒤 α 0.5·1.5에서 반복해 MPS k의 부하 의존성.
- 운영 관점 기록: MPS 서버는 사용자당 하나라 공유 노드에서 여러 사용자가 한 GPU를 MPS로 나눠 쓸 수 없고, 클라이언트 오류가 서버 전체로 번진다. solver의 "MPS 공유" 레버는 같은 테넌트(같은 사용자) 서비스 간에만 성립한다.

## B300 포팅 Stage 0-2b: 노드 간 KV 전송 프로브 (2026-11-05 14:19~14:24, B300 GPU3 → PRO6000 GPU1, 25 GbE)

`bench/hetero/pd_xfer_probe_mn.sh`(새 파일): prefill vLLM = B300 GPU3, decode vLLM = PRO6000 GPU1(ssh -n -f 로 원격 기동), 프록시·로드젠 = B300. 3B, 2048 토큰 프롬프트(KV 235 MB), 순차 8 / 동시 16. 출력 `results/hetero/pd_xfer_mn/<tag>/` (decode.log 는 원격에서 복사). 실패한 시도(ssh 세션·포트 충돌·해시 불일치)의 폴더는 정리 때 삭제, 내용은 아래 '발견한 것'에 요약.

### 포팅 중 발견한 것 (실행 전제)
1. **어텐션 백엔드가 NIXL 호환 해시에 들어간다.** B300(sm_103)은 FLASHINFER, PRO 6000(sm_120)은 FLASH_ATTN 을 자동 선택해 "NIXL compatibility hash mismatch" 로 모든 요청이 실패했다. vLLM 0.28 에서는 `VLLM_ATTENTION_BACKEND` 환경변수가 없어졌고 `--attention-backend FLASHINFER` CLI 인자로 양쪽을 맞춰야 한다. 이기종 노드 간 PD 의 필수 조건이며 멀티노드 하네스에도 넣는다.
2. ssh 로 원격 vLLM 을 띄울 때 `ssh host "cmd &"` 는 서버가 세션을 잡아 돌아오지 않는다(setsid·stdin 분리로도 안 됨). `ssh -n -f` 로 해결.
3. 같은 노드에서 캡처 하네스(8100/8192/5600)와 포트가 겹치면 요청이 엉뚱한 서버로 간다. 프로브는 8110/8193/5610 을 쓴다. 이 충돌로 MPS dit 단독 캡처 1회가 오염되어 재실행.

### 관측

| 태그 | UCX 설정 | 선택 경로 | 순차 8건 평균 / p90 | 실효 | 동시 16건 평균 / p90 |
|---|---|---|---|---|---|
| tcp_tuned_bond0 | `cuda_copy,tcp,self`, 4m seg, 64m buf, 4 lanes, `NET_DEVICES=bond0` | tcp/bond0 | 162 / 174 ms | **1.45 GB/s** (vLLM 집계 1,380 MB/s) | 1,340 / 2,250 ms |
| default | 없음 | **tcp/vxlan.calico** (Calico VXLAN 오버레이) | 657 / 685 ms | 0.36 GB/s | 10,037 / 10,102 ms |

- 노드 간 튜닝 TCP 의 순차 속도(1.45 GB/s)는 같은 노드 호스트 경유 TCP 와 같다. 즉 현재 상한은 25 GbE 링크(이론 3.1 GB/s)가 아니라 cuda_copy + TCP 스택의 호스트 측 복사다. 이전 서버(2.4 GB/s)보다 낮은 이유는 미확인(CPU 세대·NUMA·버퍼 설정 후보).
- 튜닝 없는 기본값은 UCX 가 Kubernetes Calico 의 VXLAN 오버레이 인터페이스를 고른다. 같은 노드에서는 어느 인터페이스든 lo 로 돌아 차이가 없었지만, 노드 간에서는 캡슐화를 거쳐 4배 느리고(0.36 GB/s) 동시 요청에서 10 s 로 직렬화된다. 노드 간 PD 는 반드시 `UCX_NET_DEVICES=bond0` + 튜닝 TCP 로 띄운다.
- 동시 16건 1.3 s 는 1.45 GB/s 를 16개가 나눈 값(16 × 235 MB / 1.45 GB/s ≈ 2.6 s 총, 평균 대기 ~1.3 s). solver 의 링크 제약: λ·KV ≤ 1.45 GB/s → 3B 2k 프롬프트 기준 약 6 req/s, 1.5k(p50) 기준 약 8.6 req/s.

### 제안
- 멀티노드 하네스(`run_mixed_modal_mn.sh`)에 `--attention-backend FLASHINFER` 와 튜닝 TCP + `UCX_NET_DEVICES=bond0`(양 노드) 를 고정. 같은 노드 PD 는 UCX 환경변수 없이.
- 링크 상한을 올리려면 RoCE(BCM57414 는 RoCE 지원)로 UCX `rc` 전송을 여는 것이 후보. 관리자 설정 필요, 후순위.
- 다음: C_PRO6000(prefill B300, decode+DiT PRO6000) 캡처로 노드 간 배치의 간섭 모양 확인.

## B300 공유 방식 비교 2: MIG 반반 분할(S4_B300_MIG) + 세 방식 종합 (2026-11-05 14:30~14:47)

설정: GPU1에 MIG 켜고(`sudo nvidia-smi -i 1 -mig 1`) 3g.135gb 두 조각(`mig -cgi 9,9 -C`, 각 SM 70개·133 GB). vLLM = 조각 1(MIG-b6fc…), dit_server = 조각 2(MIG-a4b5…). `profile_mixed_nsys.sh`에 `PLACE=S4_B300_MIG` 추가(프로세스는 MIG 조각 하나만 볼 수 있어 S4 분기의 dit 가시 장치를 분리). 나머지는 S4_B300과 동일. 출력 `results/hetero/nsys_s4_b300_mig/`. **MIG는 아직 켜져 있다**(GPU1을 통째로 쓰려면 `sudo nvidia-smi mig -i 1 -dci; -dgi; nvidia-smi -i 1 -mig 0`).

### 관측 (1 ms 문턱)

| 지표 | MIG 단독 → 혼합 | 비고 |
|---|---|---|
| decode 스텝 p50 / p90 ms | 2.51 / 2.61 → 2.51 / 2.61 (**1.00×**) | 통째 GPU 단독 1.84 → 반쪽 SM이라 1.36× 느림 |
| 스텝 사이 간격 p50 ms | 0.32 → 0.32 | |
| 요청 TPOT p50 / p90 ms | 2.73 / 3.03 → 2.72 / 3.08 (**1.00× / 1.02×**) | 통째 단독 1.96 / 2.32 |
| 요청 TTFT p50 / p90 ms | 31.2 / 83.8 → 32.4 / 87.5 | |
| DiT 커널 p50 / p90 / p99 µs | 5.2 / 26.2 / 53.4 → 동일 | 통째 GPU 4.6 / 13.4 / 29.5 → 긴 커널이 2× 느림(SM 절반) |
| DiT 2.2~3.0 ms 커널 | 0 → 0 | 멈춤 없음 |
| 두 프로세스 동시 실행 / decode busy | 0.39 | 서로 다른 조각에서 실제 동시 실행 |

- **간섭 0(k=1.00).** 단독과 혼합이 커널·스텝·요청 수준 모두 소수점까지 같다. 하드웨어 분할의 정의 그대로.
- **대가는 용량.** 같은 요청에 decode 토큰당 1.36×, DiT 긴 커널 2×. 조각 안에서 포화가 먼저 온다. 이번 요청률(점유 0.16 + 0.29)에서는 안 보이지만 결정 지도에서는 이 축이 승부를 가른다.

### 세 공유 방식 종합 (같은 카드 GPU1, 같은 플랜 alpha1_seed901, LLM 3B + DiT SD3.5m 512²)

| | 기본 시간 분할 | MPS | MIG 반반 |
|---|---|---|---|
| 기제 | 컨텍스트 교대(슬라이스 ~2.3 ms) | 커널 동시 실행, SM 공유 | 하드웨어 분리 |
| decode 스텝 | 1.84 → 1.85 (그대로) | 2.01 → 2.49 (1.24×) | 2.51 → 2.51 |
| 스텝 사이 간격 | 0.21 → 1.68 ms (대기) | 0.21 → 0.25 | 0.32 → 0.32 |
| 토큰당 시간 | **1.72×** | **1.23×** | **1.00×** (절대값은 통째의 1.36×) |
| TPOT p90 | 2.0× | 1.31× | 1.02× |
| DiT 커널 | p50 동일, 끼어든 464개만 +2.3 ms | 전부 +8~16% | 동일 (절대값 2×) |
| 혼합 측정 필요 | 앵커 2 (R2) | 1 (부하 의존성 미확인) | 0 |
| 운영 제약 | 없음 | 사용자당 서버 1, 격리 없음 | 조각 메뉴 고정, 바꿀 때 GPU 비움 |

**판단(관측 수준).** 간섭의 "형태"는 공유 방식이 정하고, 그 형태마다 필요한 측정 횟수가 다르다(0/1/2). 시간 분할만 부하 의존 k가 필요하고, MIG는 용량 모델만, MPS는 1점 상수인지 2점인지 미정. 이 표가 solver의 "공유 방식" 레버의 입력 정의가 된다.

### 제안 (가지치기 논의 2026-11-05 반영)
1. MPS α 0.5·1.5 두 번으로 MPS k의 부하 의존성 확정 → MPS 측정 횟수 결정. 그 뒤 공유 방식 축의 실험은 종료.
2. 간섭 모델링은 기본 시간 분할에만. MIG는 조각 단독 프로파일, 분리는 단독 프로파일.
3. VLM·C_split은 실측에서 제외(시뮬 확장). 배치는 S4·S3·C·노드 간 C 네 개.
4. 노드 간 링크는 25 GbE TCP 1.45 GB/s로 고정(RoCE 범위 밖, 사용자 결정).
