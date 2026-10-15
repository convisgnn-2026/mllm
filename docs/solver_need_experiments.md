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


현재 판단 및 근거는 [10/13 미팅](1013_meeting.md), 기존 데이터 분석은 [보고서](../results/hetero/solver_need/summary.md)를 참고한다.

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

[실행 안내](../bench/hetero/vlm/README.md), [진행 상태](../results/hetero/vlm_encoder_check_status.json). 먼저 이미지 응답·encoder cache 전달·consumer 재인코딩 여부를 확인하고, 다음에 공유 간섭과 encoder 실행 허용 제어를 시험한다. 기능 검증 단계의 파일 전달 시간을 최적화 성능으로 해석하지 않는다.


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

실행: bench/hetero/vlm/run_interference_pilot.py. 제어 프록시: bench/hetero/vlm/gated_proxy.py. 상태와 결과: results/hetero/vlm_interference_pilot/status.json, results.json, 조건별 summary.json·requests.live.jsonl. 전역 encoder_admission.jsonl은 요청 허용 대기를 기록한다.

판정: 무제어 공유의 추가 악화를 확인한 다음 고정 제한이 LLM 지연 감소와 VLM 추가 대기 사이에서 유용한 영역을 만드는지 본다. 효과가 없으면 그대로 보고한다. 현재 장치를 solver의 성능 개선으로 미리 주장하지 않는다.

## 다음 단계: encoder 실행 시점 제어 (seed 0)

질문: **encoder를 언제 실행할지 조절하면, LLM 응답 끊김을 줄이면서 VLM 대기 손해를 작게 유지할 수 있는가?** 반복 seed는 추가하지 않고, 기존 high 부하(LLM 1 req/s, VLM 2 req/s)의 같은 요청을 네 조건에 재생한다.

기존 파일럿은 7조건 완료, 측정 요청 오류 0건이다. high 부하에서 단독→공유 시 LLM 완료시간 p90은 4.07→4.63초, VLM 첫 응답 p90은 0.23→0.43초였다. 단순 직렬화에서는 각각 4.70초, 0.38초로 양쪽 손해를 함께 줄이지 못했다.

기존 기록을 시간축으로 대조하면 high 무제어에서 LLM 요청의 encoder primer 겹침 비율과 요청별 chunk 간격 p90의 Pearson 상관은 약 0.77이다. **탐색적 연관성**이며 인과관계나 GPU 커널 충돌의 증명이 아니다. primer에는 HTTP·전처리·파일 전달이 포함되고, 기존 기록에는 개별 chunk 시각이 없다. 같은 부하 변동이 양쪽 지연을 동시에 늘렸을 수도 있다. [분석 결과](../results/hetero/vlm_interference_pilot/overlap_analysis.json).

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
/home/jinwoo.kim27/venv/jinwoo/bin/python bench/hetero/vlm/run_interference_pilot.py \
  --control-study --seed 0 --out results/hetero/vlm_encoder_control_seed0
```

[진행 상태](../results/hetero/vlm_encoder_control_seed0/status.json). 완료 후 `results.json`과 조건별 `summary.json`을 비교한다. 기존 파일럿 결과는 보존했다.

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

[결과](../results/hetero/vlm_encoder_control_seed0/results.json), [시간 겹침 분석](../results/hetero/vlm_encoder_control_seed0/overlap_analysis.json).

## 후속 실험: 지연 후 반응과 첫 응답 전 보호 비교

가설: chunk 지연을 보고 나서 encoder를 늦추는 방식은 이미 진행 중인 encoder를 중단하지 못한다. LLM 첫 응답 전 구간을 보호하는 신호가 더 유용한지 비교한다. 보호 대상은 **LLM 요청 전송 후 첫 content chunk 전까지**이며, 실제 GPU prefill 구간과 동일하지 않다(큐·네트워크 포함).

seed 0, warmup 15초 + 측정 60초, LLM 1 req/s·VLM 2 req/s, 동일 입력과 도착 계획, 순서 shuffle. 조건은 `serialized`(하나씩 실행), `reactive_gap`(25ms 초과 chunk 신호 후 양보), `prefill_guard`(첫 응답 대기 중 LLM이 있을 때 양보)이다. 두 제어의 추가 정책 대기는 약 100ms로 제한한다. 큐 전체 대기 상한은 아니다. 피드백은 모든 조건에서 100ms마다 동일하게 보낸다. 첫 응답 전 상태도 최대 300ms 지난 신호는 무시한다. 기존 결과보다 관측 길이가 짧으므로 이번 세 조건끼리 비교한다. 추가 seed는 없다.

계측: chunk 시각, 요청 전송→첫 응답 상태, 피드백 발신·수신 시각, primer 호출 시작·완료, 정책 대기. `analyze_signal_timing.py`와 `analyze_encoder_overlap.py`로 시간 관계를 분석한다. `nsys`·`ncu` 실행 파일은 현재 PATH에서 찾지 못했다. 이번 실험에서 GPU 커널 충돌의 직접 계측을 수행했다고 주장하지 않는다. 커널 원인 분리는 별도 profiler 단계가 필요하다.

실행:
```bash
/home/jinwoo.kim27/venv/jinwoo/bin/python bench/hetero/vlm/run_interference_pilot.py \
  --timing-study --duration 60 --seed 0 --out results/hetero/vlm_signal_timing_seed0
```

[상태](../results/hetero/vlm_signal_timing_seed0/status.json). 결과가 개선되지 않으면 그대로 기록하고 신호·제어 시점 가설을 수정한다.

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

[결과](../results/hetero/vlm_signal_timing_seed0/results.json), [신호 시각 분석](../results/hetero/vlm_signal_timing_seed0/signal_timing_analysis.json), [겹침 분석](../results/hetero/vlm_signal_timing_seed0/overlap_analysis.json).

## GPU 실행 구간 계측: 수집 완료

seed 0에서 LLM만 요청 / VLM만 요청 / 함께 요청의 3조건을 실행했다. warmup 15초 + 부하 관측 30초이며, 각 조건에서 warmup 후 5초에 encoder(8510), 독립 LLM(8540)의 PyTorch profiler를 시작하고 약 8초 후 중지했다. 시작·중지 호출과 trace 저장 때문에 조건별 실제 소요시간이 늘어날 수 있다. profiler가 성능을 교란하므로 이 실행의 지연 값으로 앞선 무계측 실험 대비 개선을 주장하지 않는다.

수집 상태는 complete(3/3), 측정 요청 오류 0건이다. encoder·LLM 각각 3개의 trace가 생성됐다. 요청이 없는 서비스의 trace는 GPU kernel 이벤트 0개이고, 요청이 있는 단독·공유 trace에는 GPU kernel 이벤트가 존재하는 것을 확인했다. 파일 목록과 이벤트 수는 `results/hetero/vlm_gpu_profile_seed0/trace_inventory.json`에 기록했다. **GPU trace 수집 완료와 원인 분석 완료는 구분한다.** 다음은 커널 실행시간·실행 간격과 클라이언트 응답 간격의 대조다.

실행: `run_interference_pilot.py --profile-study --duration 30 --seed 0 --out results/hetero/vlm_gpu_profile_seed0`.

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

재현: `analyze_gpu_profile.py results/hetero/vlm_gpu_profile_seed0`, `plot_gpu_profile_summary.py results/hetero/vlm_gpu_profile_seed0`.

[상세 수치](../results/hetero/vlm_gpu_profile_seed0/gpu_analysis.json), [요약 그림](../results/hetero/vlm_gpu_profile_seed0/gpu_wait_summary.png). 단일 seed·짧은 공통 구간·profiler 오버헤드가 있는 탐색적 진단이며 추가 GPU 실험은 실행하지 않았다.

## Solver 후보 배치 비교: encoder를 어느 GPU에 둘 것인가

GPU 실행 대기 증가 관측을 구성 선택 문제로 연결한다. 2개 GPU·모델·입출력·도착 계획·seed 0을 고정하고 encoder 배치만 바꾼다.

| 후보 | 3090 | 4090 |
|---|---|---|
| with_llm: LLM과 encoder 공유 | 독립 LLM + VLM encoder | VLM prefill/decode |
| with_vlm: VLM 단계들을 같은 GPU에 배치 | 독립 LLM | VLM encoder + VLM prefill/decode |

with_vlm도 encoder/언어 처리 프로세스와 파일 전달 connector를 유지한다. 통합 VLM 서버와 동일한 구현이 아니다. 전송 방식을 바꾸지 않은 상태에서 배치 후보를 비교한다. 각 프로세스 메모리 비율은 기존과 동일(encoder 0.1, 각 언어 모델 0.4)이며 GPU 수·종류는 고정이다. 같은 GPU에서도 프로세스별 KV cache 확보량은 환경에 따라 달라질 수 있다.

각 배치에서 LLM만 요청, VLM만 요청, 함께 요청의 3조건을 실행한다. 모델은 모든 조건에 상주한다. 총 6조건, warmup 15초 + 측정 90초, 최대 drain 120초, profiler 없음. 신규 배치인 with_vlm의 실행 가능성을 먼저 확인하고 with_llm을 이어서 실행한다. 조건 순서는 기존 seed 기반 shuffle을 사용한다. 배치 블록 순서는 고정이므로 시간 경과 영향을 배제했다고 주장하지 않는다. 추가 seed 반복은 없다.

판정: 각 배치의 혼합 성능과 자기 배치 단독 기준 대비 악화율을 함께 본다. encoder GPU가 3090→4090으로 바뀌므로 절대 성능 차이를 전부 간섭 감소라고 해석하지 않는다. with_vlm에서는 encoder와 VLM 언어 처리 사이 간섭이 새로 생길 수 있다. 후보 중 한 배치가 모든 지표에서 우세하면 그 결과를 그대로 기록하며, 현 조건에서 복잡한 solver 필요성을 입증한 것으로 주장하지 않는다. 같은 후보 공간에서 단독 성능 기반 선택과 혼합 실행을 반영한 선택이 달라지는지는 다음 판단 대상이다.

실행: `bench/hetero/vlm/run_placement_study.py`. [전체 상태](../results/hetero/vlm_placement_seed0/status.json)와 배치 하위 `status.json`에 진행 상황을 기록한다. 완료 시 동일 payload/계획을 검증하고 `comparison.json`에 단독 대비 악화율을 자동 저장한다. 기존 결과는 보존한다.

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

[원시 비교 결과](../results/hetero/vlm_placement_seed0/comparison.json).

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

실행: `bench/hetero/vlm/run_placement_study.py --load-sweep`.
[상태](../results/hetero/vlm_placement_load_sweep_seed0/status.json). 배치별 하위 `status.json`에서 현재 조건을 볼 수 있고, 완료 후 `comparison.json`에 결과와 동일 계획/payload 검증 결과를 바탕으로 한 비교 입력이 저장된다. 이 실행은 비용 절감이나 solver의 최적성을 미리 주장하지 않는다.

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

[결과](../results/hetero/vlm_placement_load_sweep_seed0/comparison.json). LLM 표본 수는 기존 비율 48건, LLM 중심·양쪽 고부하 242건, VLM 중심 15건으로 특히 마지막 조건의 LLM p90은 탐색적이다.
