# Stage 1 파일럿 프로토콜 (고정 비용 capacity map, 2026-10-15 동결)

목적: 고정 GPU 구성에서 기준 mix의 공통 배율 α를 올릴 때 후보별 두 서비스 공동 SLO 충족 한계를 **탐색**한다. seed 901 한 번의 탐색 지도이며 선택기 개선 검증이 아니다. 발견한 경계는 Stage 2에서 새 seed(902 이상)·별도 실행으로 검증한다. 결과 보고 뒤 baseline·가격·SLO 범위를 바꾸지 않는다.

동결 파일: [`results/hetero/stage1_pilot/protocol.json`](../results/hetero/stage1_pilot/protocol.json) (`protocol_sha256 981282a73e6f…`). plan 4개는 `results/hetero/stage1_plans/alpha{0.5,1,1.5,2}_seed901.json`.

## 고정 조건

| 항목 | 값 |
|---|---|
| 후보 7 | S4_4090, S3_3090 (1 GPU) · S4_3090, S3_4090, C_4090, C_3090, C_split (2 GPU). Stage 0 kvfix smoke 13/13 통과 조건부 |
| 모델 | Llama-3.2-3B-Instruct bf16, max-model-len 8192, util 0.5, **KV 4 GiB/engine 고정**(37,440 토큰 기대) · SD3.5-medium 512² 28 step batch 1, T5 제외 |
| KV 물리 점유 | PD 후보(C_*)는 engine 2개 = 두 카드에 4 GiB씩 8 GiB, 단일 engine 후보는 한 카드 4 GiB. 전략의 점유 차이로 비용·메모리 보고에 포함 |
| 수요 | λ0 = (LLM 2 rps, DiT 0.3 rps), code trace. α ∈ {0.5, 1, 1.5, 2} → LLM {1,2,3,4} rps, DiT {0.15,0.3,0.45,0.6} rps |
| 관측 | warmup 60 s 제외, 관측 600 s, drain 최대 300 s |
| plan | seed 901, α별 1개. 같은 α의 모든 후보가 같은 plan·prompt(hash 기록) 재생. replay=both만 (상주형 단독 run 없음) |
| 실행 순서 | α 오름차순, α 안에서 후보 shuffle(`random.Random(901+i)`). 28 run 전부 순차 1회. **성능에 따른 자동 skip·확장 없음** |
| 측정 표본 | α=0.5: LLM 590 / DiT 92. α=1: 1152/166. α=1.5: 1749/258. α=2: 2336/346 (DiT ≥ 60 충족) |
| plan 검증 | 실제 토큰(prompt 재생성·재토큰화)으로 in+out ≤ 8192, out ≥ 1 확인. 최대 in+out 7,733. 위반 0. 기존 `controlled_mixed.prepare`의 규칙(ContextTokens<8→8, >8000→8000)이 적용된 행: α별 0/5/5/6건 — 원 trace는 수정하지 않았고 규칙과 건수를 기록 |

## SLO 정의와 격자 (전체 192 셀 보고)

- TTFT = 계획 도착 → 첫 content chunk (client send lag 포함, p99 별도 기록).
- TPOT = (end − first)/(completion_tokens − 1). **completion_tokens = 1이면 정의 불가**(생성 간격 없음): 그 요청은 TTFT 조건만 적용하고 건수를 보고. tail ITL 아님.
- DiT E2E = 계획 도착 → 이미지 완료.
- 충족률 분모 = 측정 요청 전체(오류·drain_timeout·미완료 포함). LLM 요청은 TTFT·TPOT **동시** 만족.
- 셀 통과 = 두 서비스 모두 목표 충족률 이상. 격자: TTFT {0.3,0.5,1,2} s × TPOT {12,20,30,50} ms × DiT {3,5,10,20} s × 목표 {0.9,0.95,0.99}.
- α_max(셀) = 시험한 α 중 관측상 통과한 최대값(`observed`). 실제 용량의 등호·상한·신뢰한계가 아니다. α=2에서 통과하면 `upper-unmeasured`, 통과가 없으면 시험 범위의 통과 관측 없음으로 표시한다. α별 pass/fail과 미측정을 전부 표기하고, 비단조(낮은 α 실패·높은 α 통과)는 `NM`으로 표시한다.
- 성공 요청 p50/p90은 별도 보조 지표이며 충족률 판정이 아님. 99 % 목표는 DiT 90건대 표본에서 점추정(실패 1건이면 깨짐).

## 안정성·실패 분류

200 s × 3 창마다 서비스별 계획 도착 수, 성공 완료 수, 실패 종료 수, 창 끝 in-flight. 관측 종료 시 backlog, drain 중 완료 수, drain 사용 시간, drain_timeout 수. 마지막 창 in-flight > 첫 창 and timeout 존재면 "포화 의심"(판정 아님). 고부하 timeout은 유효 결과로 보존하며 infra 실패(exit≠0, manifest 없음)와 분리. 같은 후보 재실행 없음.

## 비용·자원

1차: 점유 카드 종류·수 × 관측 600 s GPU-time, 카드별 Wh(관측 창 / drain 포함 전체 따로). 2차: p_4090=1 단위의 시간당 비용 n_4090 + r·n_3090, r ∈ {0.5, 0.75, 0.94, 1.0}. **모두 연구 가정**이며 임대 시세 주장 아님. 0.94는 출시가 1499/1599 상각 근사 예시. 유휴 카드 과금 포함·제외 둘 다 보고(포함 시 1 GPU 후보 비용 = 2 GPU 후보). 고정 비용 비교는 같은 카드 구성(2 GPU 5종)끼리, 1 GPU 후보는 가격 모델로 비교. 시세표를 확보하면 별도 분석 버전으로 추가한다(이 프로토콜 교체 아님).

## 실행 절차와 gate

1. **Smoke gate** (`run_stage1_pilot.py gate`): `stage0_smoke_kvfix` batch_status `complete`, 13 run 모두 `done`+manifest, 요청 오류 0, deployment.json `kv_cache_memory_bytes == 4294967296`, 필요 서버(S*: 8100, C: 8100+8200) 모두 KV 37,440 토큰(누락·미파싱은 실패), replay 종류별 strict parity(plan hash·prompt hash·CSV 열·분모가 각각 정확히 1개의 비어 있지 않은 값).
2. 동일 출력의 `runner.lock` 및 전역 GPU lock `results/hetero/gpu_batch.lock`(flock) + `nvidia-smi --query-compute-apps` 비어 있음 확인. 남의 PID는 절대 죽이지 않고 blocked 상태만 기록. 하네스 파일 hash가 protocol과 다르면 blocked(`--allow-harness-drift`로 기록 후 진행 가능).
3. run마다: 하네스 drift와 plan hash 재확인 → 기존 `done`은 skip, done 없는 출력 존재 시 blocked(삭제 없음) → 환경변수 명시 통제(상속된 REPLAY_ONLY/ONLY/CONTROLLED_*/DRAIN_TIMEOUT/KV_BYTES/GPU_UTIL/SIZE/STEPS/MODEL/DMODEL/N/LRATES/DRATES/WORK/TRACE/ND 제거 후 protocol 값 설정) → `bash bench/hetero/run_mixed_modal.sh <LLM> <DIT>`를 새 세션(process group)으로 실행, 콘솔 로그 `<out>/<cand>/<plan>.console.log`, exit 코드 파일 → 종료 후 KV strict·plan/prompt hash·CSV 요청 ID/도착 시각/측정 구간/분모/성공 출력 길이 검사(불일치 시 중단·결과 보존) → compute-apps가 60 s 안에 비지 않으면 blocked(죽이지 않음) → `status/<run_id>.json`·`batch_status.json` 원자적 기록 → 증분 보고서 갱신.
4. infra 실패 2회 누적이면 중단. SIGTERM/SIGINT/SIGHUP 수신 시 자식 process group에 SIGTERM 전달(기존 cleanup trap이 서버 정리) 후 `aborting` 기록.
5. 종료 후 상주형 단독 실험을 자동으로 잇지 않는다(후속 보류, 각 후보 상주 상태로 별도 설계).

명령(사용자의 실험 진행 승인에 따라 Codex 최종 검토 후 실행):

```
# gate만 확인
~/venv/jinwoo/bin/python bench/hetero/run_stage1_pilot.py gate
# 순서·명령·env·gate 확인 (GPU 미사용)
~/venv/jinwoo/bin/python bench/hetero/run_stage1_pilot.py dry-run
# 실행 (smoke lock 해제를 기다린 뒤 gate 통과 시에만 시작; 세션과 무관하게 nohup)
cd /home/jinwoo.kim27/mllm && nohup ~/venv/jinwoo/bin/python -u bench/hetero/run_stage1_pilot.py run --wait-smoke \
  > results/hetero/stage1_pilot/runner.log 2>&1 &
# 진행 중 언제든
~/venv/jinwoo/bin/python bench/hetero/stage1_report.py results/hetero/stage1_pilot
```

산출물: `results/hetero/stage1_pilot/{protocol.json, smoke_gate.json, batch_status.json, status/*.json, runner.log, <cand>/<plan>/…, stage1_summary.json, stage1_summary.md, stage1_cells.csv}`.

## CPU 검증 (GPU 미사용)

`~/venv/jinwoo/bin/python -m unittest bench/hetero/tests/test_stage1_pilot.py bench/hetero/tests/test_controlled_mixed.py` → 12 tests OK. 합성 요청으로 확인한 것: 오류·timeout이 분모에 남는 공동 충족률, n=1 TPOT 비적용, 한 서비스만 느릴 때 공동 실패, 200 s 창·drain 완료·timeout 집계, KV 누락/미파싱/빈 값/바이트 불일치가 모두 gate 실패, strict parity가 빈 값 거부, plan 생성 시 8192 초과 거부·clamp 건수 보고, 재-prepare 멱등성과 상수 변경 시 동결 거부, α 비단조 `NM`·상한 미측정·미측정 셀·무효 run(KV/plan 불일치) 제외·비용표·Wh 적분, dry-run이 28 run 명령과 env를 출력하고 상속 REPLAY_ONLY 등을 제거, smoke gate 13/13·KV·done 검사. 실제 prepare는 venv 토크나이저로 실행했고 dry-run 실행 중 GPU는 건드리지 않았다.

## 기존 파일 변경 제안 (smoke 완료 전 적용 금지, diff만)

- `run_controlled_batch.py`: 출력 폴더별 `batch.lock` 외에 전역 `results/hetero/gpu_batch.lock`도 잡고 시작 전 compute-apps를 확인하면 서로 다른 batch의 GPU 동시 점유를 막을 수 있다. Stage 1 러너는 이미 전역 lock을 잡으므로 이후 batch 도구에 동일 규칙 적용 권장.
- `stage0_report.py`의 KV 동일성 검사(`identical=len(distinct)<=1`)는 빈 값도 통과한다. `stage1_report.strict_kv_check`(서버별 기대값·바이트 필수) 사용으로 교체 권장. 기존 smoke 결과 파일은 수정하지 않는다.

## 실행 시작

2026-10-15 12:57 KST, 사용자 승인에 따라 seed 901의 혼합 28회 실행을 백그라운드에서 시작했다. PID와 명령은 `results/hetero/stage1_pilot/launch.json`, 상태는 `batch_status.json`, 로그는 `runner.log`에 기록한다. 시작 시 strict smoke gate 통과, GPU compute 프로세스 없음, 하네스 drift 없음, CPU 테스트 12건 통과를 확인했다. 첫 후보는 S3_3090, α=0.5다. 진행·완료 여부는 상태 파일을 기준으로 확인하며 이 시작 기록을 완료 결과로 해석하지 않는다. 로드·drain 길이에 따라 전체 소요시간은 수 시간이며 약 6–9시간 추정이다.

Codex 최종 검토로 요청 로그 무결성(계획 ID·도착·측정 구간·분모·성공 출력 길이) 검사와 출력별 runner lock, 프로토콜 본문 hash 및 매 run 하네스 drift 검사를 보완했다. 기존 무효 run은 정상 run의 parity 집계에서 제외한다. 실패·timeout은 분모에 남기고 성공 출력에만 길이 검사를 적용한다. 수정 전 protocol은 `protocol.pre_codex_review.json`으로 보존했고, 실험 조건 hash는 유지한 채 실행 전 source hash를 다시 기록했다.
