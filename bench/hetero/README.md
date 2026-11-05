# bench/hetero — 실측 하네스

모든 스크립트는 저장소 루트에서 실행한다(`cd "$(dirname "$0")/../.."`로 스스로 이동). 파일을 옮기면 이 경로 가정이 깨지므로 이름·위치는 유지하고 역할만 여기 적는다.

## 서버와 부하 생성
| 파일 | 역할 |
|---|---|
| `dit_server.py` | Diffusers `StableDiffusion3Pipeline`을 FastAPI로 감싼 DiT 서버. `/generate`, 단계별 시간(encode/denoise/vae) 반환, `--max-batch` |
| `rr_proxy_server.py` | vLLM 여러 대 앞의 라운드로빈 프록시 (8192) |
| `toy_proxy_server.py` | PD 분할용 prefill→decode 프록시 (vLLM 예제 기반) |
| `loadgen.py` | LLM 부하 생성기(프롬프트 길이 믹스, 동시성, 요약 JSON) |
| `dit_loadgen.py` | DiT 부하 생성기 |
| `controlled_mixed.py` | 플랜(JSON)대로 LLM·DiT 요청을 같은 시계로 재생. Stage 1/2의 요청 기록을 만든다 |
| `traces/` | Azure LLM 추론 트레이스(플랜 생성 재료) |

## 실험 실행 (한 번에 서버 기동 → 재생 → 정리)
| 파일 | 역할 |
|---|---|
| `run_mixed_modal.sh` | 배치 하나로 플랜 재생 (단일 노드, 4090+3090 시절부터 사용) |
| `run_mixed_modal_mn.sh` | 위의 멀티노드 판 (`host:uuid` 역할 지정, ssh 원격 기동) |
| `run_stage1_pilot.py` | Stage 1 전체: protocol 동결, gate, 28회 순차 실행, 상태 기록 |
| `run_stage1_refine.py`, `run_stage2_solo.py`, `run_controlled_batch.py` | Stage 1 보강 격자, Stage 2 단독(resident-solo) 실행, 초기 반복 비교 |
| `profile_mixed_nsys.sh` | 공유 GPU 간섭의 Nsight 캡처. `PLACE=` S4_4090, C_3090, S4_B300, C_B300, S4_B300_MIG; `MPS_PIPE=`로 MPS |
| `pd_smoke.sh` | PD 분할 기동 확인 |
| `pd_xfer_probe.sh`, `pd_xfer_probe_b300.sh`, `pd_xfer_probe_mn.sh` | KV 전송 대역 프로브 (4090→3090 / B300 같은 노드 / B300→PRO 6000) |
| `run_ab.sh` | 9월 A/B 실행(`results/hetero/ab`의 출처). 그 밖의 초기 스크립트는 `archive/legacy/early-2026-10/bench/` |

## 채점·보고서 (판정 정의, 함부로 바꾸지 않음)
| 파일 | 역할 |
|---|---|
| `stage0_report.py` | Stage 0 smoke gate (KV 토큰 동일성 등) |
| `stage1_report.py` | Stage 1 셀 판정과 표 (SLO 격자 × α) |
| `stage2_report.py`, `stage2b_report.py` | resident-solo 비교, k 외삽·solver 유무 비교 |
| `score_goodput.py`, `score_slo_mixed.py`, `map_score.py`, `splitwise_score.py` | goodput·SLO 채점, 결정 지도 점수 |
| `mixed_modal_summary.py`, `compare_controlled.py`, `summarize_ab.py`, `iso_cost.py`, `power_summary.py` | 초기 실험 요약·비교 |

## 분석
| 파일 | 역할 |
|---|---|
| `analyze_nsys.py` | Nsight sqlite에서 decode 스텝·간격·DiT 커널·겹침 집계 (스텝 = graph ≥ 2 ms, 4090 기준) |
| `analyze_nsys_steps.py` | 위와 같되 스텝 문턱을 `STEP_MIN_MS`로 (B300은 1) |
| `analyze_k_model.py`, `analyze_k_timeshare.py` | k 모델 적합과 시간 분할 k 가설(R7) 검증 |
| `analyze_solver_need.py`, `analyze_hol.py` | solver 필요성 분석, HOL 블로킹 분석 |
| `model_llm.py`, `throughput_model.py` | LLM 스텝 시간·처리량 해석 모델 |
| (`vlm/`은 `archive/legacy/vlm-2026-10/`로 이동) | |
| `tests/` | `controlled_mixed.py`, `run_stage1_pilot.py` 단위 테스트 |

## 그림
`plot_nsys_timeline.py`(커널 타임라인), `plot_k_timeshare_explained.py`, `plot_stage1_map.py`, `plot_stage1_seeds.py`, `plot_stage1_motivation.py`, `plot_stage2_explained.py`, `plot_map.py`, `plot_map_groups.py`, `plot_pd_where.py`, `plot_mixed.py`, `plot_ieee.py`, `plot_ieee_pd.py`.
각 그림의 입력·출력은 `docs/solver_need_experiments.md`의 해당 절과 결과 폴더의 README에 있다.
