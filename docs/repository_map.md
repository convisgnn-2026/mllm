# 저장소 구조와 정리 이력

## 구조 (2026-11-05)

| 경로 | 역할 |
|---|---|
| `bench/hetero/` | 실측 하네스·서버·채점·분석·그림 코드. 파일별 역할은 `bench/hetero/README.md` |
| `results/hetero/` | 실험 결과. 폴더 하나가 실험 하나. 요약(json/md)·그림·로그·요청 기록만 git에 있고, Nsight 원본(`.nsys-rep`)과 sqlite는 `.gitignore` |
| `docs/` | 실험 기록 `solver_need_experiments.md`, 프로토콜 `stage1_protocol.md`, 환경 `research_env.md`, 포팅 `handoff_b300.md`·`b300_playbook.md` |
| `docs/notes/` | 미팅(`1013_meeting.md`)·메일(`1102_mail*.md`)·리뷰(`research_discussion/`, `claude_research_handoff_20261015.md`, `cost_efficiency_paper_comparison_20261015.md`)·이전 정리 이력(`reorganization.json`) |
| `env/` | venv 패키지 고정 목록 (`requirements-b300-freeze.txt` 현재, `jinwoo-venv-freeze.txt` 4090 서버) |
| `papers/` | 참고 논문 PDF(`refs/`)와 외부 코드(`code/tetriserve`) |
| `presentations/` | 발표 자료(pptx/pdf)와 슬라이드 생성 코드 |
| `archive/legacy/` | 이전 연구 보존. `solver/`·`hetsim/`·`scripts/`(MIG·시뮬레이터), `vlm-2026-10/`(LLM+VLM 실험, 실측 범위에서 제외), `early-2026-10/`(Stage 1 이전 개방 루프·프로파일) |

## 정리 이력

- **2026-11-05 (2차).** VLM 실험 7개 폴더와 `bench/hetero/vlm/`을 `archive/legacy/vlm-2026-10/`로, 초기 개방 루프·프로파일(`open`, `profile_llm`, `mixed_summary`, `uncontended`)과 스크립트 5개를 `archive/legacy/early-2026-10/`로 이동. 참조 없는 `pd_smoke_graph`, `solver_v0`, `vlm_encoder_smoke_missing_embeds_flag`, `stage2_b` 삭제. 문서 링크는 새 경로로 수정.
- **2026-11-05.** 실패한 시도의 결과 폴더(nsys 2025.6.3 캡처, 오염된 MPS dit, 노드 간 프로브 실패 8개)와 재생성 가능한 sqlite 내보내기를 삭제. 어디서도 참조하지 않는 옛 스크립트 16개 삭제(`open_eval.py`, `plot_ab.py`, `plot_controlled_tradeoff.py`, `plot_decision.py`, `plot_mixed_modal*.py`, `plot_mixed_summary.py`, `plot_model.py`, `plot_power.py`, `plot_throughput_model.py`, `plot_tradeoff_simple.py`, `slo_map_mixed.py`, `solver_v0.py`, `summarize_profile.py`, `uncontended_ref.sh`, `analyze_nsys_node.py`). 미팅·메일·리뷰 문서를 `docs/notes/`로, `mllm_v0.6.pptx`를 `presentations/`로 이동하고 링크 수정. README를 현재 상태로 다시 씀. 모두 git 이력에서 복구 가능.
- **2026-10-15.** 이전 연구 자산(`solver/`, `hetsim/`, `scripts/`, `slo_configs/`, `figures/`, MIG harness, 비 hetero 결과)을 `archive/legacy/`로 이동. 이동 목록은 `docs/notes/reorganization.json`. 발표 슬라이드 JPG 미리보기와 Python cache 제거.
