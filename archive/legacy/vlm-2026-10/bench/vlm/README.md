# VLM encoder 分리 기능 검증

연구 대상은 LLM·VLM·DiT 모두 유지한다. 현재 4090+3090 환경에서 단계 제어 장치를 시험하기 위해 VLM encoder 분리 실행부터 확인한다.

- 모델: Qwen/Qwen2-VL-2B-Instruct (공개 모델, 최초 다운로드 필요).
- Encoder: 3090, encoder-only producer, eager.
- 언어 prefill/decode: 4090, EC consumer, eager.
- 전달: ECExampleConnector의 파일 기반 공유 저장소.
- 검사: 서로 다른 두 합성 이미지에 대한 답변 생성, encoder cache 파일 생성, 프로세스 GPU 배치, 상세 로그.

```bash
~/venv/jinwoo/bin/python bench/hetero/vlm/run_encoder_check.py
```

GPU 접근 가능한 환경에서 실행한다. 다운로드와 smoke 실행 상태는 results/hetero/vlm_encoder_check_status.json, 실제 기능 검증 결과는 results/hetero/vlm_encoder_smoke/에 저장한다. 기존 출력은 덮어쓰지 않는다.

이는 **기능 검증**이다. 파일 기반 전달은 성능 최적화 전송 경로가 아니다. 답변 생성만으로 consumer가 encoder를 다시 실행하지 않았다는 것을 확정하지 않으며, cache load 로그와 proxy metadata rewrite를 함께 확인해야 한다. 미확인 단계는 성공으로 판정하지 않는다. 성능 이점·LLM 공유 간섭·solver 효과는 후속 실험이다.

## 공식 예제 출처

- disagg_epd_proxy.py 및 disagg_1e1pd_example.sh: vLLM v0.28.0, https://github.com/vllm-project/vllm/tree/v0.28.0/examples/disaggregated/disaggregated_encoder
- 공식 문서: https://docs.vllm.ai/en/latest/features/disagg_encoder/

공식 파일의 Apache-2.0 헤더를 유지했다. 공식 shell 예제는 참고용이며 실행하지 않는다. smoke_encoder_split.py가 프로세스 group 관리와 정리를 수행한다.

### Encoder 실행 시점 제어 파일럿

`run_interference_pilot.py --control-study --seed 0 --out results/hetero/vlm_encoder_control_seed0`는 high 공유 부하를 자유 실행 / 직렬 실행 / 고정 20ms 휴식 / LLM chunk 지연에 따른 최대 약 100ms 양보로 비교한다. 추가 seed 반복은 없다. 새 요청 기록은 chunk 시각을 포함하고 모든 조건에서 100ms 주기로 동일하게 피드백을 보낸다. 정책 대기 상한은 큐 전체 대기 상한이 아니다. 상세 가설과 판정은 [실험 문서](../../../docs/solver_need_experiments.md)를 참고한다.

`analyze_encoder_overlap.py <결과 폴더>`는 primer 구간과 LLM 요청·chunk 간격의 연관성을 저장한다. 순수 GPU encoder 시간이나 인과관계로 해석하지 않는다.

`--timing-study --duration 60 --seed 0 --out results/hetero/vlm_signal_timing_seed0`는 직렬 기준 / 지연 후 양보 / 첫 응답 전 보호를 비교한다. `prefill_guard`는 실제 GPU prefill 계측이 아니라 요청 전송→첫 content 응답 상태를 사용한다. `analyze_signal_timing.py <결과 폴더>`는 피드백 시각·정책 대기·primer 구간의 관계를 기록한다.
