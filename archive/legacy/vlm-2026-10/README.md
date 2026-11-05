# LLM+VLM 공유·배치 실험 (2026-10, 보관)

VLM(encoder + LLM)을 LLM과 같은 GPU에 두거나 encoder를 다른 카드로 보냈을 때의 간섭·배치 비교. 4090+3090, seed 0.
2026-11-05 가지치기에서 VLM은 실측 범위에서 제외(시뮬 확장으로 이관)하기로 해 여기로 옮겼다. 코드는 `bench/vlm/`, 결과는 `results/`.
원래 경로는 `bench/hetero/vlm/`, `results/hetero/vlm_*`였고, 실행 방법은 `bench/vlm/README.md`에 있다(경로는 옮기기 전 기준).
