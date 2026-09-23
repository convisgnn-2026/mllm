# PD KV 전송 대역폭 프로브 — 4090(P) → 3090(D), NIXL/UCX (2026-09-23)

`bench/hetero/pd_xfer_probe.sh`. 2048 토큰 프롬프트(KV 268 MB, GQA 128 KB/토큰) 요청을 순차 8건 / 동시 16건 보내고
디코더의 `KV Transfer metrics`(요청당 전송 시간)를 읽음. vLLM 0.28.0 NixlConnector, NIXL 1.4.1 번들 UCX.

| 설정 (`UCX_TLS` 등) | 순차 전송 ms | 실효 GB/s | 동시 16건 평균/p90 ms | 동시 16건 총 시간 | 비고 |
|---|---|---|---|---|---|
| 기본 (`UCX_NET_DEVICES=all`) | 504 | 0.53 | 2,884 / 4,302 | 8.8 s | 전송 직렬화 |
| `cuda_copy,cma,posix,sysv,self` (TCP 제외) | — | — | — | — | **연결 실패**: "no active messages transport … no peer failure handler" |
| `cuda_copy,posix,sysv,self` | — | — | — | — | 위와 동일 실패 |
| `cuda_copy,tcp,self` | 447 | 0.60 | 4,465 / 5,210 | 7.9 s | 기본과 같음 → 기본도 TCP 였음 |
| `cuda_copy,tcp,cma,self` | 476 | 0.56 | 5,388 / 6,149 | 8.4 s | CMA 는 데이터 경로에 안 쓰임 |
| **TCP 튜닝** `TX/RX_SEG_SIZE=4m SNDBUF/RCVBUF=64m MAX_RNDV_LANES=4 RNDV_THRESH=64k` | **110** | **2.4** | **130 / 147** | **3.8 s** | 채택 |
| 튜닝 + cma | 115 | 2.3 | 130 / 146 | 3.8 s | 동일 |
| 튜닝2 `8m / 128m / lanes 8 / rails 8` | 108 | 2.5 | (동일) | 3.8 s | 상한 |

## 결론

- NIXL 은 상대 장애 감지가 되는 전송(TCP, IB)을 요구한다. 공유메모리(posix/sysv/cma)만으로는 엔드포인트가 안 만들어진다.
- 4090↔3090 은 P2P(`cuda_ipc`)가 없으므로 KV 는 GPU→호스트→TCP 루프백→호스트→GPU 로 간다.
  UCX 기본 TCP 설정(작은 세그먼트, 단일 레인)은 **0.5 GB/s** 이고 동시 요청에서 직렬화된다.
- 세그먼트 4 MB · 소켓 버퍼 64 MB · rendezvous 레인 4 로 **2.4 GB/s**, 동시 16건에서도 요청당 130 ms. 그 이상은 상한.
- 어제 `torch.copy_` 로 잰 호스트 경유 PCIe 9.4 GB/s 대비 1/4. 남은 차이는 TCP 루프백 복사 비용. IB 나 P2P 가 있는
  노드(B300 ×8)에서는 이 벽이 없다 → **같은 노드 안이라도 이기종 카드 사이의 전송 비용은 solver 가 봐야 할 제약**.
- 8B 2K 프롬프트 기준 링크 상한 = 2.4 GB/s ÷ 268 MB ≈ **9 req/s** (hetsim 의 `λ·KV ≤ BW` 제약값).

## 추가: cross-layers 블록 (2026-09-23 오후)

`enable_cross_layers_blocks: true` 는 32 레이어의 KV 블록을 하나의 연속 텐서로 묶어 전송 단위를 64 KB → 2 MB 로 키운다.
**vLLM 0.28.0 기본 V2 모델 러너(`v1/worker/gpu/`)는 이 옵션을 무시한다** (`kv_connector.py`: TODO). `VLLM_USE_V2_MODEL_RUNNER=0` 로
구 러너를 써야 활성화되며, 활성 여부는 NIXL 호환성 해시(`a428f594…` → `ce985a45…`)와 `remote_block_len=2097152` 로 확인.

| 구 러너 + TCP 튜닝 | 순차 전송 (p90) | 동시 16건 (p90) | TTFT 순차 |
|---|---|---|---|
| cross off | 117 (146) ms | 130 (146) ms | 408 ms |
| **cross on** | **103 (109) ms** | **113 (124) ms** | **371 ms** |

평균 12%, p90 25% 개선. 실효 2.6 GB/s. 러너 자체(v2→v1)의 차이는 없음(off 끼리 117 vs 110~115 ms).
