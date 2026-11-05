#!/usr/bin/env bash
# B300 variant (2026-11-05) of pd_xfer_probe.sh: P=B300 GPU3 → D=B300 GPU1 (same node), 3B model, venv ~/venv/mllm.
# PD KV 전송 대역폭 프로브: 4090(P) → 3090(D) 로 2048 토큰 프롬프트(KV 268 MB) 요청 n 개를 보내고
# 디코더가 보고하는 "KV Transfer metrics" 의 전송 시간을 읽어 실효 GB/s 를 계산한다.
#   UCX_TLS="cuda_copy,cma,sm,tcp" UCX_LOG_LEVEL=info bench/hetero/pd_xfer_probe.sh tag
set -uo pipefail
cd "$(dirname "$0")/../.."
TAG=${1:-default}
MODEL=${MODEL:-meta-llama/Llama-3.2-3B-Instruct}
V=~/venv/mllm/bin; export PATH=$HOME/venv/mllm/bin:$PATH HF_HUB_OFFLINE=1
LOG=results/hetero/pd_xfer_b300/$TAG; mkdir -p "$LOG"
P_GPU=${P_GPU:-GPU-55867e89-32c1-52ae-b925-34a116ba798d}; D_GPU=${D_GPU:-GPU-1b5a091d-563d-e5ed-8201-2f3a95e09483}
export CUDA_DEVICE_ORDER=PCI_BUS_ID VLLM_USE_FLASHINFER_SAMPLER=0
EXTRA=""; [ "${CROSS:-0}" = 1 ] && EXTRA=',"kv_connector_extra_config":{"enable_cross_layers_blocks":"true"}'
KVP="{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_producer\",\"kv_load_failure_policy\":\"fail\"$EXTRA}"
KVD="{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_consumer\",\"kv_load_failure_policy\":\"fail\"$EXTRA}"
COMMON=(--dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.5 --no-enable-prefix-caching --max-num-seqs 64)
PIDS=(); cleanup() { kill "${PIDS[@]}" 2>/dev/null; sleep 4; kill -9 "${PIDS[@]}" 2>/dev/null; true; }; trap cleanup EXIT
env | grep -E "^UCX_" | sed 's/^/  env /'

CUDA_VISIBLE_DEVICES=$P_GPU VLLM_NIXL_SIDE_CHANNEL_PORT=5600 $V/vllm serve "$MODEL" --port 8100 "${COMMON[@]}" \
  --kv-transfer-config "$KVP" > "$LOG/prefill.log" 2>&1 & PIDS+=($!)
CUDA_VISIBLE_DEVICES=$D_GPU VLLM_NIXL_SIDE_CHANNEL_PORT=5601 $V/vllm serve "$MODEL" --port 8200 "${COMMON[@]}" \
  --kv-transfer-config "$KVD" > "$LOG/decode.log" 2>&1 & PIDS+=($!)
for port in 8100 8200; do for i in $(seq 1 100); do curl -sf 127.0.0.1:$port/health >/dev/null && break; sleep 3; done; done
$V/python bench/hetero/toy_proxy_server.py --port 8192 --prefiller-hosts 127.0.0.1 --prefiller-ports 8100 --decoder-hosts 127.0.0.1 --decoder-ports 8200 > "$LOG/proxy.log" 2>&1 & PIDS+=($!); sleep 3

# 1) 순차 8건 (경합 없음) → 요청당 전송 시간   2) 동시 16건 → 경합 시
$V/python bench/hetero/loadgen.py --model "$MODEL" --n 8 --mix "2048:16=1.0" --max-concurrency 1 --tag seq --out "$LOG/seq" 2>&1 | grep "^\["
sleep 12   # 메트릭 로그 주기(10 s) 대기
echo "seq   : $(grep -o 'KV Transfer metrics:.*P90 xfer time (ms)=[0-9.]*' "$LOG/decode.log" | tail -1 | cut -c1-120)"
$V/python bench/hetero/loadgen.py --model "$MODEL" --n 16 --mix "2048:16=1.0" --tag conc --out "$LOG/conc" 2>&1 | grep "^\["
sleep 12
echo "conc  : $(grep -o 'KV Transfer metrics:.*P90 xfer time (ms)=[0-9.]*' "$LOG/decode.log" | tail -1 | cut -c1-120)"
echo "cross-layers: $(grep -o -i 'cross.layers[^,]*' "$LOG/decode.log" | head -2 | tr '\n' ' ')"
echo "--- UCX transport lines (decode)"; grep -i -E "ucx.*(tl|transport|using|selected)|cuda_copy|cuda_ipc|cma|posix|sysv|tcp|net_device|iface" "$LOG/decode.log" | grep -v -i "UCX_RCACHE" | head -12 | cut -c1-220
echo "PROBE DONE $TAG"
