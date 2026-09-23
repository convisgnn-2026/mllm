#!/usr/bin/env bash
# PD 분할 스모크 테스트: 프리필러(4090) → NIXL → 디코더(3090), 프록시 경유로 요청 2개 통과시키기.
# vLLM 0.28.0 NixlConnector (docs/features/nixl_connector_usage.md 레시피). 끝나면 세 프로세스를 모두 정리한다.
set -uo pipefail
cd "$(dirname "$0")/../.."
MODEL=${MODEL:-meta-llama/Llama-3.1-8B-Instruct}
V=~/venv/jinwoo/bin
LOG=${LOG:-results/hetero/pd_smoke}; mkdir -p "$LOG"
P_GPU=GPU-d5bd27b1-7998-9ae2-b764-3c528e12a47b   # 4090
D_GPU=GPU-208c8076-e9ae-fde6-aacf-b0c3fae16a9e   # 3090
export CUDA_DEVICE_ORDER=PCI_BUS_ID VLLM_USE_FLASHINFER_SAMPLER=0 UCX_NET_DEVICES=all
COMMON=(--dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.85 --no-enable-prefix-caching)
EAGER=${EAGER---enforce-eager}   # 문서 예제대로 eager. 성능 실험에서는 EAGER="" 로 CUDA graph 사용을 시도.

cleanup() { echo "cleanup"; kill $P_PID $D_PID $X_PID 2>/dev/null; sleep 3; kill -9 $P_PID $D_PID $X_PID 2>/dev/null; }
trap cleanup EXIT

CUDA_VISIBLE_DEVICES=$P_GPU VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
  $V/vllm serve "$MODEL" --port 8100 "${COMMON[@]}" $EAGER \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}' \
  > "$LOG/prefill.log" 2>&1 &  P_PID=$!
CUDA_VISIBLE_DEVICES=$D_GPU VLLM_NIXL_SIDE_CHANNEL_PORT=5601 \
  $V/vllm serve "$MODEL" --port 8200 "${COMMON[@]}" $EAGER \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"}' \
  > "$LOG/decode.log" 2>&1 &  D_PID=$!

wait_ready() { local port=$1 name=$2 pid=$3; for i in $(seq 1 120); do
  curl -sf "127.0.0.1:$port/health" >/dev/null && { echo "$name ready ($((i*3))s)"; return 0; }
  kill -0 $pid 2>/dev/null || { echo "$name DIED"; tail -5 "$LOG/$name.log"; return 1; }; sleep 3; done
  echo "$name TIMEOUT"; return 1; }
wait_ready 8100 prefill $P_PID || exit 1
wait_ready 8200 decode  $D_PID || exit 1

$V/python bench/hetero/toy_proxy_server.py --port 8192 --prefiller-hosts 127.0.0.1 --prefiller-ports 8100 \
  --decoder-hosts 127.0.0.1 --decoder-ports 8200 > "$LOG/proxy.log" 2>&1 &  X_PID=$!
sleep 3

for i in 1 2; do
  t0=$(date +%s.%N)
  curl -s 127.0.0.1:8192/v1/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"prompt\":\"The three laws of thermodynamics are\",\"max_tokens\":32,\"temperature\":0}" \
    | $V/python -c "import sys,json; d=json.load(sys.stdin); print('OUT:', repr(d['choices'][0]['text'][:80]), '| usage', d.get('usage'))"
  echo "req$i wall $(echo "$(date +%s.%N) - $t0" | bc) s"
done
echo "--- transfer evidence (decode.log)"; grep -i -E "nixl|kv.*transfer|remote|handshake" "$LOG/decode.log" | grep -v -i warn | tail -5 | cut -c1-200
echo "--- proxy"; tail -3 "$LOG/proxy.log" | cut -c1-160
echo "SMOKE OK"
