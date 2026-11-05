#!/usr/bin/env bash
# 열린 도착 실험: 포아송 λ 로 n 개 요청 투입. run_ab.sh 와 같은 구성/서버, loadgen --rate.
#   RATES="0.8 1.6 2.4" MIXES="prefill:2048:128=1.0 mixed:..." N=150 bench/hetero/run_open.sh B
# 결과 results/hetero/open/<CONFIG>/<mix>_r<rate>/
set -uo pipefail
cd "$(dirname "$0")/../.."
CONFIG=${1:?A|B|C|D|S4|S3}
MODEL=${MODEL:-meta-llama/Llama-3.1-8B-Instruct}; V=~/venv/jinwoo/bin
OUT=results/hetero/open/$CONFIG; mkdir -p "$OUT"
G4090=GPU-d5bd27b1-7998-9ae2-b764-3c528e12a47b; G3090=GPU-208c8076-e9ae-fde6-aacf-b0c3fae16a9e
export CUDA_DEVICE_ORDER=PCI_BUS_ID VLLM_USE_FLASHINFER_SAMPLER=0 UCX_NET_DEVICES=all
export UCX_TLS=cuda_copy,tcp,self UCX_TCP_TX_SEG_SIZE=4m UCX_TCP_RX_SEG_SIZE=4m UCX_TCP_SNDBUF=64m UCX_TCP_RCVBUF=64m UCX_MAX_RNDV_LANES=4 UCX_RNDV_THRESH=64k
COMMON=(--dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.85 --no-enable-prefix-caching --max-num-seqs 256)
N=${N:-150}; RATES=${RATES:?"RATES=\"r1 r2 ...\""}; MIXES=${MIXES:-"prefill:2048:128=1.0 mixed:2048:128=0.5,128:1024=0.5 decode:128:1024=1.0"}
PIDS=(); SM=""
cleanup() { echo "cleanup"; kill "${PIDS[@]}" ${SM:-} 2>/dev/null; sleep 4; kill -9 "${PIDS[@]}" 2>/dev/null; pkill -9 -f "toy_proxy_server|rr_proxy_server" 2>/dev/null; }
trap cleanup EXIT
serve() { local gpu=$1 port=$2 role=$3 sp=$4 kv=()
  [ "$role" = producer ] && kv=(--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}')
  [ "$role" = consumer ] && kv=(--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"}')
  CUDA_VISIBLE_DEVICES=$gpu VLLM_NIXL_SIDE_CHANNEL_PORT=$sp $V/vllm serve "$MODEL" --port "$port" "${COMMON[@]}" "${kv[@]}" > "$OUT/serve_$port.log" 2>&1 & PIDS+=($!); }
wait_ready() { for i in $(seq 1 150); do curl -sf "127.0.0.1:$1/health" >/dev/null && { echo "  :$1 ready ($((i*3))s)"; return 0; }; sleep 3; done; echo ":$1 TIMEOUT"; exit 1; }
echo "== open $CONFIG  $(date +%T)"
case $CONFIG in
  A) serve $G4090 8100 full 5600; serve $G3090 8200 full 5601; wait_ready 8100; wait_ready 8200
     $V/python bench/hetero/rr_proxy_server.py --port 8192 --backends 127.0.0.1:8100 127.0.0.1:8200 > "$OUT/proxy.log" 2>&1 & PIDS+=($!) ;;
  B) serve $G4090 8100 full 5600; serve $G3090 8200 full 5601; wait_ready 8100; wait_ready 8200
     $V/python bench/hetero/rr_proxy_server.py --port 8192 --backends 127.0.0.1:8100 127.0.0.1:8200 --weights 2 1 > "$OUT/proxy.log" 2>&1 & PIDS+=($!) ;;
  C) serve $G4090 8100 producer 5600; serve $G3090 8200 consumer 5601; wait_ready 8100; wait_ready 8200
     $V/python bench/hetero/toy_proxy_server.py --port 8192 --prefiller-hosts 127.0.0.1 --prefiller-ports 8100 --decoder-hosts 127.0.0.1 --decoder-ports 8200 > "$OUT/proxy.log" 2>&1 & PIDS+=($!) ;;
  D) serve $G3090 8100 producer 5600; serve $G4090 8200 consumer 5601; wait_ready 8100; wait_ready 8200
     $V/python bench/hetero/toy_proxy_server.py --port 8192 --prefiller-hosts 127.0.0.1 --prefiller-ports 8100 --decoder-hosts 127.0.0.1 --decoder-ports 8200 > "$OUT/proxy.log" 2>&1 & PIDS+=($!) ;;
  S4) serve $G4090 8100 full 5600; wait_ready 8100                      # 4090 단독 (프록시 경유로 오버헤드 동일)
     $V/python bench/hetero/rr_proxy_server.py --port 8192 --backends 127.0.0.1:8100 > "$OUT/proxy.log" 2>&1 & PIDS+=($!) ;;
  S3) serve $G3090 8100 full 5600; wait_ready 8100                      # 3090 단독
     $V/python bench/hetero/rr_proxy_server.py --port 8192 --backends 127.0.0.1:8100 > "$OUT/proxy.log" 2>&1 & PIDS+=($!) ;;
esac
sleep 4; { curl -sf 127.0.0.1:8192/health >/dev/null || curl -sf 127.0.0.1:8192/healthcheck >/dev/null; } || { echo "proxy not up"; exit 1; }
$V/python bench/hetero/loadgen.py --model "$MODEL" --n 8 --mix "512:32=1.0" --tag warm --out "$OUT/_warmup" > /dev/null 2>&1
if [ -n "${TRACE:-}" ]; then MIXES="${TRACE_NAME:-$(basename "$TRACE" .csv)}:trace"; fi
if [ -n "${TRACES:-}" ]; then MIXES="$(for t in $TRACES; do printf '%s:tracefile=%s ' "${t%%=*}" "${t#*=}"; done)${MIXES_EXTRA:-}"; fi
for mixspec in $MIXES; do name=${mixspec%%:*}; mix=${mixspec#*:}
  TRACE_CELL=""; case "$mix" in tracefile=*) TRACE_CELL=${mix#tracefile=} ;; esac
  for r in $RATES; do cell="$OUT/${name}_r${r}"; mkdir -p "$cell"
    [ -s "$cell/summary.json" ] && { echo "skip $cell"; continue; }
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw,clocks.sm,clocks.mem --format=csv,noheader -lms 1000 > "$cell/gpu.csv" 2>/dev/null & SM=$!
    if [ -n "$TRACE_CELL" ]; then
      $V/python bench/hetero/loadgen.py --model "$MODEL" --n "$N" --trace "$TRACE_CELL" --trace-arrivals "${TRACE_ARRIVALS:-poisson}" --rate "$r" --seed 0 --tag "$CONFIG" --out "$cell" 2>&1 | grep -v -E "^(INFO|WARNING)"
    elif [ -n "${TRACE:-}" ]; then
      $V/python bench/hetero/loadgen.py --model "$MODEL" --n "$N" --trace "$TRACE" --trace-arrivals "${TRACE_ARRIVALS:-real}" --rate "$r" --seed 0 --tag "$CONFIG" --out "$cell" 2>&1 | grep -v -E "^(INFO|WARNING)"
    else
      $V/python bench/hetero/loadgen.py --model "$MODEL" --n "$N" --mix "$mix" --rate "$r" --seed 0 --tag "$CONFIG" --out "$cell" 2>&1 | grep -v -E "^(INFO|WARNING)"
    fi
    kill $SM 2>/dev/null; wait $SM 2>/dev/null; SM=""; sleep 5
  done
done
echo "== done $CONFIG  $(date +%T)"
