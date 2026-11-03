#!/usr/bin/env bash
# Diagnostic (not a protocol run): kernel-level timeline of the two processes that share one GPU.
#   PLACE=C_3090  (default) P/D split, decode vLLM (:8200) + dit_server (:8300) share the RTX 3090; prefill (4090) untraced
#   PLACE=S4_4090 single vLLM engine (:8100) + dit_server (:8300) share the RTX 4090
# The two sharing processes are started under one `nsys launch` session so they land in a single report; capture is a
# 90 s window after the plan's warmup.
#   [PLACE=C_3090|S4_4090] bench/hetero/profile_mixed_nsys.sh mixed|llm|dit [OUT_ROOT]
# Same servers, ports, KV budget and plan (alpha 1.0, seed 901) as run_mixed_modal.sh; the replay is stopped after
# the capture window (+10 s), so request records are partial and must not be used as protocol results.
set -uo pipefail
cd "$(dirname "$0")/../.."
MODE=${1:?mixed|llm|dit}; PLACE=${PLACE:-C_3090}; case $PLACE in C_3090) DEF=results/hetero/nsys_c3090; SHARED_GPU=1 ;; S4_4090) DEF=results/hetero/nsys_s4_4090; SHARED_GPU=0 ;; *) echo "bad PLACE=$PLACE"; exit 1 ;; esac
ROOT_OUT=${2:-$DEF}
OUT=$ROOT_OUT/$MODE; [ ! -e "$OUT" ] || { echo "output exists: $OUT"; exit 1; }; mkdir -p "$OUT"
NSYS=${NSYS:-$HOME/tools/nsys/opt/nvidia/nsight-systems-cli/2026.5.1/target-linux-x64/nsys}
V=~/venv/jinwoo/bin; MODEL=meta-llama/Llama-3.2-3B-Instruct; DMODEL=stabilityai/stable-diffusion-3.5-medium
PLAN=results/hetero/stage1_plans/alpha1_seed901.json
WARMUP=60; PRE=30; CAP=${CAP:-90}; SESSION=${PLACE,,}_$MODE
G4090=GPU-d5bd27b1-7998-9ae2-b764-3c528e12a47b; G3090=GPU-208c8076-e9ae-fde6-aacf-b0c3fae16a9e
export CUDA_DEVICE_ORDER=PCI_BUS_ID VLLM_USE_FLASHINFER_SAMPLER=0 UCX_NET_DEVICES=all HF_HUB_OFFLINE=1
export UCX_TLS=cuda_copy,tcp,self UCX_TCP_TX_SEG_SIZE=4m UCX_TCP_RX_SEG_SIZE=4m UCX_TCP_SNDBUF=64m UCX_TCP_RCVBUF=64m UCX_MAX_RNDV_LANES=4 UCX_RNDV_THRESH=64k
KV_BYTES=4294967296
COMMON="--dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.5 --kv-cache-memory-bytes $KV_BYTES --no-enable-prefix-caching --max-num-seqs 256"
KVP='{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}'
KVC='{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"}'
PIDS=(); DM=""
cleanup() { echo "cleanup $(date +%T)"; "$NSYS" stop --session="$SESSION" >/dev/null 2>&1 || true
  for pid in "${PIDS[@]}"; do kill -- -"$pid" 2>/dev/null || true; done; [ -z "$DM" ] || kill "$DM" 2>/dev/null || true
  sleep 5; for pid in "${PIDS[@]}"; do kill -9 -- -"$pid" 2>/dev/null || true; done; }
trap cleanup EXIT
wait_ready() { for i in $(seq 1 150); do curl -sf "127.0.0.1:$1/health" >/dev/null && { echo "  :$1 ready ($((i*3))s)"; return 0; }; sleep 3; done; echo ":$1 TIMEOUT"; exit 1; }
echo "== nsys diagnostic $PLACE mode=$MODE $(date +%T)"; echo "$SHARED_GPU" > "$OUT/dmon_gpu_index.txt"
if [ "$PLACE" = C_3090 ]; then
  # prefill vLLM on the 4090 (not traced)
  CUDA_VISIBLE_DEVICES=$G4090 VLLM_NIXL_SIDE_CHANNEL_PORT=5600 setsid "$V/vllm" serve "$MODEL" --port 8100 $COMMON --kv-transfer-config "$KVP" > "$OUT/serve_8100.log" 2>&1 & PIDS+=($!)
  # decode vLLM (3090) + dit_server (3090) under one nsys session, collection initially off
  setsid "$NSYS" launch --session-new="$SESSION" --trace=cuda,nvtx --cuda-graph-trace=graph \
    bash -c "CUDA_VISIBLE_DEVICES=$G3090 VLLM_NIXL_SIDE_CHANNEL_PORT=5601 $V/vllm serve $MODEL --port 8200 $COMMON --kv-transfer-config '$KVC' > $OUT/serve_8200.log 2>&1 &
             CUDA_VISIBLE_DEVICES=$G4090,$G3090 $V/python bench/hetero/dit_server.py --model $DMODEL --port 8300 --enc-dev cuda:1 --dit-dev cuda:1 --vae-dev cuda:1 --size 512 --steps 28 > $OUT/dit_server.log 2>&1 &
             wait" > "$OUT/nsys_launch.log" 2>&1 & PIDS+=($!)
  wait_ready 8100; wait_ready 8200
  setsid "$V/python" bench/hetero/toy_proxy_server.py --port 8192 --prefiller-hosts 127.0.0.1 --prefiller-ports 8100 --decoder-hosts 127.0.0.1 --decoder-ports 8200 > "$OUT/proxy.log" 2>&1 & PIDS+=($!)
else
  # single full vLLM engine (4090) + dit_server (4090) under one nsys session
  setsid "$NSYS" launch --session-new="$SESSION" --trace=cuda,nvtx --cuda-graph-trace=graph \
    bash -c "CUDA_VISIBLE_DEVICES=$G4090 VLLM_NIXL_SIDE_CHANNEL_PORT=5600 $V/vllm serve $MODEL --port 8100 $COMMON > $OUT/serve_8100.log 2>&1 &
             CUDA_VISIBLE_DEVICES=$G4090,$G3090 $V/python bench/hetero/dit_server.py --model $DMODEL --port 8300 --enc-dev cuda:0 --dit-dev cuda:0 --vae-dev cuda:0 --size 512 --steps 28 > $OUT/dit_server.log 2>&1 &
             wait" > "$OUT/nsys_launch.log" 2>&1 & PIDS+=($!)
  wait_ready 8100
  setsid "$V/python" bench/hetero/rr_proxy_server.py --port 8192 --backends 127.0.0.1:8100 > "$OUT/proxy.log" 2>&1 & PIDS+=($!)
fi
sleep 4; { curl -sf 127.0.0.1:8192/health >/dev/null || curl -sf 127.0.0.1:8192/healthcheck >/dev/null; } || { echo "proxy not up"; exit 1; }
"$V/python" bench/hetero/loadgen.py --model "$MODEL" --n 8 --mix "512:32=1.0" --tag warm --out "$OUT/_warmup" > /dev/null 2>&1
for i in $(seq 1 200); do grep -q READY "$OUT/dit_server.log" && { echo "  dit ready ($((i*3))s)"; break; }; grep -q "Error\|Traceback" "$OUT/dit_server.log" && { echo "dit server failed"; tail -5 "$OUT/dit_server.log"; exit 1; }; sleep 3; done
grep -q READY "$OUT/dit_server.log" || { echo "dit TIMEOUT"; exit 1; }
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tee "$OUT/mem_after_load.txt"
case $MODE in mixed) ONLY=both ;; llm) ONLY=llm ;; dit) ONLY=dit ;; *) echo bad mode; exit 1 ;; esac
nvidia-smi dmon -s um -d 1 -o T > "$OUT/dmon.txt" 2>/dev/null & DM=$!
echo "replay only=$ONLY start $(date +%T)"; date +%s.%N > "$OUT/replay_start_unix.txt"
setsid "$V/python" bench/hetero/controlled_mixed.py replay --plan "$PLAN" --only "$ONLY" --out "$OUT/replay" --drain-timeout 60 > "$OUT/replay.log" 2>&1 & PIDS+=($!); RP=$!
sleep $((WARMUP + PRE))
echo "nsys start $(date +%T)"; date +%s.%N > "$OUT/capture_start_unix.txt"
"$NSYS" start --session="$SESSION" --sample=none --cpuctxsw=none --output="$OUT/report" --force-overwrite=true > "$OUT/nsys_start.log" 2>&1 || { echo "nsys start failed"; cat "$OUT/nsys_start.log"; exit 1; }
sleep "$CAP"
"$NSYS" stop --session="$SESSION" > "$OUT/nsys_stop.log" 2>&1; echo "nsys stop $(date +%T)"; date +%s.%N > "$OUT/capture_end_unix.txt"
sleep 10; kill -- -"$RP" 2>/dev/null || true
ls -la "$OUT"/report.nsys-rep 2>/dev/null && date > "$OUT/done"
