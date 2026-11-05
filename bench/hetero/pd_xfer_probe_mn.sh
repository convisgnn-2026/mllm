#!/usr/bin/env bash
# Cross-node PD KV transfer probe (2026-11-05): prefill vLLM on B300 (this host) -> decode vLLM on the RTX PRO 6000 node over 25 GbE.
# Same measurement as pd_xfer_probe.sh / pd_xfer_probe_b300.sh: 2048-token prompts (3B: KV 235 MB), 8 sequential then 16 concurrent,
# read the decoder's "KV Transfer metrics". The decode server is started over ssh and its log is copied back.
#   UCX_TLS=... UCX_NET_DEVICES=bond0 bench/hetero/pd_xfer_probe_mn.sh tag
set -uo pipefail
cd "$(dirname "$0")/../.."
TAG=${1:-default}; MODEL=${MODEL:-meta-llama/Llama-3.2-3B-Instruct}
V=~/venv/mllm/bin; export PATH=$HOME/venv/mllm/bin:$PATH HF_HUB_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID VLLM_USE_FLASHINFER_SAMPLER=0
REMOTE=${REMOTE:-10.201.4.101}; LOCAL_IP=${LOCAL_IP:-10.201.4.151}
PP=${PP:-8110}; PX=${PX:-8193}; SC=${SC:-5610}
# Both engines must pick the same attention backend: it is part of the NIXL compatibility hash (B300 auto-picks FLASHINFER,
# PRO 6000 auto-picks FLASH_ATTN -> "NIXL compatibility hash mismatch", 2026-11-05). FLASHINFER matches the B300 captures.
ATTN=${ATTN:-FLASHINFER}   # vLLM 0.28: CLI --attention-backend (the VLLM_ATTENTION_BACKEND env var is gone)   # prefill port, proxy port, prefill side-channel port (distinct from the capture harness)
P_GPU=${P_GPU:-GPU-55867e89-32c1-52ae-b925-34a116ba798d}            # B300 GPU3
D_GPU=${D_GPU:-GPU-d8d3cf1e-eafd-b8f8-e95e-916eefc2730b}            # PRO6000 GPU1 (GPU0 is MIG-enabled again as of 2026-11-05)
LOG=results/hetero/pd_xfer_mn/$TAG; mkdir -p "$LOG"
KVP='{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}'
KVD='{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"}'
COMMON="--dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.5 --no-enable-prefix-caching --max-num-seqs 64 --host 0.0.0.0 --attention-backend $ATTN"
UCXENV=$(env | grep -E "^UCX_" | tr '\n' ' '); echo "  env: $UCXENV"
PIDS=(); RPID=""
cleanup() { kill "${PIDS[@]}" 2>/dev/null; [ -z "$RPID" ] || ssh -o BatchMode=yes "$REMOTE" "kill $RPID 2>/dev/null; sleep 4; kill -9 $RPID 2>/dev/null; pkill -9 -u \$USER -f 'vllm serve $MODEL --port 8200' 2>/dev/null; true"
  sleep 4; kill -9 "${PIDS[@]}" 2>/dev/null; scp -q -o BatchMode=yes "$REMOTE:~/mllm/$LOG/decode.log" "$LOG/decode.log" 2>/dev/null; true; }
trap cleanup EXIT
# decode on the remote node (log stays remote until cleanup copies it back)
# ssh -n -f: ssh itself goes to the background once the remote command is started, so a long-lived remote server cannot hold
# the session open (plain `ssh host "cmd &"` hung here even with setsid and stdin from /dev/null, 2026-11-05).
ssh -n -f -o BatchMode=yes "$REMOTE" "mkdir -p ~/mllm/$LOG && cd ~/mllm && exec env PATH=\$HOME/venv/mllm/bin:\$PATH HF_HUB_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID VLLM_USE_FLASHINFER_SAMPLER=0 $UCXENV CUDA_VISIBLE_DEVICES=$D_GPU VLLM_NIXL_SIDE_CHANNEL_HOST=$REMOTE VLLM_NIXL_SIDE_CHANNEL_PORT=5601 \$HOME/venv/mllm/bin/vllm serve $MODEL --port 8200 $COMMON --kv-transfer-config '$KVD' > ~/mllm/$LOG/decode.log 2>&1"
sleep 5; RPID=$(ssh -o BatchMode=yes "$REMOTE" "pgrep -u \$USER -f 'vllm serve $MODEL --port 8200' | head -1"); echo "  remote decode pid ${RPID:-?}"
CUDA_VISIBLE_DEVICES=$P_GPU VLLM_NIXL_SIDE_CHANNEL_HOST=$LOCAL_IP VLLM_NIXL_SIDE_CHANNEL_PORT=$SC $V/vllm serve "$MODEL" --port $PP $COMMON --kv-transfer-config "$KVP" > "$LOG/prefill.log" 2>&1 & PIDS+=($!)
for i in $(seq 1 100); do curl -sf 127.0.0.1:$PP/health >/dev/null && { echo "  prefill ready $((i*3))s"; break; }; sleep 3; done
for i in $(seq 1 100); do curl -sf "$REMOTE:8200/health" >/dev/null && { echo "  decode ready $((i*3))s"; break; }; sleep 3; done
curl -sf "$REMOTE:8200/health" >/dev/null || { echo "DECODE TIMEOUT"; ssh -o BatchMode=yes "$REMOTE" "tail -15 ~/mllm/$LOG/decode.log"; exit 1; }
$V/python bench/hetero/toy_proxy_server.py --port $PX --prefiller-hosts 127.0.0.1 --prefiller-ports $PP --decoder-hosts "$REMOTE" --decoder-ports 8200 > "$LOG/proxy.log" 2>&1 & PIDS+=($!); sleep 3
$V/python bench/hetero/loadgen.py --url http://127.0.0.1:$PX --model "$MODEL" --n 8 --mix "2048:16=1.0" --max-concurrency 1 --tag seq --out "$LOG/seq" 2>&1 | grep "^\["
sleep 12
echo "seq   : $(ssh -o BatchMode=yes "$REMOTE" "grep -o 'KV Transfer metrics:.*P90 xfer time (ms)=[0-9.]*' ~/mllm/$LOG/decode.log | tail -1 | cut -c1-120")"
$V/python bench/hetero/loadgen.py --url http://127.0.0.1:$PX --model "$MODEL" --n 16 --mix "2048:16=1.0" --tag conc --out "$LOG/conc" 2>&1 | grep "^\["
sleep 12
echo "conc  : $(ssh -o BatchMode=yes "$REMOTE" "grep -o 'KV Transfer metrics:.*P90 xfer time (ms)=[0-9.]*' ~/mllm/$LOG/decode.log | tail -1 | cut -c1-120")"
echo "--- UCX transport lines (decode, remote)"; ssh -o BatchMode=yes "$REMOTE" "grep -hoE '(self|intra-node|inter-node) cfg#[0-9]+ .*' ~/mllm/$LOG/decode.log | sort | uniq -c | head -3 | cut -c1-200"
echo "PROBE DONE $TAG"
