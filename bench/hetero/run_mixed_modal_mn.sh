#!/usr/bin/env bash
# 멀티노드 판 run_mixed_modal.sh — B300 노드(로컬) + RTX PRO 6000 노드(원격) 이기종 풀.
# 역할마다 "호스트:GPU-UUID" 를 받는다. 호스트가 local 이면 이 노드, 아니면 ssh 로 띄운다(양쪽 ~/mllm, ~/venv/mllm 동일 경로 가정).
#
#   LLM=S|C   DIT=yes|none
#   S : 단일 엔진       LLM_GPU=host:uuid
#   C : PD 분할         P_GPU=host:uuid  D_GPU=host:uuid
#   DiT 서버            DIT_GPU=host:uuid   (enc/dit/vae 모두 같은 카드. split 은 1차 범위 밖)
#
#   예) 공유 한 장(B300):      LLM=S LLM_GPU=local:GPU-1b5a… DIT_GPU=local:GPU-1b5a…
#       서비스 분리(노드 간):   LLM=S LLM_GPU=local:GPU-1b5a… DIT_GPU=10.201.4.101:GPU-bdb8…
#       PD 노드 간 + DiT PRO:   LLM=C P_GPU=local:GPU-1b5a… D_GPU=10.201.4.101:GPU-bdb8… DIT_GPU=10.201.4.101:GPU-d8d3…
#
#   CONTROLLED_PLAN=<plan.json> [CONTROLLED_OUT=<dir>] [REPLAY_ONLY=llm|dit|both] bench/hetero/run_mixed_modal_mn.sh
# 판정 정의(stage1_report.py)·replay(controlled_mixed.py)·loadgen 은 원본 그대로. 서버 기동·주소·수집만 다르다.
set -uo pipefail
cd "$(dirname "$0")/../.."
LLM=${LLM:?S|C}; DIT=${DIT:-yes}
MODEL=${MODEL:-meta-llama/Llama-3.2-3B-Instruct}; DMODEL=${DMODEL:-stabilityai/stable-diffusion-3.5-medium}
V=$HOME/venv/mllm/bin; REPO=$HOME/mllm
GPU_UTIL=${GPU_UTIL:-0.5}; SIZE=${SIZE:-512}; STEPS=${STEPS:-28}
KV_BYTES=${KV_BYTES:-4294967296}
LOCAL_IP=${LOCAL_IP:-$(hostname -I | awk '{print $2}')}        # 10.201.4.x (bond0)
NET_DEV=${NET_DEV:-bond0}
CONTROLLED_PLAN=${CONTROLLED_PLAN:?plan.json 필요}
CONTROLLED_PLAN=$(realpath "$CONTROLLED_PLAN"); [ -f "$CONTROLLED_PLAN" ] || { echo "missing plan"; exit 1; }
tag="${LLM}_$(echo "${LLM_GPU:-${P_GPU:-}}" | cut -d: -f1)_dit_$(echo "${DIT_GPU:-none}" | cut -d: -f1)"
OUT=${CONTROLLED_OUT:-results/hetero/controlled_runs_mn/${tag}/$(basename "$CONTROLLED_PLAN" .json)}
[ ! -e "$OUT" ] || { echo "output exists: $OUT"; exit 1; }; mkdir -p "$OUT"
"$V/python" - "$CONTROLLED_PLAN" "$MODEL" <<'PYCHECK'
import json, sys
assert json.load(open(sys.argv[1]))['model'] == sys.argv[2], 'plan/server model mismatch'
PYCHECK
[ $? -eq 0 ] || exit 1

host_of() { echo "${1%%:*}"; }; uuid_of() { echo "${1#*:}"; }
ip_of() { [ "$1" = local ] && echo "$LOCAL_IP" || echo "$1"; }
UCX_ENV="UCX_NET_DEVICES=$NET_DEV UCX_TLS=cuda_copy,tcp,self UCX_TCP_TX_SEG_SIZE=4m UCX_TCP_RX_SEG_SIZE=4m UCX_TCP_SNDBUF=64m UCX_TCP_RCVBUF=64m UCX_MAX_RNDV_LANES=4 UCX_RNDV_THRESH=64k"
BASE_ENV="CUDA_DEVICE_ORDER=PCI_BUS_ID VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1 $UCX_ENV"
COMMON="--dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization $GPU_UTIL --kv-cache-memory-bytes $KV_BYTES --no-enable-prefix-caching --max-num-seqs 256 --host 0.0.0.0"
KVP='{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}'
KVC='{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"}'

# ---- 프로세스 관리: 로컬은 setsid PID, 원격은 ssh + 원격 PID 파일
LPIDS=(); RHOSTS=(); RPIDS=(); SMS=()
launch() { # launch <host> <logname> <env...> -- <cmd...>
  local host=$1 log=$2; shift 2; local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  if [ "$host" = local ]; then
    env "${envs[@]}" setsid "$@" > "$OUT/$log" 2>&1 & LPIDS+=($!)
  else
    local q; printf -v q '%q ' "$@"
    local pid; pid=$(ssh -o BatchMode=yes "$host" "mkdir -p $REPO/$OUT && cd $REPO && nohup env ${envs[*]} setsid $q > $REPO/$OUT/$log 2>&1 < /dev/null & echo \$!")
    RHOSTS+=("$host"); RPIDS+=("$pid")
  fi
}
cleanup() { echo "cleanup $(date +%T)"
  for s in "${SMS[@]}"; do kill "$s" 2>/dev/null || true; done
  for pid in "${LPIDS[@]}"; do kill -- -"$pid" 2>/dev/null || true; done
  for i in "${!RPIDS[@]}"; do ssh -o BatchMode=yes "${RHOSTS[$i]}" "kill -- -${RPIDS[$i]} 2>/dev/null; sleep 4; kill -9 -- -${RPIDS[$i]} 2>/dev/null" || true; done
  sleep 4; for pid in "${LPIDS[@]}"; do kill -9 -- -"$pid" 2>/dev/null || true; done
  # 원격 로그 회수
  for h in $(printf '%s\n' "${RHOSTS[@]}" | sort -u); do [ -n "$h" ] && scp -q -o BatchMode=yes "$h:$REPO/$OUT/*.log" "$OUT/" 2>/dev/null; done
}
trap cleanup EXIT
wait_http() { local url=$1; for i in $(seq 1 200); do curl -sf "$url" >/dev/null && { echo "  $url ready ($((i*3))s)"; return 0; }; sleep 3; done; echo "$url TIMEOUT"; exit 1; }

serve_llm() { # serve_llm <host:uuid> <port> <role full|producer|consumer> <side_port>
  local host; host=$(host_of "$1"); local uuid; uuid=$(uuid_of "$1"); local port=$2 role=$3 sp=$4 kv=()
  [ "$role" = producer ] && kv=(--kv-transfer-config "$KVP"); [ "$role" = consumer ] && kv=(--kv-transfer-config "$KVC")
  launch "$host" "serve_$port.log" $BASE_ENV "CUDA_VISIBLE_DEVICES=$uuid" "VLLM_NIXL_SIDE_CHANNEL_PORT=$sp" "VLLM_NIXL_SIDE_CHANNEL_HOST=$(ip_of "$host")" \
    -- "$V/vllm" serve "$MODEL" --port "$port" $COMMON "${kv[@]}"
}
echo "== mixed_modal_mn LLM=$LLM DIT=$DIT  $(date +%T)  out=$OUT"
# ---- DiT 서버
if [ "$DIT" != none ]; then
  dh=$(host_of "$DIT_GPU"); du=$(uuid_of "$DIT_GPU")
  launch "$dh" dit_server.log $BASE_ENV "CUDA_VISIBLE_DEVICES=$du" -- "$V/python" bench/hetero/dit_server.py --model "$DMODEL" --port 8300 --host 0.0.0.0 \
    --enc-dev cuda:0 --dit-dev cuda:0 --vae-dev cuda:0 --size "$SIZE" --steps "$STEPS"
  DIT_URL="http://$(ip_of "$dh"):8300"
fi
# ---- LLM
case $LLM in
  S) serve_llm "$LLM_GPU" 8100 full 5600; LH=$(ip_of "$(host_of "$LLM_GPU")"); wait_http "http://$LH:8100/health"
     setsid "$V/python" bench/hetero/rr_proxy_server.py --port 8192 --backends "$LH:8100" > "$OUT/proxy.log" 2>&1 & LPIDS+=($!) ;;
  C) serve_llm "$P_GPU" 8100 producer 5600; serve_llm "$D_GPU" 8200 consumer 5601
     PH=$(ip_of "$(host_of "$P_GPU")"); DH=$(ip_of "$(host_of "$D_GPU")"); wait_http "http://$PH:8100/health"; wait_http "http://$DH:8200/health"
     setsid "$V/python" bench/hetero/toy_proxy_server.py --port 8192 --prefiller-hosts "$PH" --prefiller-ports 8100 --decoder-hosts "$DH" --decoder-ports 8200 > "$OUT/proxy.log" 2>&1 & LPIDS+=($!) ;;
  *) echo "bad LLM=$LLM"; exit 1 ;;
esac
sleep 4; { curl -sf 127.0.0.1:8192/health >/dev/null || curl -sf 127.0.0.1:8192/healthcheck >/dev/null; } || { echo "proxy not up"; exit 1; }
"$V/python" bench/hetero/loadgen.py --model "$MODEL" --n 8 --mix "512:32=1.0" --tag warm --out "$OUT/_warmup" > /dev/null 2>&1
if [ "$DIT" != none ]; then wait_http "$DIT_URL/health"; fi
# ---- 메타데이터 + GPU 샘플링(양쪽 노드)
for h in local $(printf '%s\n' "${RHOSTS[@]}" | sort -u); do
  if [ "$h" = local ]; then nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader > "$OUT/mem_after_load.local.txt"
  else ssh -o BatchMode=yes "$h" nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader > "$OUT/mem_after_load.$h.txt"; fi
done
"$V/python" - "$OUT" "$LLM" "$DIT" "$MODEL" "$DMODEL" "$GPU_UTIL" "$SIZE" "$STEPS" "$KV_BYTES" "${LLM_GPU:-}" "${P_GPU:-}" "${D_GPU:-}" "${DIT_GPU:-}" <<'PYMETA'
import json, sys, platform, re, importlib.metadata as md
from pathlib import Path
out, llm, dit, model, dmodel, util, size, steps, kv_bytes, llm_gpu, p_gpu, d_gpu, dit_gpu = sys.argv[1:]
versions={k: md.version(k) for k in ('torch','vllm','diffusers','transformers','httpx','nixl')}
Path(out,'deployment.json').write_text(json.dumps(dict(harness='run_mixed_modal_mn.sh',llm=llm,dit=dit,model=model,dit_model=dmodel,gpu_util=float(util),
  kv_cache_memory_bytes=int(kv_bytes),size=int(size),steps=int(steps),placement=dict(llm=llm_gpu,prefill=p_gpu,decode=d_gpu,dit=dit_gpu),
  python=platform.python_version(),versions=versions),indent=2))
PYMETA
Q="--query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw,clocks.sm,clocks.mem --format=csv,noheader -lms 1000"
nvidia-smi $Q > "$OUT/gpu.local.csv" 2>/dev/null & SMS+=($!)
for h in $(printf '%s\n' "${RHOSTS[@]}" | sort -u); do [ -n "$h" ] && { ssh -o BatchMode=yes "$h" nvidia-smi $Q > "$OUT/gpu.$h.csv" 2>/dev/null & SMS+=($!); }; done
ONLY=${REPLAY_ONLY:-both}; [ "$DIT" != none ] || ONLY=llm
echo "replay only=$ONLY"
"$V/python" bench/hetero/controlled_mixed.py replay --plan "$CONTROLLED_PLAN" --only "$ONLY" --out "$OUT/replay" \
  --llm-url http://127.0.0.1:8192 --dit-url "${DIT_URL:-http://127.0.0.1:8300}" --drain-timeout "${DRAIN_TIMEOUT:-300}"
status=$?
[ "$status" -ne 0 ] || date > "$OUT/done"
exit "$status"
