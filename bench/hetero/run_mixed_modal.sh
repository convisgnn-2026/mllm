#!/usr/bin/env bash
# 혼재(LLM + DiT) 정적 배치 실험. LLM 구성(S4|S3|C|A|none) × DiT 위치(4090|3090|split|none) × (LLM 도착률, DiT 도착률).
#   LRATES="1.0 2.0" DRATES="0.1 0.2" N=300 bench/hetero/run_mixed_modal.sh C 4090
#   CONTROLLED_PLAN=<plan.json> [CONTROLLED_OUT=<dir>] [REPLAY_ONLY=llm|dit] bench/hetero/run_mixed_modal.sh S4 3090
#     REPLAY_ONLY 는 상주형 단독: 두 서버를 모두 띄운 채 한 서비스 요청만 재생한다(LLM=none/DIT=none 과 다름).
# 결과 results/hetero/mixed_modal/<LLM>_<DIT>/<work>_L<lr>_D<dr>/{llm,dit}/ + gpu.csv
# cuda:0 = 4090, cuda:1 = 3090 (UUID 고정). LLM 메모리는 GPU_UTIL(기본 0.5) 로 고정해 단독/혼재 조건을 같게 둔다.
set -uo pipefail
cd "$(dirname "$0")/../.."
LLM=${1:?S4|S3|C|A|none}; DIT=${2:?4090|3090|split|none}
MODEL=${MODEL:-meta-llama/Llama-3.2-3B-Instruct}; DMODEL=${DMODEL:-stabilityai/stable-diffusion-3.5-medium}; V=~/venv/jinwoo/bin
GPU_UTIL=${GPU_UTIL:-0.5}; SIZE=${SIZE:-512}; STEPS=${STEPS:-28}
N=${N:-300}; LRATES=${LRATES:-"0"}; DRATES=${DRATES:-"0"}; WORK=${WORK:-code}
TRACE=${TRACE:-bench/hetero/traces/AzureLLMInferenceTrace_${WORK}.csv}
OUT=results/hetero/mixed_modal/${LLM}_${DIT}
if [ -n "${CONTROLLED_PLAN:-}" ]; then
  CONTROLLED_PLAN=$(realpath "$CONTROLLED_PLAN")
  [ -f "$CONTROLLED_PLAN" ] || { echo "missing plan"; exit 1; }
  OUT=${CONTROLLED_OUT:-results/hetero/controlled_runs/${LLM}_${DIT}/$(basename "$CONTROLLED_PLAN" .json)}
  [ ! -e "$OUT" ] || { echo "output exists: $OUT"; exit 1; }
  "$V/python" - "$CONTROLLED_PLAN" "$MODEL" <<'PYCHECK'
import json, sys
assert json.load(open(sys.argv[1]))['model'] == sys.argv[2], 'plan/server model mismatch'
PYCHECK
  [ $? -eq 0 ] || exit 1
  nvidia-smi >/dev/null || { echo "GPU unavailable"; exit 1; }
fi
mkdir -p "$OUT"
G4090=GPU-d5bd27b1-7998-9ae2-b764-3c528e12a47b; G3090=GPU-208c8076-e9ae-fde6-aacf-b0c3fae16a9e
export CUDA_DEVICE_ORDER=PCI_BUS_ID VLLM_USE_FLASHINFER_SAMPLER=0 UCX_NET_DEVICES=all HF_HUB_OFFLINE=1
export UCX_TLS=cuda_copy,tcp,self UCX_TCP_TX_SEG_SIZE=4m UCX_TCP_RX_SEG_SIZE=4m UCX_TCP_SNDBUF=64m UCX_TCP_RCVBUF=64m UCX_MAX_RNDV_LANES=4 UCX_RNDV_THRESH=64k
# KV 예산은 바이트로 고정한다. gpu-memory-utilization 만으로는 profiling 시점의 컴파일·activation 피크에 따라 KV 용량이 달라진다
# (Stage 0: 같은 후보가 37,776 vs 50,272 토큰). KV_BYTES=0 이면 예전처럼 util 기반 자동 산정.
KV_BYTES=${KV_BYTES:-4294967296}
KVARG=(); [ "$KV_BYTES" = 0 ] || KVARG=(--kv-cache-memory-bytes "$KV_BYTES")
COMMON=(--dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization "$GPU_UTIL" "${KVARG[@]}" --no-enable-prefix-caching --max-num-seqs 256)
PIDS=(); SM=""
cleanup() {
  echo "cleanup"
  for pid in "${PIDS[@]}"; do kill -- -"$pid" 2>/dev/null || true; done
  [ -z "${SM:-}" ] || kill "$SM" 2>/dev/null || true
  sleep 4
  for pid in "${PIDS[@]}"; do kill -9 -- -"$pid" 2>/dev/null || true; done
}
trap cleanup EXIT
serve() { local gpu=$1 port=$2 role=$3 sp=$4 kv=()
  [ "$role" = producer ] && kv=(--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}')
  [ "$role" = consumer ] && kv=(--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"}')
  CUDA_VISIBLE_DEVICES=$gpu VLLM_NIXL_SIDE_CHANNEL_PORT=$sp setsid "$V/vllm" serve "$MODEL" --port "$port" "${COMMON[@]}" "${kv[@]}" > "$OUT/serve_$port.log" 2>&1 & PIDS+=($!); }
wait_ready() { for i in $(seq 1 150); do curl -sf "127.0.0.1:$1/health" >/dev/null && { echo "  :$1 ready ($((i*3))s)"; return 0; }; sleep 3; done; echo ":$1 TIMEOUT"; exit 1; }
echo "== mixed_modal LLM=$LLM DIT=$DIT  $(date +%T)"
# --- DiT 서버 (먼저 띄워 메모리를 선점; LLM 은 남은 메모리의 GPU_UTIL 이 아니라 전체 대비 비율이므로 순서 무관)
case $DIT in
  4090)  DEV=(--enc-dev cuda:0 --dit-dev cuda:0 --vae-dev cuda:0) ;;
  3090)  DEV=(--enc-dev cuda:1 --dit-dev cuda:1 --vae-dev cuda:1) ;;
  split) DEV=(--enc-dev cuda:1 --dit-dev cuda:0 --vae-dev cuda:1) ;;   # Denoise 4090, 인코더·VAE 3090
  none)  DEV=() ;;
esac
if [ "$DIT" != none ]; then
  CUDA_VISIBLE_DEVICES=$G4090,$G3090 setsid "$V/python" bench/hetero/dit_server.py --model "$DMODEL" --port 8300 "${DEV[@]}" --size "$SIZE" --steps "$STEPS" > "$OUT/dit_server.log" 2>&1 & PIDS+=($!)
fi
# --- LLM
case $LLM in
  A)  serve $G4090 8100 full 5600; serve $G3090 8200 full 5601; wait_ready 8100; wait_ready 8200
      setsid "$V/python" bench/hetero/rr_proxy_server.py --port 8192 --backends 127.0.0.1:8100 127.0.0.1:8200 > "$OUT/proxy.log" 2>&1 & PIDS+=($!) ;;
  C)  serve $G4090 8100 producer 5600; serve $G3090 8200 consumer 5601; wait_ready 8100; wait_ready 8200
      setsid "$V/python" bench/hetero/toy_proxy_server.py --port 8192 --prefiller-hosts 127.0.0.1 --prefiller-ports 8100 --decoder-hosts 127.0.0.1 --decoder-ports 8200 > "$OUT/proxy.log" 2>&1 & PIDS+=($!) ;;
  S4) serve $G4090 8100 full 5600; wait_ready 8100
      setsid "$V/python" bench/hetero/rr_proxy_server.py --port 8192 --backends 127.0.0.1:8100 > "$OUT/proxy.log" 2>&1 & PIDS+=($!) ;;
  S3) serve $G3090 8100 full 5600; wait_ready 8100
      setsid "$V/python" bench/hetero/rr_proxy_server.py --port 8192 --backends 127.0.0.1:8100 > "$OUT/proxy.log" 2>&1 & PIDS+=($!) ;;
  none) ;;
esac
if [ "$LLM" != none ]; then
  sleep 4; { curl -sf 127.0.0.1:8192/health >/dev/null || curl -sf 127.0.0.1:8192/healthcheck >/dev/null; } || { echo "proxy not up"; exit 1; }
  $V/python bench/hetero/loadgen.py --model "$MODEL" --n 8 --mix "512:32=1.0" --tag warm --out "$OUT/_warmup" > /dev/null 2>&1
fi
if [ "$DIT" != none ]; then
  for i in $(seq 1 200); do grep -q READY "$OUT/dit_server.log" && { echo "  dit ready ($((i*3))s)"; break; }; grep -q "Error\|Traceback" "$OUT/dit_server.log" && { echo "dit server failed"; tail -5 "$OUT/dit_server.log"; exit 1; }; sleep 3; done
  grep -q READY "$OUT/dit_server.log" || { echo "dit TIMEOUT"; exit 1; }
fi
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader > "$OUT/mem_after_load.txt"; cat "$OUT/mem_after_load.txt"
if [ -n "${CONTROLLED_PLAN:-}" ]; then
  "$V/python" - "$OUT" "$LLM" "$DIT" "$MODEL" "$DMODEL" "$GPU_UTIL" "$SIZE" "$STEPS" "$G4090" "$G3090" "$KV_BYTES" <<'PYMETA'
import json, sys, platform, re
from pathlib import Path
out, llm, dit, model, dmodel, util, size, steps, gpu4, gpu3, kv_bytes = sys.argv[1:]
kv_tokens={}
for log in sorted(Path(out).glob('serve_*.log')):
    m=re.search(r'GPU KV cache size: ([\d,]+) tokens', log.read_text(errors='replace'))
    kv_tokens[log.stem]=int(m.group(1).replace(',','')) if m else None
import importlib.metadata as md
versions={k: md.version(k) for k in ('torch','vllm','diffusers','transformers','httpx')}
Path(out,'deployment.json').write_text(json.dumps(dict(llm=llm,dit=dit,model=model,dit_model=dmodel,gpu_util=float(util),kv_cache_memory_bytes=int(kv_bytes),kv_cache_tokens=kv_tokens,size=int(size),steps=int(steps),gpu_uuids=[gpu4,gpu3],python=platform.python_version(),versions=versions),indent=2))
PYMETA
  [ $? -eq 0 ] || exit 1
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw,clocks.sm,clocks.mem --format=csv,noheader -lms 1000 > "$OUT/gpu.csv" 2>/dev/null & SM=$!
  ONLY=both
  [ "$LLM" != none ] || ONLY=dit
  [ "$DIT" != none ] || ONLY=llm
  # 상주형 단독(resident solo): 두 서버를 모두 올린 채 한 서비스에만 요청을 보낸다. REPLAY_ONLY=llm|dit 로 지정.
  if [ -n "${REPLAY_ONLY:-}" ]; then
    case $REPLAY_ONLY in both|llm|dit) ;; *) echo "bad REPLAY_ONLY=$REPLAY_ONLY"; exit 1 ;; esac
    { [ "$REPLAY_ONLY" = llm ] && [ "$LLM" = none ]; } && { echo "REPLAY_ONLY=llm but LLM=none"; exit 1; }
    { [ "$REPLAY_ONLY" = dit ] && [ "$DIT" = none ]; } && { echo "REPLAY_ONLY=dit but DIT=none"; exit 1; }
    ONLY=$REPLAY_ONLY
  fi
  echo "replay only=$ONLY"
  "$V/python" bench/hetero/controlled_mixed.py replay --plan "$CONTROLLED_PLAN" --only "$ONLY" --out "$OUT/replay" --drain-timeout "${DRAIN_TIMEOUT:-300}"
  status=$?
  kill "$SM" 2>/dev/null; wait "$SM" 2>/dev/null; SM=""
  [ "$status" -ne 0 ] || date > "$OUT/done"
  exit "$status"
fi
for lr in $LRATES; do for dr in $DRATES; do
  [ "$LLM" = none ] && lr=0; [ "$DIT" = none ] && dr=0
  cell="$OUT/${WORK}_L${lr}_D${dr}"; [ -s "$cell/done" ] && { echo "skip $cell"; continue; }; mkdir -p "$cell/llm" "$cell/dit"
  # 두 부하의 길이를 맞춘다: LLM N 건이 lr 로 들어오는 동안(N/lr 초) DiT 는 dr 로 들어온다. LLM 없으면 DiT ND 건.
  if [ "$LLM" != none ] && [ "$lr" != 0 ]; then dur=$(awk "BEGIN{print $N/$lr}"); nd=$(awk "BEGIN{v=int($dr*$dur+0.5); print (v<20?20:v)}"); else dur=0; nd=${ND:-60}; fi
  echo "-- cell $cell  (LLM n=$N @${lr}, DiT n=$nd @${dr})  $(date +%T)"
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw,clocks.sm,clocks.mem --format=csv,noheader -lms 1000 > "$cell/gpu.csv" 2>/dev/null & SM=$!
  LP=""; DP=""
  if [ "$LLM" != none ] && [ "$lr" != 0 ]; then
    $V/python bench/hetero/loadgen.py --model "$MODEL" --n "$N" --trace "$TRACE" --trace-arrivals poisson --rate "$lr" --seed 0 --tag "${LLM}_${DIT}" --out "$cell/llm" 2>&1 | grep -v -E "^(INFO|WARNING)" & LP=$!
  fi
  if [ "$DIT" != none ] && [ "$dr" != 0 ]; then
    $V/python bench/hetero/dit_loadgen.py --url http://127.0.0.1:8300 --n "$nd" --rate "$dr" --seed 0 --tag "${LLM}_${DIT}" --out "$cell/dit" & DP=$!
  fi
  [ -n "$LP" ] && wait $LP; [ -n "$DP" ] && wait $DP
  kill $SM 2>/dev/null; wait $SM 2>/dev/null; SM=""; date > "$cell/done"; sleep 5
done; done
echo "== done LLM=$LLM DIT=$DIT  $(date +%T)"
