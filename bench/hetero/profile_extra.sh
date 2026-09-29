#!/usr/bin/env bash
# 프로파일 격자의 공선성을 깨는 추가 점: 큰 배치×긴 문맥, 작은 배치×긴 문맥. profile_llm.sh 와 같은 설정.
set -uo pipefail; cd "$(dirname "$0")/../.."
CARD=${1:?4090|3090}; MODEL=${MODEL:-meta-llama/Llama-3.1-8B-Instruct}; PY=~/venv/jinwoo/bin/vllm
OUT=results/hetero/profile_llm/$CARD
case $CARD in 4090) UUID=GPU-d5bd27b1-7998-9ae2-b764-3c528e12a47b;; 3090) UUID=GPU-208c8076-e9ae-fde6-aacf-b0c3fae16a9e;; esac
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$UUID VLLM_USE_FLASHINFER_SAMPLER=0
COMMON=(--model "$MODEL" --dtype bfloat16 --max-model-len 8192 --no-enable-prefix-caching --gpu-memory-utilization 0.90 --seed 0 --num-iters-warmup 3 --num-iters 10)
run() { local tag=$1 in=$2 out=$3 b=$4; local f="$OUT/$tag.json"; [ -s "$f" ] && { echo "skip $tag"; return; }
  echo "== $CARD $tag in=$in out=$out b=$b $(date +%T)"
  $PY bench latency "${COMMON[@]}" --input-len "$in" --output-len "$out" --batch-size "$b" --max-num-seqs "$b" --output-json "$f" > "$OUT/$tag.log" 2>&1 \
    || { echo "FAILED $tag"; grep -E "Error|error" "$OUT/$tag.log" | tail -2; return; }
  grep -E "Avg latency" "$OUT/$tag.log" | tail -1; }
run decode_b1_in4096 4096 256 1
run decode_b1_in2048 2048 256 1
run decode_b32_in1024 1024 256 32
run decode_b64_in4096 4096 256 64      # 64×4352 토큰 KV ≈ 36 GB > 24 GB → 실패하면 max-num-seqs 로 잘림; 그 경우 b16 로 대체
run decode_b16_in4096 4096 256 16
echo "done -> $OUT"
