#!/usr/bin/env bash
# Splitwise 식 무경합 기준: 3090 한 장, 프록시 없이, 요청을 하나씩(rate 매우 낮음) 보내 크기별 TTFT/TPOT 를 잰다.
# 결과 results/hetero/uncontended/<card>/requests.csv → fit 은 uncontended_fit.py
set -uo pipefail; cd "$(dirname "$0")/../.."
CARD=${1:-3090}; MODEL=${MODEL:-meta-llama/Llama-3.1-8B-Instruct}; V=~/venv/jinwoo/bin
G4090=GPU-d5bd27b1-7998-9ae2-b764-3c528e12a47b; G3090=GPU-208c8076-e9ae-fde6-aacf-b0c3fae16a9e
[ "$CARD" = 4090 ] && GPU=$G4090 || GPU=$G3090
export CUDA_DEVICE_ORDER=PCI_BUS_ID VLLM_USE_FLASHINFER_SAMPLER=0
OUT=results/hetero/uncontended/$CARD; mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=$GPU $V/vllm serve "$MODEL" --port 8100 --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.85 --no-enable-prefix-caching --max-num-seqs 256 > "$OUT/serve.log" 2>&1 & P=$!
trap 'kill $P 2>/dev/null; sleep 4; kill -9 $P 2>/dev/null; pkill -9 -f VLLM::EngineCore' EXIT
for i in $(seq 1 150); do curl -sf 127.0.0.1:8100/health >/dev/null && break; sleep 3; done
$V/python bench/hetero/loadgen.py --url http://127.0.0.1:8100 --model "$MODEL" --n 6 --mix "512:32=1.0" --tag warm --out "$OUT/_warmup" >/dev/null 2>&1
# 크기 격자: 입력 {64,128,256,512,1024,2048,4096,8000} × 출력 {32,256}; 순차 (max-concurrency 1) 로 무경합 보장
MIX=$(for s in 64 128 256 512 1024 2048 4096 8000; do for o in 32 256; do printf "%s:%s=0.0625," $s $o; done; done); MIX=${MIX%,}
$V/python bench/hetero/loadgen.py --url http://127.0.0.1:8100 --model "$MODEL" --n 48 --mix "$MIX" --rate inf --max-concurrency 1 --seed 0 --tag "unc_$CARD" --out "$OUT" 2>&1 | grep -v -E "^(INFO|WARNING)"
