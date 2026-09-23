#!/usr/bin/env bash
# 카드별 LLM 단계 프로파일 (계수 추출용). vLLM 내장 `vllm bench latency` 로 스케줄러를 고정한 채
# (prefix caching off, 단일 배치, eager 아님) 프리필·디코드 지연을 잰다. 결과는 JSON 한 파일씩.
#
#   prefill : batch 1, output_len 1, input_len 스윕            → T_prefill(S_in)
#   decode  : input_len 128 고정, output_len 256, batch 스윕    → 스텝 시간(B), 문맥 ~128+128
#   context : batch 8 고정, output_len 256, input_len 스윕      → 스텝 시간(KV 총량)
#
# 사용:  bench/hetero/profile_llm.sh 4090      (또는 3090)
# 환경:  ~/venv/jinwoo, CUDA_DEVICE_ORDER=PCI_BUS_ID, GPU 는 UUID 로 지정 (번호 쓰지 말 것).
set -euo pipefail

CARD=${1:?usage: profile_llm.sh 4090|3090}
MODEL=${MODEL:-meta-llama/Llama-3.1-8B-Instruct}
PY=~/venv/jinwoo/bin/vllm
OUT=results/hetero/profile_llm/$CARD
mkdir -p "$OUT"

case $CARD in
  4090) UUID=GPU-d5bd27b1-7998-9ae2-b764-3c528e12a47b ;;
  3090) UUID=GPU-208c8076-e9ae-fde6-aacf-b0c3fae16a9e ;;
  *) echo "unknown card $CARD"; exit 1 ;;
esac
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$UUID
# 이 서버에는 CUDA 툴킷(nvcc)이 없어 FlashInfer 샘플러 JIT 가 실패한다 → 샘플러만 torch 경로로.
export VLLM_USE_FLASHINFER_SAMPLER=0

COMMON=(--model "$MODEL" --dtype bfloat16 --max-model-len 8192
        --no-enable-prefix-caching --gpu-memory-utilization 0.90 --seed 0
        --num-iters-warmup 3 --num-iters 10)

run() {  # tag input_len output_len batch
  local tag=$1 in=$2 out=$3 b=$4
  local f="$OUT/$tag.json"
  [ -s "$f" ] && { echo "skip $tag"; return; }
  echo "== $CARD $tag  in=$in out=$out b=$b  $(date +%T)"
  # 전체 출력은 구성별 로그로 남기고(오류를 삼키지 않기 위해), 화면에는 요약만.
  if ! $PY bench latency "${COMMON[@]}" --input-len "$in" --output-len "$out" --batch-size "$b" \
        --max-num-seqs "$b" --output-json "$f" > "$OUT/$tag.log" 2>&1; then
    echo "FAILED $tag — see $OUT/$tag.log"; grep -E "Error|error" "$OUT/$tag.log" | tail -3; exit 1
  fi
  grep -E "Avg latency" "$OUT/$tag.log" | tail -1
}

# 1) prefill
for s in 128 256 512 1024 2048 4096; do run "prefill_in${s}" "$s" 1 1; done
# 2) decode vs batch
for b in 1 2 4 8 16 32 64; do run "decode_b${b}_in128" 128 256 "$b"; done
# 3) decode vs context
for s in 128 512 1024 2048 4096; do run "decode_b8_in${s}" "$s" 256 8; done

echo "done -> $OUT"
