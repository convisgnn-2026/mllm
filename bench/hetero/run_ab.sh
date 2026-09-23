#!/usr/bin/env bash
# 닫힌 실험: 같은 n 개 요청을 구성별로 처리해 총 완료 시간·처리량·요청별 지연·GPU 메모리를 기록한다.
#
#   bench/hetero/run_ab.sh A     복제 + 라운드로빈      (4090 full, 3090 full, rr_proxy 1:1)
#   bench/hetero/run_ab.sh B     복제 + 가중 배정        (rr_proxy 2:1  — 프리필 2.2×·디코드 1.4× 의 중간값)
#   bench/hetero/run_ab.sh C     PD 분할                 (4090 prefill → 3090 decode, toy_proxy)
#   bench/hetero/run_ab.sh D     PD 반대                 (3090 prefill → 4090 decode)
#
# 셀 = MIXES × NS. 결과 results/hetero/ab/<CONFIG>/<mix>_n<n>/{requests.csv,summary.json,gpu.csv}
set -uo pipefail
cd "$(dirname "$0")/../.."
CONFIG=${1:?A|B|C|D}
MODEL=${MODEL:-meta-llama/Llama-3.1-8B-Instruct}
V=~/venv/jinwoo/bin
OUT=results/hetero/ab/$CONFIG${OUT_SUFFIX:-}; mkdir -p "$OUT"
G4090=GPU-d5bd27b1-7998-9ae2-b764-3c528e12a47b
G3090=GPU-208c8076-e9ae-fde6-aacf-b0c3fae16a9e
export CUDA_DEVICE_ORDER=PCI_BUS_ID VLLM_USE_FLASHINFER_SAMPLER=0 UCX_NET_DEVICES=all
# NIXL 은 상대 장애 감지가 되는 전송(TCP/IB)을 요구해 공유메모리만으로는 연결이 안 되고, 4090↔3090 은 P2P 가 없어
# KV 가 호스트 경유 TCP 루프백으로 간다. 기본값은 0.5 GB/s 로 PD 를 망친다 → 세그먼트·버퍼·레인 튜닝으로 2.4 GB/s.
# (results/hetero/pd_xfer/README.md)
export UCX_TLS=cuda_copy,tcp,self UCX_TCP_TX_SEG_SIZE=4m UCX_TCP_RX_SEG_SIZE=4m UCX_TCP_SNDBUF=64m UCX_TCP_RCVBUF=64m \
       UCX_MAX_RNDV_LANES=4 UCX_RNDV_THRESH=64k
COMMON=(--dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.85 --no-enable-prefix-caching --max-num-seqs 256)
NS=${NS:-"50 100 200"}
MIXES=${MIXES:-"prefill:2048:128=1.0 decode:128:1024=1.0 mixed:2048:128=0.5,128:1024=0.5"}

PIDS=()
cleanup() { echo "cleanup"; kill "${PIDS[@]}" 2>/dev/null; sleep 4; kill -9 "${PIDS[@]}" 2>/dev/null; pkill -9 -f "toy_proxy_server|rr_proxy_server" 2>/dev/null; }
trap cleanup EXIT

serve() { # gpu port role(full|producer|consumer) side_port
  local gpu=$1 port=$2 role=$3 sp=$4 kv=()
  # CROSS=1: 32 레이어의 KV 블록을 하나의 연속 텐서로 묶어 전송 (enable_cross_layers_blocks). HND + FLASH_ATTN 필요.
  local extra=""; [ "${CROSS:-0}" = 1 ] && extra=',"kv_connector_extra_config":{"enable_cross_layers_blocks":"true"}'
  [ "$role" = producer ] && kv=(--kv-transfer-config "{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_producer\",\"kv_load_failure_policy\":\"fail\"$extra}")
  [ "$role" = consumer ] && kv=(--kv-transfer-config "{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_consumer\",\"kv_load_failure_policy\":\"fail\"$extra}")
  CUDA_VISIBLE_DEVICES=$gpu VLLM_NIXL_SIDE_CHANNEL_PORT=$sp \
    $V/vllm serve "$MODEL" --port "$port" "${COMMON[@]}" "${kv[@]}" > "$OUT/serve_$port.log" 2>&1 &
  PIDS+=($!)
}
wait_ready() { for i in $(seq 1 150); do curl -sf "127.0.0.1:$1/health" >/dev/null && { echo "  :$1 ready ($((i*3))s)"; return 0; }; sleep 3; done; echo ":$1 TIMEOUT"; exit 1; }

echo "== config $CONFIG  $(date +%T)"
case $CONFIG in
  A) serve $G4090 8100 full 5600; serve $G3090 8200 full 5601; wait_ready 8100; wait_ready 8200
     $V/python bench/hetero/rr_proxy_server.py --port 8192 --backends 127.0.0.1:8100 127.0.0.1:8200 > "$OUT/proxy.log" 2>&1 & PIDS+=($!) ;;
  B) serve $G4090 8100 full 5600; serve $G3090 8200 full 5601; wait_ready 8100; wait_ready 8200
     $V/python bench/hetero/rr_proxy_server.py --port 8192 --backends 127.0.0.1:8100 127.0.0.1:8200 --weights 2 1 > "$OUT/proxy.log" 2>&1 & PIDS+=($!) ;;
  C) serve $G4090 8100 producer 5600; serve $G3090 8200 consumer 5601; wait_ready 8100; wait_ready 8200
     $V/python bench/hetero/toy_proxy_server.py --port 8192 --prefiller-hosts 127.0.0.1 --prefiller-ports 8100 --decoder-hosts 127.0.0.1 --decoder-ports 8200 > "$OUT/proxy.log" 2>&1 & PIDS+=($!) ;;
  D) serve $G3090 8100 producer 5600; serve $G4090 8200 consumer 5601; wait_ready 8100; wait_ready 8200
     $V/python bench/hetero/toy_proxy_server.py --port 8192 --prefiller-hosts 127.0.0.1 --prefiller-ports 8100 --decoder-hosts 127.0.0.1 --decoder-ports 8200 > "$OUT/proxy.log" 2>&1 & PIDS+=($!) ;;
  *) echo "unknown config"; exit 1 ;;
esac
sleep 4; { curl -sf 127.0.0.1:8192/health >/dev/null || curl -sf 127.0.0.1:8192/healthcheck >/dev/null; } || { echo "proxy not up"; exit 1; }

# 워밍업 (핸드셰이크·첫 요청 비용 제거)
$V/python bench/hetero/loadgen.py --model "$MODEL" --n 8 --mix "512:32=1.0" --tag warm --out "$OUT/_warmup" > /dev/null 2>&1

for mixspec in $MIXES; do
  name=${mixspec%%:*}; mix=${mixspec#*:}
  for n in $NS; do
    cell="$OUT/${name}_n${n}"; mkdir -p "$cell"
    [ -s "$cell/summary.json" ] && { echo "skip $cell"; continue; }
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader -lms 500 > "$cell/gpu.csv" 2>/dev/null & SM=$!
    $V/python bench/hetero/loadgen.py --model "$MODEL" --n "$n" --mix "$mix" --rate inf --seed 0 --tag "$CONFIG" --out "$cell" 2>&1 | grep -v -E "^(INFO|WARNING)"
    kill $SM 2>/dev/null
    sleep 5   # 인스턴스 안정화
  done
done
echo "== done $CONFIG  $(date +%T)"
