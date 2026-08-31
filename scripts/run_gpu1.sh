#!/bin/bash
# NOTE: serve_bench.py's worker sets CUDA_VISIBLE_DEVICES from --devices itself,
# so --devices must carry the PHYSICAL index. Do not also export it here:
# exporting it in the parent does not compose, it gets overwritten.
cd /home/jinwoo.kim27/vidbench
export HF_HOME=/home/jinwoo.kim27/hf
PY=/home/jinwoo.kim27/venv/jinwoo/bin/python
GPU=1

# safety: refuse to run if anyone else is on the target GPU (shared server)
UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader --id=$GPU)
OTHERS=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -c "$UUID")
if [ "$OTHERS" -ne 0 ]; then echo "ABORT: GPU $GPU already has $OTHERS process(es)"; exit 1; fi
echo "GPU $GPU is free ($UUID)"

for D in evenly long_seqlen_high; do for R in 0.10 0.20; do for P in full_fcfs full_srpt; do
  OUT=results/runs/${P}_${D}_r${R}.json
  [ -f "$OUT" ] && { echo "skip $OUT"; continue; }
  echo "=== $P $D r$R ==="
  $PY bench/serve_bench.py --workload results/wl_${D}_r${R}.json --policy $P \
      --devices $GPU -o $OUT 2>&1 | grep -E "policy=|workers ready|makespan"
done; done; done
echo "GPU1 ALL DONE"
