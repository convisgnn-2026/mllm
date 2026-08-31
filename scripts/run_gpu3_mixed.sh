#!/bin/bash
cd /home/jinwoo.kim27/vidbench
export HF_HOME=/home/jinwoo.kim27/hf
PY=/home/jinwoo.kim27/venv/jinwoo/bin/python
DEVS=$(nvidia-smi -L | grep -oP 'MIG-[0-9a-f-]+' | paste -sd,)
for D in evenly long_seqlen_high; do for R in 0.10 0.20; do
  OUT=results/runs/mig_mixed_${D}_r${R}.json
  [ -f "$OUT" ] && { echo "skip $OUT"; continue; }
  echo "=== mig_mixed $D r$R ==="
  $PY bench/serve_bench.py --workload results/wl_${D}_r${R}.json --policy mig_mixed \
      --devices "$DEVS" -o $OUT 2>&1 | grep -E "policy=|workers ready|makespan"
done; done
echo "GPU3 MIXED DONE"
