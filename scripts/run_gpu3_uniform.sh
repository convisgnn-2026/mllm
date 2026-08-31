#!/bin/bash
# Requires GPU 3 reconfigured to 7 x 1g.34gb first.
cd /home/jinwoo.kim27/vidbench
export HF_HOME=/home/jinwoo.kim27/hf
PY=/home/jinwoo.kim27/venv/jinwoo/bin/python
DEVS=$(nvidia-smi -L | grep -oP 'MIG-[0-9a-f-]+' | paste -sd,)
N=$(echo $DEVS | tr ',' '\n' | wc -l)
echo "slices: $N"
for D in evenly long_seqlen_high; do for R in 0.10 0.20; do
  OUT=results/runs/mig_uniform_${D}_r${R}.json
  [ -f "$OUT" ] && { echo "skip $OUT"; continue; }
  echo "=== mig_uniform $D r$R ==="
  $PY bench/serve_bench.py --workload results/wl_${D}_r${R}.json --policy mig_uniform \
      --devices "$DEVS" -o $OUT 2>&1 | grep -E "policy=|workers ready|makespan"
done; done
echo "GPU3 UNIFORM DONE"
