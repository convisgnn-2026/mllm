#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export HF_HOME=/home/jinwoo.kim27/hf
PY=/home/jinwoo.kim27/venv/jinwoo/bin/python
mkdir -p results/small
for R in 0.8 1.0 1.2; do
  for CFG in "2F 2 GPU-1b5a091d-563d-e5ed-8201-2f3a95e09483,GPU-b917bb97-e5dd-1410-5902-8ce63987c0cd" "1F+1x2 2 GPU-1b5a091d-563d-e5ed-8201-2f3a95e09483,MIG-e94c4746-8a74-5abe-ae64-6aeaf3e1ed8c,MIG-f70b0bbf-208f-5e11-826a-abc2b4a5a7b3" "0F+2x2 2 MIG-ea951677-a1af-53cd-879c-3652e1ab5332,MIG-c8f67af1-7eb2-58ae-8cd7-25d82a0fec12,MIG-e94c4746-8a74-5abe-ae64-6aeaf3e1ed8c,MIG-f70b0bbf-208f-5e11-826a-abc2b4a5a7b3"; do
    set -- $CFG; TAG=$1; NG=$2; DEV=$3
    OUT=results/small/${TAG}_r${R}.json
    [ -f "$OUT" ] && { echo "skip $TAG r$R"; continue; }
    echo "=== $TAG  rate $R ==="
    $PY bench/serve_bench.py --workload results/wlS_r${R}.json --policy "$TAG" \
        --queue srpt --select match --devices "$DEV" --n-gpus $NG -o "$OUT" \
        2>&1 | grep -E "workers ready|cost:|makespan"
  done
done
echo "SMALL DONE"
