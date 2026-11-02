#!/bin/bash
# Seed validation chain (started 2026-10-27): seed 902 then 903, 24 runs each, sequential under the global GPU lock.
cd /home/jinwoo.kim27/mllm
V=/home/jinwoo.kim27/venv/jinwoo/bin/python
for d in stage1_seed902 stage1_seed903; do
  echo "=== $(date '+%F %T') start $d"
  $V bench/hetero/run_stage1_refine.py run --out results/hetero/$d
  echo "=== $(date '+%F %T') end $d exit $?"
done
echo "=== $(date '+%F %T') chain complete"
