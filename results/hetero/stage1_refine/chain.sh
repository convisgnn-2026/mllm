#!/bin/bash
# Stage 1 follow-up chain (started 2026-10-25): KV controls first, then alpha refinement. Sequential; each batch takes the global GPU lock.
cd /home/jinwoo.kim27/mllm
V=/home/jinwoo.kim27/venv/jinwoo/bin/python
for d in stage1_kv2g stage1_kv8g stage1_refine; do
  echo "=== $(date '+%F %T') start $d"
  $V bench/hetero/run_stage1_refine.py run --out results/hetero/$d
  echo "=== $(date '+%F %T') end $d exit $?"
done
echo "=== $(date '+%F %T') chain complete"
