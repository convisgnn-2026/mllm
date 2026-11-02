#!/bin/bash
# Stage 2b resident-solo chain: refined-alpha profiles for S4_4090 and C_4090 (8 runs LLM-only, 8 runs DiT-only).
cd /home/jinwoo.kim27/mllm
V=/home/jinwoo.kim27/venv/jinwoo/bin/python
for k in llm dit; do
  echo "=== $(date '+%F %T') start stage2_solo_${k}_b"
  $V bench/hetero/run_stage2_solo.py run --out results/hetero/stage2_solo_${k}_b
  echo "=== $(date '+%F %T') end stage2_solo_${k}_b exit $?"
done
echo "=== $(date '+%F %T') chain complete"
