#!/bin/bash
# Stage 2 resident-solo chain: waits for the seed-validation chain, then LLM-only and DiT-only profiles (14 runs each).
cd /home/jinwoo.kim27/mllm
V=/home/jinwoo.kim27/venv/jinwoo/bin/python
for k in llm dit; do
  echo "=== $(date '+%F %T') start stage2_solo_$k"
  $V bench/hetero/run_stage2_solo.py run --out results/hetero/stage2_solo_$k --after-log results/hetero/stage1_validation/chain.log
  echo "=== $(date '+%F %T') end stage2_solo_$k exit $?"
done
echo "=== $(date '+%F %T') chain complete"
