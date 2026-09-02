#!/bin/bash
# Experiment U2 — how long does a MIG reconfiguration take, end to end?
#
# Gate for the dynamic manager.  Three costs, measured separately, five
# repetitions each, on one idle GPU:
#   (1) mig_off : destroy CI+GI, disable MIG mode            (nvidia-smi)
#   (2) mig_on  : enable MIG mode, create 2 x 3g.135gb         (nvidia-smi)
#   (3) cold    : start a serve_bench worker on the new slice and reach
#                 "ready" (weights loaded, first step done)
# Writes results/mig_reconfig.csv : rep, op, seconds
set -u
cd /home/jinwoo.kim27/vidbench
export HF_HOME=/home/jinwoo.kim27/hf
PY=/home/jinwoo.kim27/venv/jinwoo/bin/python
GPU=${1:-3}; REPS=${2:-5}
OUT=results/mig_reconfig.csv
FORBIDDEN="0 2 4 5 7"
for f in $FORBIDDEN; do [ "$GPU" = "$f" ] && { echo "ABORT: GPU $GPU off limits"; exit 1; }; done
u=$(nvidia-smi --query-gpu=uuid --format=csv,noheader --id=$GPU)
[ "$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$u")" -ne 0 ] && { echo "ABORT: GPU $GPU busy"; exit 1; }

now(){ date +%s.%N; }
echo "rep,op,seconds" > $OUT
for i in $(seq 1 $REPS); do
  # ---- (1) off
  t0=$(now)
  sudo nvidia-smi mig -i $GPU -dci >/dev/null 2>&1; sudo nvidia-smi mig -i $GPU -dgi >/dev/null 2>&1
  sudo nvidia-smi -i $GPU -mig 0 >/dev/null 2>&1
  until [ "$(nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader --id=$GPU)" = "Disabled" ]; do sleep 0.2; done
  t1=$(now); echo "$i,mig_off,$(echo "$t1 - $t0" | bc)" | tee -a $OUT
  # ---- (2) on, 2 x 3g
  t0=$(now)
  sudo nvidia-smi -i $GPU -mig 1 >/dev/null 2>&1
  until [ "$(nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader --id=$GPU)" = "Enabled" ]; do sleep 0.2; done
  sudo nvidia-smi mig -i $GPU -cgi 9,9 -C >/dev/null 2>&1
  until [ "$(nvidia-smi -L | sed -n "/GPU $GPU:/,/GPU $((GPU+1)):/p" | grep -c 'MIG 3g')" -eq 2 ]; do sleep 0.2; done
  t1=$(now); echo "$i,mig_on_2x3g,$(echo "$t1 - $t0" | bc)" | tee -a $OUT
  # ---- (3) worker cold start on the first new slice: load weights + one request
  S=$(nvidia-smi -L | sed -n "/GPU $GPU:/,/GPU $((GPU+1)):/p" | grep -oP 'MIG-[0-9a-f-]+' | head -1)
  t0=$(now)
  $PY bench/serve_bench.py --workload results/wl_smoke.json --policy reconfig_probe \
      --queue fcfs --slo-config video_3.0x --select any --devices "$S" --n-gpus 1 \
      --max-requests 1 -o /tmp/claude-1007/-home-jinwoo-kim27/860884a2-cad1-4010-b0d6-d1a149c6cbfe/scratchpad/reconfig_probe.json >/dev/null 2>&1
  t1=$(now); echo "$i,worker_cold_start_1req,$(echo "$t1 - $t0" | bc)" | tee -a $OUT
done
echo "--- summary (median seconds) ---"
for op in mig_off mig_on_2x3g worker_cold_start_1req; do
  printf "  %-24s %s\n" "$op" "$(grep ",$op," $OUT | cut -d, -f3 | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}')"
done
