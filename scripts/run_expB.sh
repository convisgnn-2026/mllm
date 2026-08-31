#!/bin/bash
# Experiment A+B: does a well-sliced N-1 GPU pool match N full GPUs?
#
# Cost model: you pay for provisioned physical GPUs x wall time, so slicing one
# GPU into 7 workers still costs one GPU.  The question is whether the extra
# concurrency slicing buys is worth the per-request slowdown it costs.
#
# GPU roles:  1, 3, 5 stay full;  6 is the one we re-partition.
# GPUs 0,2,4,7 are never touched -- root's tritonserver lives there.
set -u
cd /home/jinwoo.kim27/vidbench
export HF_HOME=/home/jinwoo.kim27/hf
PY=/home/jinwoo.kim27/venv/jinwoo/bin/python
mkdir -p results/expB logs

FORBIDDEN="0 2 4 7"
check_free() {                       # refuse to touch a GPU someone else is on
  for g in "$@"; do
    for f in $FORBIDDEN; do [ "$g" = "$f" ] && { echo "ABORT: GPU $g is off limits"; exit 1; }; done
    u=$(nvidia-smi --query-gpu=uuid --format=csv,noheader --id=$g)
    n=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$u")
    [ "$n" -ne 0 ] && { echo "ABORT: GPU $g busy ($n procs)"; exit 1; }
  done
}

wait_idle() {                        # never reconfigure under a live process
  for i in $(seq 1 120); do
    n=$(ps -ef | grep -c "[s]erve_bench.py"); [ "$n" -eq 0 ] && return 0; sleep 5
  done
  echo "ABORT: serve_bench.py still running after 10 min"; exit 1
}

mig_off() {                          # and verify, instead of assuming
  sudo nvidia-smi mig -i $1 -dci >/dev/null 2>&1
  sudo nvidia-smi mig -i $1 -dgi >/dev/null 2>&1
  sudo nvidia-smi -i $1 -mig 0 >/dev/null 2>&1
  sleep 2
  m=$(nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader --id=$1)
  [ "$m" = "Disabled" ] || { echo "ABORT: GPU $1 MIG is '$m', expected Disabled"; exit 1; }
  echo "GPU $1 MIG off"
}
mig_on()  { sudo nvidia-smi -i $1 -mig 1 >/dev/null 2>&1; shift; \
            sudo nvidia-smi mig -i $MIGGPU -cgi $1 -C 2>&1 | grep -c Successfully; }
slices()  { nvidia-smi -L | sed -n "/GPU $1:/,/GPU $(($1+1)):/p" | grep -oP 'MIG-[0-9a-f-]+' | paste -sd,; }

MIGGPU=6

# Numeric CUDA_VISIBLE_DEVICES indices shift as soon as any GPU on the box is
# in MIG mode -- that is how an earlier attempt landed on GPUs 4 and 7.  Resolve
# every full GPU to its UUID once, up front, and use only UUIDs from here on.
uuid_of() { nvidia-smi --query-gpu=uuid --format=csv,noheader --id=$1 | tr -d ' '; }
G1=$(uuid_of 1); G3=$(uuid_of 3); G5=$(uuid_of 5); G6=$(uuid_of 6)
echo "GPU uuids: 1=$G1  3=$G3  5=$G5  6=$G6"

run() {   # run <tag> <n_gpus> <devices> <queue> <select> <workload-name>
  local tag=$1 ng=$2 dev=$3 q=$4 sel=$5 wl=$6
  local out=results/expB/${tag}_${wl}.json
  [ -f "$out" ] && { echo "  skip $tag/$wl"; return; }
  echo "--- $tag  ($wl)  n_gpus=$ng  devices=$(echo $dev | cut -c1-40)..."
  $PY bench/serve_bench.py --workload results/wlB_${wl}.json --policy "$tag" \
      --queue "$q" --select "$sel" --devices "$dev" --n-gpus "$ng" -o "$out" \
      2>&1 | grep -E "workers ready|cost:|makespan"
}

wait_idle
echo "######################## LAYOUT 0: no MIG ########################"
mig_off 3; mig_off 6
sleep 5
check_free 1 3 5 6
nvidia-smi --query-gpu=index,mig.mode.current --format=csv,noheader --id=1,3,5,6
for W in evenly skewed; do
  run C1_4full      4 "$G1,$G3,$G5,$G6" srpt any  $W
  run C1f_4full_fcfs 4 "$G1,$G3,$G5,$G6" fcfs any  $W
  run C2_3full      3 "$G1,$G3,$G5"   srpt any  $W
  run C3_2full      2 "$G1,$G3"     srpt any  $W
done

echo "################ LAYOUT 1: GPU$MIGGPU = 2 x 3g.135gb ################"
wait_idle; check_free $MIGGPU
mig_on $MIGGPU 9,9
S2=$(slices $MIGGPU); echo "2-way slices: $S2"
for W in evenly skewed; do
  run C4_2full_1x2way 3 "$G1,$G3,$S2" srpt match $W
  run C6_1full_1x2way 2 "$G1,$S2"   srpt match $W
  run A2_1gpu_2way    1 "$S2"     srpt match $W
done

echo "################ LAYOUT 2: GPU$MIGGPU = 7 x 1g.34gb ################"
wait_idle; check_free $MIGGPU
mig_off $MIGGPU; sleep 3; mig_on $MIGGPU 19,19,19,19,19,19,19
S7=$(slices $MIGGPU); echo "7-way slices: $S7"
for W in evenly skewed; do
  run C5_2full_1x7way 3 "$G1,$G3,$S7" srpt match $W
  run A3_1gpu_7way    1 "$S7"     srpt match $W
done

echo "EXPB ALL DONE"
