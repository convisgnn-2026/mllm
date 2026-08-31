#!/bin/bash
set -u
cd /home/jinwoo.kim27/vidbench
export HF_HOME=/home/jinwoo.kim27/hf
PY=/home/jinwoo.kim27/venv/jinwoo/bin/python
mkdir -p results/fair
FORBIDDEN="0 2 4 5 7"          # 0,2,4,7 = tritonserver ; 5 = another user
check() { for g in "$@"; do
    for f in $FORBIDDEN; do [ "$g" = "$f" ] && { echo "ABORT: GPU $g off limits"; exit 1; }; done
    u=$(nvidia-smi --query-gpu=uuid --format=csv,noheader --id=$g)
    n=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$u")
    [ "$n" -ne 0 ] && { echo "ABORT: GPU $g busy"; exit 1; }; done; }
wait_idle(){ for i in $(seq 1 120); do [ "$(ps -ef|grep -c '[s]erve_bench.py')" -eq 0 ] && return; sleep 5; done; exit 1; }
mig_off(){ sudo nvidia-smi mig -i $1 -dci >/dev/null 2>&1; sudo nvidia-smi mig -i $1 -dgi >/dev/null 2>&1
           sudo nvidia-smi -i $1 -mig 0 >/dev/null 2>&1; sleep 2
           [ "$(nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader --id=$1)" = "Disabled" ] || { echo "ABORT mig_off $1"; exit 1; }; }
mig_on(){ sudo nvidia-smi -i $1 -mig 1 >/dev/null 2>&1; sleep 2; sudo nvidia-smi mig -i $1 -cgi $2 -C >/dev/null 2>&1; }
slices(){ nvidia-smi -L | sed -n "/GPU $1:/,/GPU $(($1+1)):/p" | grep -oP 'MIG-[0-9a-f-]+' | paste -sd,; }
run(){ # tag ngpu devices queue rate
  out=results/fair/$1_$4_r$5.json; [ -f "$out" ] && { echo "skip $1/$4/r$5"; return; }
  echo "--- $1 | $4 | r$5"
  $PY bench/serve_bench.py --workload results/wlF_r$5.json --policy "$1" \
     --queue "$4" --slo-config video_3.0x --select match --devices "$3" --n-gpus "$2" \
     -o "$out" 2>&1 | grep -E "cost:|makespan"; }

wait_idle
echo "########## A: Full 2장 (GPU 1,3) ##########"
mig_off 3; mig_off 6; check 1 3
G1=$(nvidia-smi --query-gpu=uuid --format=csv,noheader --id=1|tr -d ' ')
G3=$(nvidia-smi --query-gpu=uuid --format=csv,noheader --id=3|tr -d ' ')
for Q in fcfs sjf edf vtc; do for R in 0.55 0.85 1.10; do run "2F" 2 "$G1,$G3" $Q $R; done; done

echo "########## B: 2분할 x2 (GPU 3,6) ##########"
wait_idle; check 3 6
mig_on 3 9,9; mig_on 6 9,9
S=$(slices 3),$(slices 6); echo "slices: $S"
for Q in fcfs sjf edf vtc; do for R in 0.55 0.85 1.10; do run "2x2way" 2 "$S" $Q $R; done; done
echo "FAIR DONE"
