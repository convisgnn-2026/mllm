#!/bin/bash
# Main experiment chain on GPUs 1,3,6 — runs the blocks below in order, one
# after another, skipping anything already on disk.  Each block ends with a
# commit + push (push failure is logged, never fatal).  Judgment criteria for
# every block are fixed in docs/research-flow.md §14 before this runs.
#
#   D    rescore fairness runs across SLO slack           (no GPU)
#   A    Full ×3 on the fair trace                        (GPU 1,3,6)
#   U2   MIG reconfiguration cost                         (GPU 3)
#   U1   heterogeneous layouts on the fair trace          (GPU 3,6)
#   B    mix sweep, D share 5/11/22/37 %                  (GPU 3,6)
#   C    load sweep 80→124 %                              (GPU 3,6)
#   E    three more seeds at 124 %                        (GPU 3,6)
#   U5   production-shaped skew (A-heavy, D 5 %)          (GPU 3,6)
#   U6   3-GPU scale: Full ×3 vs 2-way ×3                 (GPU 1,3,6; waits for side chain)
#
# Usage:  nohup scripts/chain_main.sh > logs/chain_main.log 2>&1 &
set -u
cd /home/jinwoo.kim27/vidbench
export HF_HOME=/home/jinwoo.kim27/hf
PY=/home/jinwoo.kim27/venv/jinwoo/bin/python
mkdir -p results/fair results/chain logs
FORBIDDEN="0 2 4 5 7"
SLO=video_3.0x
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

check(){ for g in "$@"; do
    for f in $FORBIDDEN; do [ "$g" = "$f" ] && { log "ABORT: GPU $g off limits"; exit 1; }; done
    u=$(nvidia-smi --query-gpu=uuid --format=csv,noheader --id=$g)
    n=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$u")
    [ "$n" -ne 0 ] && { log "ABORT: GPU $g busy"; exit 1; }; done; }
wait_idle(){ for i in $(seq 1 360); do [ "$(ps -ef|grep -c '[s]erve_bench.py')" -eq 0 ] && return; sleep 5; done; log "wait_idle timeout"; exit 1; }
mig_off(){ sudo nvidia-smi mig -i $1 -dci >/dev/null 2>&1; sudo nvidia-smi mig -i $1 -dgi >/dev/null 2>&1
           sudo nvidia-smi -i $1 -mig 0 >/dev/null 2>&1; sleep 2
           [ "$(nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader --id=$1)" = "Disabled" ] || { log "ABORT mig_off $1"; exit 1; }; }
mig_on(){  # gpu profile-list   e.g. mig_on 3 9,9   /  mig_on 3 9,19,19,19,19
           sudo nvidia-smi -i $1 -mig 1 >/dev/null 2>&1; sleep 2
           sudo nvidia-smi mig -i $1 -cgi $2 -C >/dev/null 2>&1; sleep 1
           n=$(nvidia-smi -L | sed -n "/GPU $1:/,/GPU $(($1+1)):/p" | grep -c 'MIG ')
           want=$(echo "$2" | tr ',' '\n' | wc -l)
           [ "$n" -eq "$want" ] || { log "mig_on $1 $2 FAILED: got $n slices, want $want"; return 1; }; }
slices(){ nvidia-smi -L | sed -n "/GPU $1:/,/GPU $(($1+1)):/p" | grep -oP 'MIG-[0-9a-f-]+' | paste -sd,; }
uuid(){ nvidia-smi --query-gpu=uuid --format=csv,noheader --id=$1 | tr -d ' '; }
run(){ # outdir tag ngpu devices queue workload   → outdir/tag_queue_r<rate>.json (matches results/fair convention)
  r=$($PY -c "import json,sys;d=json.load(open(sys.argv[1]));c=d.get('workload_config') or d.get('config') or {};print(f\"{float(c.get('rate',0)):.2f}\")" "$6")
  out=$1/$2_$5_r$r.json; [ -f "$out" ] && { log "skip $out"; return; }
  log "run $2 | $5 | $(basename $6)"
  $PY bench/serve_bench.py --workload "$6" --policy "$2" --queue "$5" --slo-config $SLO \
     --select match --devices "$4" --n-gpus "$3" -o "$out" 2>&1 | grep -E "cost:|makespan" | sed 's/^/    /'; }
wl(){ # out rate probs seed
  [ -f "$1" ] || $PY bench/workload.py -n 200 --rate $2 --cv 1.0 --probs "$3" --seed $4 -o "$1" >/dev/null; }
save(){ # message
  git add -A results docs scripts solver bench 2>/dev/null
  git -c user.name=chain -c user.email=chain@local commit -q -m "$1

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01S54jNA35j2HvDfv1cCjRqj" 2>/dev/null && log "committed: $1" || log "nothing to commit"
  timeout 120 git push -q origin main 2>&1 | tail -1 && log "pushed" || log "push failed (credentials?) — continuing"; }

FAIR_PROBS="0.50,0.25,0.15,0.10"
G1=$(uuid 1); G3=$(uuid 3); G6=$(uuid 6)

# ============================================================ D
log "########## D: rescore fairness runs across SLO slack ##########"
$PY solver/score_fair_slo.py > /dev/null && log "D done → results/fair_slo_sweep.md"
save "chain D: fairness runs rescored across SLO slack"

# ============================================================ A
log "########## A: Full x3 on fair trace (GPU 1,3,6) ##########"
wait_idle; mig_off 3; mig_off 6; check 1 3 6
for Q in fcfs sjf edf vtc; do run results/fair 3F 3 "$G1,$G3,$G6" $Q results/wlF_r0.85.json; done
$PY solver/judge_fair.py > results/chain/A_judge.txt 2>&1
save "chain A: Full x3 on the fair trace — partition vs buying"
touch results/chain/A.done            # side chain waits for this before taking GPU 1

# ============================================================ U2
log "########## U2: MIG reconfiguration cost (GPU 3) ##########"
[ -f results/mig_reconfig.csv ] || bash scripts/measure_mig_reconfig.sh 3 5 2>&1 | tail -4
save "chain U2: MIG reconfiguration cost"

# ============================================================ U1
log "########## U1: heterogeneous layouts on fair trace (GPU 3,6) ##########"
wait_idle; check 3 6
# (a) both GPUs 3g + 1g x4
mig_off 3; mig_off 6
if mig_on 3 9,19,19,19,19 && mig_on 6 9,19,19,19,19; then
  S=$(slices 3),$(slices 6); log "3g+1gx4 x2: $S"
  for Q in sjf edf; do run results/fair 2x3g1g4 2 "$S" $Q results/wlF_r0.85.json; done
else log "U1(a) skipped"; fi
# (b) both GPUs 4g + 1g x3    (profile 5 = 4g.135gb; abort loudly if the id is wrong)
mig_off 3; mig_off 6
if mig_on 3 5,19,19,19 && mig_on 6 5,19,19,19; then
  S=$(slices 3),$(slices 6); log "4g+1gx3 x2: $S"
  for Q in sjf edf; do run results/fair 2x4g1g3 2 "$S" $Q results/wlF_r0.85.json; done
else log "U1(b) skipped"; fi
# (c) Full 1 + 2-way 1 on the fair trace (same budget, already measured on `small`)
mig_off 3; mig_off 6; mig_on 6 9,9 || exit 1
S=$(slices 6); log "Full1+2way1: $G3,$S"
for Q in sjf edf; do run results/fair 1F1x2 2 "$G3,$S" $Q results/wlF_r0.85.json; done
save "chain U1: heterogeneous MIG layouts on the fair trace"

# ============================================================ B
log "########## B: mix sweep, D share 5/11/22/37% (GPU 3,6) ##########"
declare -A MIX=( [d05]="0.60,0.22,0.13,0.05" [d11]="$FAIR_PROBS" [d22]="0.40,0.23,0.15,0.22" [d37]="0.25,0.20,0.18,0.37" )
for m in d05 d22 d37; do wl results/wlB2_$m.json 0.85 "${MIX[$m]}" 7; done
mig_off 3; mig_off 6; check 3 6
for m in d05 d22 d37; do for Q in sjf edf; do run results/fair 2F_$m 2 "$G3,$G6" $Q results/wlB2_$m.json; done; done
mig_on 3 9,9 && mig_on 6 9,9 || exit 1; S=$(slices 3),$(slices 6)
for m in d05 d22 d37; do for Q in sjf edf; do run results/fair 2x2way_$m 2 "$S" $Q results/wlB2_$m.json; done; done
save "chain B: mix sweep — where does partition stop helping"

# ============================================================ C
log "########## C: load sweep 80→124% (GPU 3,6) ##########"
for R in 0.60 0.65 0.70 0.75; do wl results/wlF_r$R.json $R "$FAIR_PROBS" 7; done
# 2-way is still up from B
for R in 0.60 0.65 0.70 0.75; do for Q in sjf edf; do run results/fair 2x2way 2 "$S" $Q results/wlF_r$R.json; done; done
wait_idle; mig_off 3; mig_off 6; check 3 6
for R in 0.60 0.65 0.70 0.75; do for Q in sjf edf; do run results/fair 2F 2 "$G3,$G6" $Q results/wlF_r$R.json; done; done
save "chain C: load sweep — where is the knee"

# ============================================================ E
log "########## E: seeds 8,9,10 at 124% (GPU 3,6) ##########"
for s in 8 9 10; do wl results/wlF_r0.85_s$s.json 0.85 "$FAIR_PROBS" $s; done
# Full is up from C
for s in 8 9 10; do for Q in sjf edf; do run results/fair 2F_s$s 2 "$G3,$G6" $Q results/wlF_r0.85_s$s.json; done; done
mig_on 3 9,9 && mig_on 6 9,9 || exit 1; S=$(slices 3),$(slices 6)
for s in 8 9 10; do for Q in sjf edf; do run results/fair 2x2way_s$s 2 "$S" $Q results/wlF_r0.85_s$s.json; done; done
save "chain E: three more seeds at 124%"

# ============================================================ U5
log "########## U5: production-shaped skew, A 65% D 5% (GPU 3,6) ##########"
wl results/wlU5_r0.85.json 0.85 "0.65,0.20,0.10,0.05" 7
# 2-way is up from E
for Q in sjf edf; do run results/fair 2x2way_skewreal 2 "$S" $Q results/wlU5_r0.85.json; done
wait_idle; mig_off 3; mig_off 6; check 3 6
for Q in sjf edf; do run results/fair 2F_skewreal 2 "$G3,$G6" $Q results/wlU5_r0.85.json; done
save "chain U5: production-shaped skew"

# ============================================================ U6
log "########## U6: 3-GPU scale, Full x3 vs 2-way x3 (GPU 1,3,6) ##########"
log "waiting for side chain to release GPU 1 ..."
until [ -f results/chain/side.done ]; do sleep 60; done
wait_idle; mig_off 3; mig_off 6; check 1 3 6
# Full x3 already run in A (3F_*.json) — only the partitioned side is new
mig_on 1 9,9 && mig_on 3 9,9 && mig_on 6 9,9 || exit 1
S=$(slices 1),$(slices 3),$(slices 6); log "2-way x3: $S"
for Q in sjf edf; do run results/fair 3x2way 3 "$S" $Q results/wlF_r0.85.json; done
wait_idle; mig_off 1; mig_off 3; mig_off 6
save "chain U6: 3-GPU scale"

log "########## CHAIN MAIN DONE ##########"
touch results/chain/main.done
