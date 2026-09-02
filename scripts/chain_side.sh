#!/bin/bash
# Side chain on GPU 1 — profiling work that does not need the serving pool.
# Waits for the main chain to finish block A (which uses GPU 1), then:
#
#   F   torch.compile on/off, 14 templates, 148 SM   → how much of b is launch overhead
#   G   Wan2.1-14B (Diffusers), 14 templates, 148 SM → does b survive a 10x larger model;
#       which MIG slice stops fitting.  Downloads ~40 GB from HF on first use.
#
# Usage:  nohup scripts/chain_side.sh > logs/chain_side.log 2>&1 &
set -u
cd /home/jinwoo.kim27/vidbench
export HF_HOME=/home/jinwoo.kim27/hf
PY=/home/jinwoo.kim27/venv/jinwoo/bin/python
mkdir -p results/chain logs
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
G1=$(nvidia-smi --query-gpu=uuid --format=csv,noheader --id=1 | tr -d ' ')
save(){ git add -A results 2>/dev/null
  git -c user.name=chain -c user.email=chain@local commit -q -m "$1

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01S54jNA35j2HvDfv1cCjRqj" 2>/dev/null && log "committed: $1" || log "nothing to commit"
  timeout 120 git push -q origin main 2>&1 | tail -1 && log "pushed" || log "push failed — continuing"; }

log "waiting for main chain block A to release GPU 1 ..."
until [ -f results/chain/A.done ]; do sleep 60; done
u=$G1; while [ "$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$u")" -ne 0 ]; do sleep 10; done
[ "$(nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader --id=1)" = "Disabled" ] || { log "GPU 1 is in MIG mode; abort"; exit 1; }

# ============================================================ F
log "########## F: compile on/off, 148 SM (GPU 1) ##########"
if [ ! -f results/stages_compile.csv ]; then
  CUDA_VISIBLE_DEVICES=$G1 $PY bench/profile_stages.py --sweep --steps 20 --compile \
      --tag full_compile --out results/stages_compile.csv 2>&1 | grep -E "DiT|VAE|device|OOM|err" | sed 's/^/    /'
fi
save "chain F: torch.compile profile — launch overhead vs b"

# ============================================================ G
log "########## G: Wan2.1-14B profile, 148 SM (GPU 1) ##########"
if [ ! -f results/stages_14b.csv ]; then
  CUDA_VISIBLE_DEVICES=$G1 $PY bench/profile_stages.py --sweep --steps 20 \
      --model Wan-AI/Wan2.1-T2V-14B-Diffusers --tag full_14b \
      --out results/stages_14b.csv 2>&1 | grep -E "DiT|VAE|device|OOM|err|loading" | sed 's/^/    /'
fi
save "chain G: Wan2.1-14B stage profile"

log "########## CHAIN SIDE DONE ##########"
touch results/chain/side.done
