#!/usr/bin/env python3
"""Sequential 5-seed comparison; one runner owns both GPUs at a time."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import random
import subprocess
import time
from compare_controlled import summarize

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/hetero/controlled_runs'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--wait-pid',type=int);args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'batch.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def state(status,**extra):
        (OUT/'batch_status.json').write_text(json.dumps(dict(status=status,updated_unix_s=time.time(),**extra),indent=2)+'\n')
    try:
        if args.wait_pid:
            state('waiting_for_first_run',pid=args.wait_pid)
            deadline=time.monotonic()+1800
            while True:
                try:os.kill(args.wait_pid,0)
                except ProcessLookupError:break
                if time.monotonic()>deadline:raise RuntimeError('First-run wait timeout')
                time.sleep(5)
            if not (OUT/'C_4090/seed0/done').exists():raise RuntimeError('First run failed; inspect logs')
        jobs=[]
        for seed in range(5):
            configs=[('C','4090'),('S4','3090')];random.Random(seed+1013).shuffle(configs)
            for llm,dit in configs:jobs.append((seed,llm,dit))
        for seed,llm,dit in jobs:
            case=OUT/f'{llm}_{dit}'/f'seed{seed}'
            if (case/'done').exists():continue
            if case.exists():raise RuntimeError(f'Incomplete output exists: {case}')
            state('running',seed=seed,config=f'{llm}_{dit}')
            env=dict(os.environ,CONTROLLED_PLAN=str(ROOT/f'results/hetero/controlled_plans/seed{seed}.json'))
            log=OUT/f'{llm}_{dit}_seed{seed}.console.log'
            with log.open('w') as f:
                result=subprocess.run(['bash','bench/hetero/run_mixed_modal.sh',llm,dit],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError(f'{llm}_{dit} seed{seed} failed; {log}')
            summarize()
        state('complete',completed_runs=summarize())
    except BaseException as ex:
        state('failed',error=str(ex));raise

if __name__=='__main__':main()
