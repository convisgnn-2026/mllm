#!/usr/bin/env python3
"""Sequential controlled runs; one runner owns both GPUs at a time.

Candidate spec: LLM_DIT[:only]  e.g. C_4090, S4_3090, S4_4090:llm (resident solo: both servers
loaded, only LLM requests replayed). Output: <out>/<LLM>_<DIT>[_only_<kind>]/<plan stem>/.
Default arguments reproduce the original 10/13 batch (C_4090, S4_3090 x seed0-4 -> controlled_runs).
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import random
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_OUT=ROOT/'results/hetero/controlled_runs'
DEFAULT_PLANS=ROOT/'results/hetero/controlled_plans'


def parse_candidate(spec):
    cand,_,only=spec.partition(':')
    llm,_,dit=cand.partition('_')
    if llm not in ('S4','S3','C','A','none') or dit not in ('4090','3090','split','none'):
        raise ValueError(f'bad candidate {spec}')
    only=only or 'both'
    if only not in ('both','llm','dit'):raise ValueError(f'bad only in {spec}')
    if (only=='llm' and llm=='none') or (only=='dit' and dit=='none'):raise ValueError(f'resident solo impossible for {spec}')
    if llm=='none' and dit=='none':raise ValueError(spec)
    return llm,dit,only


def case_dir(out,llm,dit,only,plan):
    name=f'{llm}_{dit}'+('' if only=='both' else f'_only_{only}')
    return out/name/plan.stem


def build_jobs(candidates,plans,shuffle_seed):
    jobs=[]
    for i,plan in enumerate(plans):
        order=list(candidates);random.Random(shuffle_seed+i).shuffle(order)
        for spec in order:jobs.append((plan,)+parse_candidate(spec))
    return jobs


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--candidates',default='C_4090,S4_3090',help='comma list of LLM_DIT[:only]')
    ap.add_argument('--plans',nargs='*',help='plan json files; default controlled_plans/seed{--seeds}.json')
    ap.add_argument('--seeds',default='0,1,2,3,4')
    ap.add_argument('--out',type=Path,default=DEFAULT_OUT)
    ap.add_argument('--drain-timeout',type=float,default=300)
    ap.add_argument('--shuffle-seed',type=int,default=1013)
    ap.add_argument('--summarize',action='store_true',help='refresh controlled_runs/comparison.md after each run (original 2-candidate batch only)')
    ap.add_argument('--wait-pid',type=int)
    ap.add_argument('--dry-run',action='store_true')
    ap.add_argument('--continue-on-failure',action='store_true',help='record a failed run and move on (Stage 0 smoke); default aborts the batch')
    args=ap.parse_args()
    candidates=[c for c in args.candidates.split(',') if c]
    plans=[Path(p).resolve() for p in args.plans] if args.plans else [DEFAULT_PLANS/f'seed{s}.json' for s in args.seeds.split(',')]
    for p in plans:
        if not p.exists():raise SystemExit(f'missing plan {p}')
    # Original behaviour: C_4090/S4_3090 x seeds with per-seed shuffle seeded by seed+1013.
    if set(candidates)=={'C_4090','S4_3090'} and not args.plans:
        jobs=[]
        for s in args.seeds.split(','):
            order=[('C','4090'),('S4','3090')];random.Random(int(s)+1013).shuffle(order)
            jobs+=[(DEFAULT_PLANS/f'seed{s}.json',l,d,'both') for l,d in order]
    else:
        jobs=build_jobs(candidates,plans,args.shuffle_seed)
    out=args.out.resolve()
    if args.dry_run:
        for plan,llm,dit,only in jobs:print(plan.name,llm,dit,only,'->',case_dir(out,llm,dit,only,plan).relative_to(ROOT))
        return
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'batch.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def state(status,**extra):
        (out/'batch_status.json').write_text(json.dumps(dict(status=status,updated_unix_s=time.time(),total_jobs=len(jobs),**extra),indent=2)+'\n')
    summarize=None
    if args.summarize:
        from compare_controlled import summarize
    try:
        if args.wait_pid:
            state('waiting_for_first_run',pid=args.wait_pid)
            deadline=time.monotonic()+1800
            while True:
                try:os.kill(args.wait_pid,0)
                except ProcessLookupError:break
                if time.monotonic()>deadline:raise RuntimeError('First-run wait timeout')
                time.sleep(5)
            first=jobs[0];first_dir=case_dir(out,*first[1:],first[0])
            if not (first_dir/'done').exists():raise RuntimeError(f'First run failed; inspect {first_dir}')
        done=0;failures=[]
        for plan,llm,dit,only in jobs:
            case=case_dir(out,llm,dit,only,plan)
            if (case/'done').exists():done+=1;continue
            if case.exists():raise RuntimeError(f'Incomplete output exists: {case}')
            state('running',plan=plan.name,config=case.parent.name,completed_jobs=done)
            env=dict(os.environ,CONTROLLED_PLAN=str(plan),CONTROLLED_OUT=str(case),DRAIN_TIMEOUT=str(args.drain_timeout))
            if only!='both':env['REPLAY_ONLY']=only
            case.parent.mkdir(parents=True,exist_ok=True)
            log=case.parent/f'{plan.stem}.console.log'
            with log.open('w') as f:
                result=subprocess.run(['bash','bench/hetero/run_mixed_modal.sh',llm,dit],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
            (case.parent/f'{plan.stem}.exit').write_text(f'{result.returncode}\n')
            if result.returncode:
                failures.append(dict(config=case.parent.name,plan=plan.name,exit=result.returncode,log=str(log)))
                if not args.continue_on_failure:raise RuntimeError(f'{case.parent.name} {plan.stem} failed (exit {result.returncode}); {log}')
                state('running',plan=plan.name,config=case.parent.name,completed_jobs=done,failures=failures);continue
            done+=1
            if summarize:summarize()
        state('complete',completed_jobs=done,failures=failures)
    except BaseException as ex:
        state('failed',error=str(ex));raise

if __name__=='__main__':main()
