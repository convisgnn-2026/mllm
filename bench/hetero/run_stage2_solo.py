#!/usr/bin/env python3
"""Stage 2 resident-solo profiles on top of the frozen Stage 1 runner.

Resident solo = the candidate's full deployment is loaded (both LLM and DiT servers resident, same
cards, same KV bytes) but only one service's requests from the *same* Stage 1 plan are replayed
(run_mixed_modal.sh REPLAY_ONLY=llm|dit). Comparing with the mixed run of the same plan isolates
runtime interference from memory placement.

  prepare --kind llm|dit --out DIR --runs C_4090:1.0,... [--seed 901]
  dry-run --out DIR
  run     --out DIR [--after-log FILE]   (--after-log: wait until FILE contains 'chain complete')

run_stage1_pilot.py and stage1_report.py are not modified. The wrapper overrides, at run time only:
  * run_env: adds REPLAY_ONLY=<kind> (the pilot strips it on purpose);
  * validate_records: checks the plan's requests *of the replayed kind* (identity, arrival,
    measured flag, denominator, output length), so a solo run is not flagged as missing requests;
  * the per-run report: stage1_report.build assumes mixed replay, so it is disabled here and
    stage2_report.py scores solo runs.
The expected prompt hash per job is the replay's own rule: sha256 over (kind, idx, prompt) of the
replayed kind in plan order.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_stage1_pilot as P  # noqa: E402
import stage1_report as R  # noqa: E402

ROOT = P.ROOT


def solo_prompt_sha(plan, kind, tok):
    from loadgen import make_prompts_lens
    reqs = [r for r in plan['requests'] if r['kind'] == kind]
    if kind == 'llm':
        prompts = make_prompts_lens(tok, [(r['in_len'], r['out_len']) for r in reqs], plan['seed'])
        items = [('llm', r['idx'], p) for r, (p, _, _) in zip(reqs, prompts)]
    else:
        items = [('dit', r['idx'], r['prompt']) for r in reqs]
    return hashlib.sha256(json.dumps(items).encode()).hexdigest()


def prepare(args):
    from transformers import AutoTokenizer
    P.SEED = args.seed
    tok = AutoTokenizer.from_pretrained(P.MODEL, local_files_only=True)
    jobs = []
    for item in [s for s in args.runs.split(',') if s.strip()]:
        c, _, a = item.partition(':'); a = float(a)
        if c not in P.CANDIDATES:
            raise SystemExit(f'unknown candidate {c}')
        llm, dit = c.split('_', 1)
        jobs.append((c, a))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    alphas = sorted({a for _, a in jobs}); plans = {}
    for a in alphas:
        p = P.DEFAULT_PLANS / f'{P.plan_stem(a)}.json'
        if not p.exists():
            raise SystemExit(f'missing plan {p}: solo profiles reuse existing Stage 1 plans only')
        plan = json.loads(p.read_text())
        assert plan['seed'] == P.SEED and plan['duration_s'] == P.DURATION_S and plan['warmup_s'] == P.WARMUP_S
        n_meas = sum(r['measured'] for r in plan['requests'] if r['kind'] == args.kind)
        plans[a] = dict(alpha=a, plan_stem=P.plan_stem(a), path=str(p.relative_to(ROOT)), plan_sha256=P.sha256_file(p),
                        rates=plan['rates'], replay=args.kind, n_measured=n_meas, prompt_sha256=solo_prompt_sha(plan, args.kind, tok))
        print(json.dumps({k: plans[a][k] for k in ('alpha', 'plan_stem', 'n_measured', 'prompt_sha256')}))
    runs = []
    for i, a in enumerate(alphas):
        order = [c for c, x in jobs if x == a]; random.Random(P.SEED + 200 + i).shuffle(order)
        x = plans[a]
        for c in order:
            runs.append(dict(run_id=f'{c}__{x["plan_stem"]}__only_{args.kind}', candidate=c, alpha=a, plan_stem=x['plan_stem'], plan=x['path'],
                             plan_sha256=x['plan_sha256'], prompt_sha256=x['prompt_sha256'], replay=args.kind,
                             expected_servers=P.expected_servers(c.split('_')[0]), cards=P.cards_of(c)))
    pilot = json.loads((P.DEFAULT_OUT / 'protocol.json').read_text())
    proto = dict(stage='stage2_resident_solo', purpose=args.purpose, replay=args.kind, parent_protocol_sha256=pilot['protocol_sha256'],
                 seed=P.SEED, candidates=sorted({c for c, _ in jobs}, key=P.CANDIDATES.index), alphas=alphas, lambda0=P.LAMBDA0,
                 warmup_s=P.WARMUP_S, duration_s=P.DURATION_S, drain_timeout_s=P.DRAIN_S, model=P.MODEL, dit_model=P.DMODEL,
                 dit=dict(size=512, steps=28, text_encoder_3=False), kv_bytes=P.KV_BYTES, kv_tokens_expected=P.KV_TOKENS_EXPECTED,
                 slo_grid=P.SLO_GRID, definitions=pilot['definitions'], env=dict(P.FIXED_ENV, REPLAY_ONLY=args.kind),
                 order_rule=f'alpha ascending; candidates shuffled within alpha with random.Random({P.SEED + 200}+i)',
                 plans=[plans[a] for a in alphas], runs=runs,
                 harness=dict(P.harness_hashes(), run_stage2_solo_py=P.sha256_file(__file__)))
    proto['protocol_sha256'] = hashlib.sha256(json.dumps({k: v for k, v in proto.items() if k != 'harness'}, sort_keys=True).encode()).hexdigest()
    proto['created_unix_s'] = time.time()
    try:
        proto['git_head'] = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        proto['git_head'] = None
    target = out / 'protocol.json'
    if target.exists():
        old = json.loads(target.read_text())
        if old['protocol_sha256'] != proto['protocol_sha256']:
            raise SystemExit(f'protocol.json already frozen with a different hash in {out}')
        print('protocol unchanged'); return
    P.atomic_write(target, proto)
    print(f'protocol frozen {proto["protocol_sha256"][:12]} -> {target} ({len(runs)} runs, replay={args.kind})')


def patch(out):
    proto = json.loads((Path(out) / 'protocol.json').read_text())
    kind = proto['replay']
    base_env = P.run_env

    def run_env(plan_path, case):
        env = base_env(plan_path, case); env['REPLAY_ONLY'] = kind; return env

    def validate_records(recs, manifest, plan):
        sub = dict(plan, requests=[r for r in plan['requests'] if r['kind'] == kind])
        other = 'dit' if kind == 'llm' else 'llm'
        m = json.loads(json.dumps(manifest))
        m.setdefault('measured', {}).setdefault(other, {})['n'] = 0
        res = R.validate_records(recs, m, sub)
        if manifest.get('only') != kind:
            res['problems'].append(f"manifest only={manifest.get('only')} != {kind}"); res['ok'] = False
        return res

    P.run_env = run_env
    P.validate_records = validate_records
    return proto


def wait_after(log):
    log = Path(log)
    while not (log.exists() and 'chain complete' in log.read_text()):
        time.sleep(60)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'dry-run', 'run'):
        p = sub.add_parser(name); p.add_argument('--out', type=Path, required=True)
        if name == 'prepare':
            p.add_argument('--kind', choices=['llm', 'dit'], required=True); p.add_argument('--runs', required=True)
            p.add_argument('--seed', type=int, default=901)
            p.add_argument('--purpose', default='resident-solo profile (both servers loaded, one service replayed) on the Stage 1 plans, for selector comparison')
        if name == 'run':
            p.add_argument('--after-log', type=Path); p.add_argument('--max-infra-failures', type=int, default=2)
            p.add_argument('--allow-harness-drift', action='store_true')
    a = ap.parse_args()
    if a.command == 'prepare':
        prepare(a)
    elif a.command == 'dry-run':
        patch(a.out); P.dry_run(SimpleNamespace(out=a.out, skip_smoke_gate=True))
    else:
        if a.after_log:
            wait_after(a.after_log)
        patch(a.out)
        P.run(SimpleNamespace(out=a.out, skip_smoke_gate=True, wait_smoke=False, report=False,
                              max_infra_failures=a.max_infra_failures, allow_harness_drift=a.allow_harness_drift))


if __name__ == '__main__':
    main()
