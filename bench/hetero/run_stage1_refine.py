#!/usr/bin/env python3
"""Stage 1 follow-up batches (alpha refinement, KV-size controls) on top of the frozen pilot runner.

Reuses run_stage1_pilot.py unchanged (plans, validation, strict KV/parity checks, runner loop,
stage1_report.build) but lets a batch be an explicit list of (candidate, alpha) jobs with one
KV_BYTES value, written to its own output directory with its own frozen protocol.json. The
pilot's protocol, plans and results are never modified; new alpha plans are generated with the
same seed/trace/rule and stored next to the pilot plans.

  prepare --out DIR --runs C_4090:1.1,C_split:1.1 [--kv-gib 4] [--stage NAME] [--purpose TEXT]
  dry-run --out DIR
  run     --out DIR            (smoke gate already passed for the pilot; it is referenced, not re-run)
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

ROOT = P.ROOT
# Llama-3.2-3B KV block: 28 layers x (K,V) x 8 kv heads x 128 dim x 16 tokens x 2 bytes (bf16).
BLOCK_BYTES = 28 * 2 * 8 * 128 * 16 * 2
assert (P.KV_BYTES // BLOCK_BYTES) * 16 == P.KV_TOKENS_EXPECTED, 'block-size assumption must reproduce the pilot KV token count'


def kv_tokens_expected(kv_bytes):
    return (kv_bytes // BLOCK_BYTES) * 16


def parse_runs(spec):
    jobs = []
    for item in [s for s in spec.split(',') if s.strip()]:
        cand, _, alpha = item.strip().partition(':')
        if cand not in P.CANDIDATES:
            raise SystemExit(f'unknown candidate {cand}')
        jobs.append((cand, float(alpha)))
    if not jobs:
        raise SystemExit('no runs')
    return jobs


def prepare(args):
    P.SEED = args.seed  # plan_stem / plan generation / validation all read this module global
    import controlled_mixed
    from transformers import AutoTokenizer
    jobs = parse_runs(args.runs)
    kv_bytes = int(args.kv_gib * (1 << 30))
    plans_dir = Path(args.plans_dir); plans_dir.mkdir(parents=True, exist_ok=True)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    trace = Path(args.trace)
    with trace.open() as f:
        trace_rows = list(csv.DictReader(f))
    tok = AutoTokenizer.from_pretrained(P.MODEL, local_files_only=True)
    alphas = sorted({a for _, a in jobs})
    plans = {}
    for alpha in alphas:
        p = plans_dir / f'{P.plan_stem(alpha)}.json'
        if not p.exists():
            ns = SimpleNamespace(trace=trace, model=P.MODEL, seed=P.SEED, duration=P.DURATION_S, warmup=P.WARMUP_S,
                                 llm_rate=P.LAMBDA0['llm'] * alpha, dit_rate=P.LAMBDA0['dit'] * alpha, out=p)
            controlled_mixed.prepare(ns)
        plan = json.loads(p.read_text())
        assert plan['seed'] == P.SEED and plan['warmup_s'] == P.WARMUP_S and plan['duration_s'] == P.DURATION_S and plan['model'] == P.MODEL
        assert abs(plan['rates']['llm'] - P.LAMBDA0['llm'] * alpha) < 1e-9 and abs(plan['rates']['dit'] - P.LAMBDA0['dit'] * alpha) < 1e-9
        v = P.validate_plan(plan, tok, trace_rows)
        plans[alpha] = dict(alpha=alpha, plan_stem=P.plan_stem(alpha), path=str(p.relative_to(ROOT)), plan_sha256=P.sha256_file(p), rates=plan['rates'], **v)
        print(json.dumps(dict(alpha=alpha, **{k: v[k] for k in ('n_llm_measured', 'n_dit_measured', 'max_in_plus_out_actual', 'violations', 'ok')})))
    if not all(x['ok'] for x in plans.values()):
        raise SystemExit('plan validation failed')
    runs = []
    for i, alpha in enumerate(alphas):
        order = [c for c, a in jobs if a == alpha]; random.Random(P.SEED + 100 + i).shuffle(order)
        x = plans[alpha]
        for c in order:
            runs.append(dict(run_id=f"{c}__{x['plan_stem']}", candidate=c, alpha=alpha, plan_stem=x['plan_stem'], plan=x['path'],
                             plan_sha256=x['plan_sha256'], prompt_sha256=x['prompt_sha256'], replay='both',
                             expected_servers=P.expected_servers(c.split('_')[0]), cards=P.cards_of(c)))
    env = dict(P.FIXED_ENV, KV_BYTES=str(kv_bytes))
    pilot = json.loads((P.DEFAULT_OUT / 'protocol.json').read_text())
    proto = dict(stage=args.stage, purpose=args.purpose, parent_protocol=P.DEFAULT_OUT.relative_to(ROOT).as_posix(), parent_protocol_sha256=pilot['protocol_sha256'],
                 seed=P.SEED, candidates=sorted({c for c, _ in jobs}, key=P.CANDIDATES.index), alphas=alphas, lambda0=P.LAMBDA0,
                 warmup_s=P.WARMUP_S, duration_s=P.DURATION_S, drain_timeout_s=P.DRAIN_S, model=P.MODEL, dit_model=P.DMODEL,
                 dit=dict(size=512, steps=28, text_encoder_3=False), max_model_len=P.MAX_MODEL_LEN,
                 kv_bytes=kv_bytes, kv_tokens_expected=kv_tokens_expected(kv_bytes),
                 kv_note=f'{args.kv_gib:g} GiB fixed per vLLM engine (pilot: 4 GiB); PD candidates run two engines. Expected tokens = floor(bytes / {BLOCK_BYTES}) * 16.',
                 min_dit_measured=P.MIN_DIT_MEASURED, slo_grid=P.SLO_GRID, definitions=pilot['definitions'], price_model=P.PRICE_MODEL,
                 order_rule=f'alpha ascending; candidates shuffled within alpha with random.Random({P.SEED + 100}+i)',
                 env=env, env_stripped=list(P.STRIP_ENV), plans=[plans[a] for a in alphas], runs=runs,
                 harness=dict(P.harness_hashes(), run_stage1_refine_py=P.sha256_file(__file__), trace_sha256=P.sha256_file(trace)),
                 smoke_gate=dict(pilot['smoke_gate'], note='evaluated and passed before the pilot batch (results/hetero/stage1_pilot/smoke_gate.json); not re-run'))
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
            raise SystemExit(f'protocol.json already frozen with a different hash in {out}; refusing to overwrite')
        print(f'protocol unchanged {proto["protocol_sha256"][:12]}'); return old
    P.atomic_write(target, proto)
    print(f'protocol frozen {proto["protocol_sha256"][:12]} -> {target}  ({len(runs)} runs, KV {args.kv_gib:g} GiB = {proto["kv_tokens_expected"]} tokens)')
    return proto


def apply_env(out):
    proto = json.loads((Path(out) / 'protocol.json').read_text())
    P.FIXED_ENV = dict(P.FIXED_ENV, KV_BYTES=str(proto['kv_bytes']))
    return proto


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'dry-run', 'run'):
        p = sub.add_parser(name); p.add_argument('--out', type=Path, required=True)
        if name == 'prepare':
            p.add_argument('--runs', required=True, help='comma list of CANDIDATE:ALPHA')
            p.add_argument('--kv-gib', type=float, default=4.0)
            p.add_argument('--seed', type=int, default=901, help='plan seed (pilot 901; validation seeds 902+)')
            p.add_argument('--stage', default='stage1_refine'); p.add_argument('--purpose', default='alpha refinement between tested grid points (seed 901); exploratory, same rules as the pilot')
            p.add_argument('--plans-dir', type=Path, default=P.DEFAULT_PLANS); p.add_argument('--trace', type=Path, default=P.TRACE)
        if name == 'run':
            p.add_argument('--no-report', dest='report', action='store_false'); p.add_argument('--max-infra-failures', type=int, default=2)
            p.add_argument('--allow-harness-drift', action='store_true')
    args = ap.parse_args()
    if args.command == 'prepare':
        prepare(args)
    elif args.command == 'dry-run':
        apply_env(args.out); P.dry_run(SimpleNamespace(out=args.out, skip_smoke_gate=True))
    else:
        apply_env(args.out)
        P.run(SimpleNamespace(out=args.out, skip_smoke_gate=True, wait_smoke=False, report=args.report,
                              max_infra_failures=args.max_infra_failures, allow_harness_drift=args.allow_harness_drift))


if __name__ == '__main__':
    main()
