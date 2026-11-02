#!/usr/bin/env python3
"""Stage 1 pilot runner: fixed-cost capacity map, 7 candidates x alpha {0.5,1,1.5,2}, seed 901.

  prepare   generate alpha plans (seed 901), validate real token lengths, freeze protocol.json
  dry-run   print run order, exact commands and env; evaluate the smoke gate; no GPU
  gate      evaluate the Stage 0 kvfix smoke gate only
  run       execute the 28 mixed runs sequentially (global GPU lock, compute-apps check,
            per-run atomic status, incremental report). Designed for nohup.

Nothing here touches run_mixed_modal.sh / controlled_mixed.py; they are invoked as-is with an
explicitly controlled environment. Existing outputs are never deleted: a run with `done` is
skipped, an incomplete output directory blocks the batch. No alpha is skipped or extended
automatically; every (candidate, alpha) in the protocol is attempted exactly once.
"""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from stage1_report import strict_kv_check, strict_parity, expected_servers, build as build_report, validate_records, load_requests  # noqa: E402

DEFAULT_OUT = ROOT / 'results/hetero/stage1_pilot'
DEFAULT_PLANS = ROOT / 'results/hetero/stage1_plans'
SMOKE_DIR = ROOT / 'results/hetero/stage0_smoke_kvfix'
GLOBAL_LOCK = ROOT / 'results/hetero/gpu_batch.lock'

CANDIDATES = ['S4_4090', 'S3_3090', 'S4_3090', 'S3_4090', 'C_4090', 'C_3090', 'C_split']
ALPHAS = [0.5, 1.0, 1.5, 2.0]
SEED = 901
LAMBDA0 = dict(llm=2.0, dit=0.3)
WARMUP_S, DURATION_S, DRAIN_S = 60.0, 600.0, 300.0
MAX_MODEL_LEN = 8192
MIN_DIT_MEASURED = 60
KV_BYTES = 4294967296
KV_TOKENS_EXPECTED = 37440
MODEL = 'meta-llama/Llama-3.2-3B-Instruct'
DMODEL = 'stabilityai/stable-diffusion-3.5-medium'
TRACE = ROOT / 'bench/hetero/traces/AzureLLMInferenceTrace_code.csv'
# Environment handed to run_mixed_modal.sh: everything the script reads is set explicitly; any
# inherited value for these names is dropped first so a parent shell cannot leak REPLAY_ONLY etc.
FIXED_ENV = dict(MODEL=MODEL, DMODEL=DMODEL, GPU_UTIL='0.5', SIZE='512', STEPS='28', KV_BYTES=str(KV_BYTES), DRAIN_TIMEOUT=str(int(DRAIN_S)))
STRIP_ENV = ('REPLAY_ONLY', 'ONLY', 'CONTROLLED_PLAN', 'CONTROLLED_OUT', 'DRAIN_TIMEOUT', 'KV_BYTES', 'GPU_UTIL', 'SIZE', 'STEPS',
             'MODEL', 'DMODEL', 'N', 'LRATES', 'DRATES', 'WORK', 'TRACE', 'ND')
SLO_GRID = dict(ttft_s=[0.3, 0.5, 1.0, 2.0], tpot_s=[0.012, 0.02, 0.03, 0.05], dit_e2e_s=[3.0, 5.0, 10.0, 20.0], target=[0.9, 0.95, 0.99])
PRICE_MODEL = dict(ratio_3090_over_4090=[0.5, 0.75, 0.94, 1.0],
                   ratio_note='all ratios are research assumptions; 0.94 ≈ launch MSRP 1499/1599 amortisation example, others span a sensitivity range; not rental quotes',
                   energy_scenario=dict(electricity_usd_per_kwh=0.12, pue=1.3, amortisation_years=4, msrp_usd=dict(gpu4090=1599, gpu3090=1499), note='assumption, reported alongside measured Wh'),
                   idle_card_billing='reported both ways: occupied cards only, and both cards billed (1-GPU candidates then cost the same as 2-GPU ones)')
SMOKE_EXPECTED = ['S4_4090', 'S3_3090', 'S4_3090', 'S3_4090'] + [f'{c}{s}' for c in ('C_4090', 'C_3090', 'C_split') for s in ('', '_only_llm', '_only_dit')]
SMOKE_PLAN = 'smoke_seed900'
VENV_PY = Path.home() / 'venv/jinwoo/bin/python'


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def atomic_write(path, obj):
    path = Path(path); tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n'); os.replace(tmp, path)


def plan_stem(alpha):
    return f'alpha{alpha:g}_seed{SEED}'


# ----------------------------------------------------------------------------- prepare
def validate_plan(plan, tok, trace_rows):
    """Real-token check: regenerate the exact prompts replay will send (same function, same seed),
    count tokens, and require in+out <= MAX_MODEL_LEN and out >= 1. Reports clamps applied by
    controlled_mixed.prepare (ContextTokens < 8 -> 8, > 8000 -> 8000) so nothing is silently edited."""
    from loadgen import make_prompts_lens
    llm = [r for r in plan['requests'] if r['kind'] == 'llm']
    dit = [r for r in plan['requests'] if r['kind'] == 'dit']
    prompts = make_prompts_lens(tok, [(r['in_len'], r['out_len']) for r in llm], plan['seed'])
    actual = [(sin, r['out_len']) for r, (_, sin, _) in zip(llm, prompts)]
    violations = [(r['idx'], sin, out) for r, (sin, out) in zip(llm, actual) if sin + out > MAX_MODEL_LEN or out < 1 or sin < 1]
    full = []
    for r, (prompt, sin, _) in zip(llm, prompts):
        full.append(('llm', r['idx'], prompt))
    full += [('dit', r['idx'], r['prompt']) for r in dit]
    full.sort(key=lambda x: next(i for i, q in enumerate(plan['requests']) if q['kind'] == x[0] and q['idx'] == x[1]))
    prompt_sha = hashlib.sha256(json.dumps(full).encode()).hexdigest()
    off = plan['trace_offset']; clamped = 0
    for r in llm:
        row = trace_rows[(off + r['idx']) % len(trace_rows)]
        ctx = int(row['ContextTokens'])
        if ctx < 8 or ctx > 8000 or int(row['GeneratedTokens']) < 1:
            clamped += 1
    return dict(n_llm=len(llm), n_dit=len(dit), n_llm_measured=sum(r['measured'] for r in llm), n_dit_measured=sum(r['measured'] for r in dit),
                max_in_plus_out_actual=max((a + b for a, b in actual), default=0), max_in_actual=max((a for a, _ in actual), default=0),
                max_out=max((b for _, b in actual), default=0), n_actual_in_below_planned=sum(1 for r, (sin, _) in zip(llm, actual) if sin < r['in_len']),
                trace_rows_clamped_by_prepare=clamped, violations=violations, prompt_sha256=prompt_sha,
                ok=not violations and sum(r['measured'] for r in dit) >= MIN_DIT_MEASURED)


def prepare(args, tok=None):
    sys.path.insert(0, str(HERE))
    import controlled_mixed
    from types import SimpleNamespace
    plans_dir = Path(args.plans_dir); plans_dir.mkdir(parents=True, exist_ok=True)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    trace = Path(args.trace)
    with trace.open() as f:
        trace_rows = list(csv.DictReader(f))
    if tok is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    plans = []
    for alpha in ALPHAS:
        p = plans_dir / f'{plan_stem(alpha)}.json'
        if not p.exists():
            ns = SimpleNamespace(trace=trace, model=MODEL, seed=SEED, duration=DURATION_S, warmup=WARMUP_S,
                                 llm_rate=LAMBDA0['llm'] * alpha, dit_rate=LAMBDA0['dit'] * alpha, out=p)
            controlled_mixed.prepare(ns)
        plan = json.loads(p.read_text())
        assert plan['seed'] == SEED and plan['warmup_s'] == WARMUP_S and plan['duration_s'] == DURATION_S and plan['model'] == MODEL, f'existing plan {p} differs from protocol'
        assert abs(plan['rates']['llm'] - LAMBDA0['llm'] * alpha) < 1e-9 and abs(plan['rates']['dit'] - LAMBDA0['dit'] * alpha) < 1e-9, f'rates differ in {p}'
        v = validate_plan(plan, tok, trace_rows)
        plans.append(dict(alpha=alpha, plan_stem=plan_stem(alpha), path=str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),
                          plan_sha256=sha256_file(p), rates=plan['rates'], **v))
        print(json.dumps(dict(alpha=alpha, **{k: v[k] for k in ('n_llm_measured', 'n_dit_measured', 'max_in_plus_out_actual', 'trace_rows_clamped_by_prepare', 'violations', 'ok')})))
    if not all(x['ok'] for x in plans):
        raise SystemExit('plan validation failed; see violations above (fix by a new explicit plan rule, not by editing requests)')
    runs = []
    for i, x in enumerate(plans):
        order = list(CANDIDATES); random.Random(SEED + i).shuffle(order)
        for c in order:
            runs.append(dict(run_id=f"{c}__{x['plan_stem']}", candidate=c, alpha=x['alpha'], plan_stem=x['plan_stem'], plan=x['path'],
                             plan_sha256=x['plan_sha256'], prompt_sha256=x['prompt_sha256'], replay='both',
                             expected_servers=expected_servers(c.split('_')[0]), cards=cards_of(c)))
    proto = dict(stage='stage1_pilot', purpose='exploratory capacity map on one seed (not selector validation; no solo runs; no automatic alpha extension)',
                 seed=SEED, candidates=CANDIDATES, alphas=ALPHAS, lambda0=LAMBDA0, warmup_s=WARMUP_S, duration_s=DURATION_S, drain_timeout_s=DRAIN_S,
                 model=MODEL, dit_model=DMODEL, dit=dict(size=512, steps=28, text_encoder_3=False), max_model_len=MAX_MODEL_LEN,
                 kv_bytes=KV_BYTES, kv_tokens_expected=KV_TOKENS_EXPECTED,
                 kv_note='4 GiB fixed per vLLM engine: PD candidates (C_*) run two engines = 8 GiB total KV on two cards; single-engine candidates 4 GiB on one card. Physical footprint difference is part of the strategy and is reported in cost/memory.',
                 min_dit_measured=MIN_DIT_MEASURED, slo_grid=SLO_GRID,
                 definitions=dict(ttft='first content chunk - planned arrival (send lag included, recorded separately)',
                                  tpot='(end - first) / (completion_tokens - 1); undefined when completion_tokens == 1 (no inter-token gap): only TTFT applies, count reported',
                                  dit_e2e='image completion - planned arrival', denominator='all measured requests incl. errors, drain timeouts, unfinished',
                                  cell_pass='both services >= target attainment; alpha_max = largest tested passing alpha, full pass/fail list reported, non-monotone flagged',
                                  stability='3 x 200 s windows: planned arrivals, ok completions, failed ends, in-flight at window end; backlog at observation end; drain usage; timeouts kept as valid results separate from infra failures'),
                 price_model=PRICE_MODEL, order_rule='alpha ascending; candidates shuffled within alpha with random.Random(901+i)',
                 env=FIXED_ENV, env_stripped=list(STRIP_ENV), plans=plans, runs=runs,
                 harness=dict(harness_hashes(), trace_sha256=sha256_file(trace)),
                 smoke_gate=dict(dir=str(SMOKE_DIR.relative_to(ROOT)), expected_runs=SMOKE_EXPECTED, plan=SMOKE_PLAN, require='13/13 done, 0 request errors, strict KV per server, strict parity per replay kind'))
    # Frozen hash covers the experimental protocol; harness file hashes are recorded for provenance and
    # checked at run time (drift is reported and blocks unless --allow-harness-drift).
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
            raise SystemExit(f'protocol.json already frozen with a different hash ({old["protocol_sha256"][:12]} vs {proto["protocol_sha256"][:12]}); refusing to overwrite')
        print(f'protocol unchanged {proto["protocol_sha256"][:12]}'); return old
    atomic_write(target, proto)
    print(f'protocol frozen {proto["protocol_sha256"][:12]} -> {target}')
    return proto


def harness_hashes():
    return dict(run_mixed_modal_sh=sha256_file(HERE / 'run_mixed_modal.sh'), controlled_mixed_py=sha256_file(HERE / 'controlled_mixed.py'),
                run_stage1_pilot_py=sha256_file(__file__), stage1_report_py=sha256_file(HERE / 'stage1_report.py'),
                dit_server_py=sha256_file(HERE / 'dit_server.py'), toy_proxy_server_py=sha256_file(HERE / 'toy_proxy_server.py'), rr_proxy_server_py=sha256_file(HERE / 'rr_proxy_server.py'),
                loadgen_py=sha256_file(HERE / 'loadgen.py'))


def cards_of(candidate):
    from stage1_report import gpus_of
    return gpus_of(candidate)


# ----------------------------------------------------------------------------- smoke gate
def smoke_gate(smoke_dir=SMOKE_DIR):
    smoke_dir = Path(smoke_dir); problems = []
    status_f = smoke_dir / 'batch_status.json'
    status = json.loads(status_f.read_text()) if status_f.exists() else {}
    if status.get('status') != 'complete':
        problems.append(f"batch status {status.get('status')!r} (completed {status.get('completed_jobs')}/{status.get('total_jobs')})")
    if status.get('failures'):
        problems.append(f"batch failures {status['failures']}")
    groups = {}; kv = {}
    for name in SMOKE_EXPECTED:
        case = smoke_dir / name / SMOKE_PLAN
        manifest = case / 'replay/manifest.json'
        if not (case / 'done').exists() or not manifest.exists():
            problems.append(f'{name}: not done'); continue
        rep = json.loads(manifest.read_text())
        if rep['errors'] != 0:
            problems.append(f"{name}: {rep['errors']} request errors")
        llm = name.split('_')[0]
        k = strict_kv_check(case, llm, KV_TOKENS_EXPECTED, KV_BYTES); kv[name] = k
        if not k['ok']:
            problems.append(f"{name}: KV {k['reason']}")
        cols = {}
        for kind in ('llm', 'dit'):
            f = case / 'replay' / kind / 'requests.csv'
            if f.exists():
                with f.open() as fh:
                    cols[kind] = next(csv.reader(fh))
        groups.setdefault(rep['only'], []).append(dict(plan_sha256=rep['plan_sha256'], prompt_sha256=rep['prompt_sha256'], columns=cols,
                                                       denominators={k2: v['n'] for k2, v in rep['measured'].items()}))
    parity = strict_parity(groups)
    for only in ('both', 'llm', 'dit'):
        if only not in parity:
            problems.append(f'no runs for replay kind {only}')
        elif not parity[only]['ok']:
            problems.append(f'parity failed for {only}: {parity[only]["fields"]}')
    return dict(ok=not problems, problems=problems, parity=parity, kv={k: v['servers'] for k, v in kv.items()}, batch_status=status)


def smoke_running(smoke_dir=SMOKE_DIR):
    lock = Path(smoke_dir) / 'batch.lock'
    if not lock.exists():
        return False
    with lock.open('a') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB); fcntl.flock(f, fcntl.LOCK_UN); return False
        except BlockingIOError:
            return True


# ----------------------------------------------------------------------------- run
def compute_apps():
    r = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory', '--format=csv,noheader'], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f'nvidia-smi failed: {r.stderr.strip()}')
    return [line.strip() for line in r.stdout.splitlines() if line.strip()]


def run_env(plan_path, case):
    env = {k: v for k, v in os.environ.items() if k not in STRIP_ENV}
    env.update(FIXED_ENV); env.update(CONTROLLED_PLAN=str(plan_path), CONTROLLED_OUT=str(case))
    return env


def command_for(job):
    llm, dit = job['candidate'].split('_', 1)
    return ['bash', 'bench/hetero/run_mixed_modal.sh', llm, dit]


def wait_for_smoke(poll_s=30, max_wait_s=6 * 3600):
    deadline = time.monotonic() + max_wait_s
    while smoke_running() or json.loads((SMOKE_DIR / 'batch_status.json').read_text()).get('status') in ('running', 'waiting_for_first_run'):
        if time.monotonic() > deadline:
            raise RuntimeError('smoke wait timeout')
        time.sleep(poll_s)


def run(args):
    out = Path(args.out).resolve(); proto = json.loads((out / 'protocol.json').read_text())
    batch_lock = (out / 'runner.lock').open('a')
    try:
        fcntl.flock(batch_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('this output already has a live runner')
    (out / 'status').mkdir(exist_ok=True)
    batch = out / 'batch_status.json'
    def state(status, **extra):
        atomic_write(batch, dict(status=status, updated_unix_s=time.time(), protocol_sha256=proto['protocol_sha256'], total_jobs=len(proto['runs']), pid=os.getpid(), **extra))
    frozen_body = {k: v for k, v in proto.items() if k not in ('harness', 'protocol_sha256', 'created_unix_s', 'git_head')}
    if hashlib.sha256(json.dumps(frozen_body, sort_keys=True).encode()).hexdigest() != proto['protocol_sha256']:
        state('blocked', reason='protocol body hash mismatch'); raise SystemExit('protocol body hash mismatch')
    drift = {k: v for k, v in harness_hashes().items() if proto['harness'].get(k) != v}
    if drift and not args.allow_harness_drift:
        state('blocked', reason='harness files changed since protocol was frozen', drift=list(drift)); raise SystemExit(f'harness drift: {sorted(drift)} (re-freeze protocol or pass --allow-harness-drift)')
    if args.wait_smoke:
        state('waiting_for_smoke'); wait_for_smoke()
    if not args.skip_smoke_gate:
        gate = smoke_gate()
        atomic_write(out / 'smoke_gate.json', gate)
        if not gate['ok']:
            state('blocked', reason='smoke gate failed', problems=gate['problems']); raise SystemExit('smoke gate failed: ' + '; '.join(gate['problems']))
    GLOBAL_LOCK.parent.mkdir(parents=True, exist_ok=True)
    glock = GLOBAL_LOCK.open('a')
    try:
        fcntl.flock(glock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        state('blocked', reason=f'global GPU lock held: {GLOBAL_LOCK}'); raise SystemExit('another batch holds the global GPU lock')
    glock.seek(0); glock.truncate(); glock.write(f'{os.getpid()} stage1_pilot {time.time()}\n'); glock.flush()
    apps = compute_apps()
    if apps:
        state('blocked', reason='GPU compute processes present (not killed)', apps=apps); raise SystemExit('GPU busy: ' + '; '.join(apps))
    child = None
    def on_signal(signum, _frame):
        state('aborting', signal=signum)
        if child and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        raise SystemExit(128 + signum)
    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(s, on_signal)
    done = 0; infra_failures = []; results = []
    try:
        for job in proto['runs']:
            drift = [k for k, v in harness_hashes().items() if proto['harness'].get(k) != v]
            if drift and not args.allow_harness_drift:
                raise RuntimeError(f'harness changed before run: {drift}')
            case = out / job['candidate'] / job['plan_stem']
            plan_path = ROOT / job['plan']
            status_f = out / 'status' / f"{job['run_id']}.json"
            if sha256_file(plan_path) != job['plan_sha256']:
                raise RuntimeError(f'plan hash changed on disk: {plan_path}')
            if (case / 'done').exists():
                done += 1; results.append(dict(run_id=job['run_id'], outcome='skipped_existing_done')); continue
            if case.exists():
                atomic_write(status_f, dict(job, outcome='blocked_incomplete_output', note='existing output without done; inspect or move it manually'))
                state('blocked', reason=f'incomplete output exists: {case}'); raise SystemExit(f'incomplete output exists: {case}')
            cmd = command_for(job); env = run_env(plan_path, case)
            rec = dict(job, command=cmd, env_explicit={k: env[k] for k in list(FIXED_ENV) + ['CONTROLLED_PLAN', 'CONTROLLED_OUT']},
                       started_unix_s=time.time(), outcome='running', log=str(case.parent / f"{job['plan_stem']}.console.log"))
            atomic_write(status_f, rec); state('running', run_id=job['run_id'], completed_jobs=done, infra_failures=infra_failures)
            case.parent.mkdir(parents=True, exist_ok=True)
            with open(rec['log'], 'w') as log:
                child = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                code = child.wait()
            child = None
            (case.parent / f"{job['plan_stem']}.exit").write_text(f'{code}\n')
            rec.update(finished_unix_s=time.time(), exit_code=code)
            manifest = case / 'replay/manifest.json'
            if code == 0 and (case / 'done').exists() and manifest.exists():
                rep = json.loads(manifest.read_text())
                kv = strict_kv_check(case, job['candidate'].split('_')[0], proto['kv_tokens_expected'], proto['kv_bytes'])
                integrity = validate_records(load_requests(case), rep, json.loads(plan_path.read_text()))
                timeouts = 0
                for kind in ('llm', 'dit'):
                    f = case / 'replay' / kind / 'requests.csv'
                    if f.exists():
                        with f.open() as fh:
                            timeouts += sum(1 for r in csv.DictReader(fh) if r['err'] == 'drain_timeout')
                rec.update(outcome='done', errors=rep['errors'], drain_timeouts=timeouts, kv_ok=kv['ok'], kv_reason=kv['reason'],
                           records_ok=integrity['ok'], records_problems=integrity['problems'],
                           plan_match=rep['plan_sha256'] == job['plan_sha256'] and rep['prompt_sha256'] == job['prompt_sha256'])
                done += 1
            else:
                rec.update(outcome='failed_infra'); infra_failures.append(job['run_id'])
            # our servers must be gone before the next run; never kill foreign PIDs
            for _ in range(12):
                apps = compute_apps()
                if not apps:
                    break
                time.sleep(5)
            if apps:
                rec.update(leftover_gpu_processes=apps); atomic_write(status_f, rec)
                state('blocked', reason='GPU processes remain after run (not killed)', apps=apps, run_id=job['run_id']); raise SystemExit('leftover GPU processes: ' + '; '.join(apps))
            atomic_write(status_f, rec); results.append(dict(run_id=job['run_id'], outcome=rec['outcome']))
            if rec.get('outcome') == 'done' and (not rec.get('kv_ok') or not rec.get('plan_match') or not rec.get('records_ok')):
                state('blocked', reason='KV or workload integrity mismatch', run_id=job['run_id'], completed_jobs=done)
                raise SystemExit('KV or workload integrity mismatch; preserve output for inspection')
            if args.report:
                try:
                    build_report(out)
                except Exception as ex:  # report must never kill the batch
                    print(f'report failed: {ex}', file=sys.stderr)
            if len(infra_failures) >= args.max_infra_failures:
                state('aborted', reason=f'{len(infra_failures)} infra failures', infra_failures=infra_failures, completed_jobs=done); raise SystemExit('too many infra failures')
        state('complete', completed_jobs=done, infra_failures=infra_failures)
        if args.report:
            build_report(out)
    except SystemExit:
        raise
    except BaseException as ex:
        state('failed', error=str(ex), completed_jobs=done, infra_failures=infra_failures); raise
    finally:
        glock.close()


def dry_run(args):
    out = Path(args.out).resolve(); proto = json.loads((out / 'protocol.json').read_text())
    print(f"protocol {proto['protocol_sha256'][:12]}  runs {len(proto['runs'])}  out {out}")
    for i, job in enumerate(proto['runs']):
        case = out / job['candidate'] / job['plan_stem']
        env = run_env(ROOT / job['plan'], case)
        st = 'done' if (case / 'done').exists() else 'INCOMPLETE' if case.exists() else 'pending'
        print(f"{i+1:2d} {job['run_id']:28s} [{st}] cards={'+'.join(job['cards'])} servers={job['expected_servers']} "
              f"cmd={' '.join(command_for(job))} env={{{', '.join(f'{k}={env[k]}' for k in FIXED_ENV)}, CONTROLLED_PLAN={env['CONTROLLED_PLAN']}, CONTROLLED_OUT={env['CONTROLLED_OUT']}}}")
    drift = sorted(k for k, v in harness_hashes().items() if proto['harness'].get(k) != v)
    print('harness drift vs protocol:', drift or 'none')
    leaked = [k for k in STRIP_ENV if k in os.environ]
    print('inherited env that would be stripped:', leaked or 'none')
    if not args.skip_smoke_gate:
        gate = smoke_gate(); print('smoke gate:', json.dumps(dict(ok=gate['ok'], problems=gate['problems'], running=smoke_running()), ensure_ascii=False))
    try:
        print('GPU compute apps now:', compute_apps() or 'none')
    except (RuntimeError, FileNotFoundError) as ex:
        print('GPU compute apps: unavailable', ex)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'dry-run', 'gate', 'run'):
        p = sub.add_parser(name)
        p.add_argument('--out', type=Path, default=DEFAULT_OUT)
        if name == 'prepare':
            p.add_argument('--plans-dir', type=Path, default=DEFAULT_PLANS); p.add_argument('--trace', type=Path, default=TRACE)
        if name in ('dry-run', 'run'):
            p.add_argument('--skip-smoke-gate', action='store_true')
        if name == 'run':
            p.add_argument('--wait-smoke', action='store_true', help='block until the Stage 0 kvfix smoke batch has released its lock')
            p.add_argument('--no-report', dest='report', action='store_false'); p.add_argument('--max-infra-failures', type=int, default=2)
            p.add_argument('--allow-harness-drift', action='store_true', help='proceed although a harness file hash differs from protocol.json (drift is recorded)')
    args = ap.parse_args()
    if args.command == 'prepare':
        prepare(args)
    elif args.command == 'dry-run':
        dry_run(args)
    elif args.command == 'gate':
        g = smoke_gate(); print(json.dumps(g, indent=2, ensure_ascii=False)); sys.exit(0 if g['ok'] else 1)
    else:
        run(args)


if __name__ == '__main__':
    main()
