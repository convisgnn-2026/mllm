"""CPU checks for the Stage 1 pilot runner/report: synthetic requests only, no GPU, no servers.
  ~/venv/jinwoo/bin/python -m unittest bench/hetero/tests/test_stage1_pilot.py
"""
from contextlib import redirect_stdout
import csv
import datetime
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f'{name}.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


R = load('stage1_report')
P = load('run_stage1_pilot')
COLS = ['kind', 'idx', 'measured', 'planned_s', 'send_s', 'send_lag_s', 'first_s', 'end_s', 'ttft', 'tpot', 'e2e', 'queue', 'encode', 'denoise', 'vae', 'out_tokens', 'err', 'in_len', 'out_len']
GRID = dict(ttft_s=[0.5, 1.0], tpot_s=[0.02, 0.05], dit_e2e_s=[5.0, 10.0], target=[0.9, 0.99])


class FakeTok:
    vocab_size = 50000
    def decode(self, ids, skip_special_tokens=True): return ' '.join(map(str, ids))
    def __call__(self, text, add_special_tokens=False): return {'input_ids': [int(x) for x in text.split()] if text else []}


def rec(kind, idx, planned, first=None, end=None, n=None, err='', measured=True, out_len=None):
    return dict(kind=kind, idx=idx, measured=measured, planned_s=planned, send_s=planned + .001, send_lag_s=.001, first_s=first, end_s=end,
                ttft=None, tpot=None, e2e=None, queue=None, encode=None, denoise=None, vae=None, out_tokens=n, err=err, in_len=8, out_len=out_len if out_len is not None else n,
                columns=tuple(COLS))


def write_run(case, recs, candidate, kv_tokens=37440, kv_bytes=4294967296, servers=None, plan_sha='p', prompt_sha='q', done=True, start=1_700_000_000.):
    case.mkdir(parents=True, exist_ok=True)
    for kind in ('llm', 'dit'):
        rows = [r for r in recs if r['kind'] == kind]
        if rows:
            d = case / 'replay' / kind; d.mkdir(parents=True, exist_ok=True)
            with (d / 'requests.csv').open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=COLS); w.writeheader()
                for r in rows:
                    w.writerow({k: ('' if r.get(k) is None else r.get(k)) for k in COLS})
    (case / 'replay').mkdir(exist_ok=True)
    meas = {k: dict(n=sum(1 for r in recs if r['kind'] == k and r['measured'])) for k in ('llm', 'dit')}
    (case / 'replay/manifest.json').write_text(json.dumps(dict(plan_sha256=plan_sha, prompt_sha256=prompt_sha, only='both', start_unix_s=start, warmup_s=2., duration_s=6.,
                                                              drain_timeout_s=3., requests=len(recs), errors=sum(bool(r['err']) for r in recs), measured=meas)))
    servers = servers if servers is not None else {s: kv_tokens for s in R.expected_servers(candidate.split('_')[0])}
    (case / 'deployment.json').write_text(json.dumps(dict(kv_cache_memory_bytes=kv_bytes, kv_cache_tokens=servers)))
    (case / 'mem_after_load.txt').write_text('0, 100 MiB\n1, 200 MiB\n')
    with (case / 'gpu.csv').open('w') as f:
        for s in range(10):
            ts = datetime.datetime.fromtimestamp(start + s).strftime('%Y/%m/%d %H:%M:%S.000')
            f.write(f'{ts}, 0, 50 %, 1000 MiB, 360.00 W, 1 MHz, 1 MHz\n{ts}, 1, 0 %, 10 MiB, 36.00 W, 1 MHz, 1 MHz\n')
    if done:
        (case / 'done').write_text('x\n')


def good_recs(n_llm=20, n_dit=10, warm=2., slow_llm=0, slow_dit=0, timeouts=0, errors=0, n1=0):
    recs = [rec('llm', 0, .5, .6, .7, 3, measured=False), rec('dit', 0, .5, None, 1.5, None, measured=False)]
    for i in range(n_llm):
        t = warm + i * .25; n = 1 if i >= n_llm - n1 else 11
        if i < slow_llm: recs.append(rec('llm', i + 1, t, t + 1.5, t + 1.5 + 10 * .03, n))
        elif i < slow_llm + timeouts: recs.append(rec('llm', i + 1, t, None, t + 3., None, err='drain_timeout'))
        elif i < slow_llm + timeouts + errors: recs.append(rec('llm', i + 1, t, None, t + .1, None, err='boom'))
        else: recs.append(rec('llm', i + 1, t, t + .2, t + .2 + (n - 1) * .01, n))
    for i in range(n_dit):
        t = warm + i * .5
        recs.append(rec('dit', i + 1, t, None, t + (8. if i < slow_dit else 2.), None))
    return recs


class ScoreTests(unittest.TestCase):
    def test_joint_attainment_denominator_and_n1(self):
        recs = good_recs(n_llm=20, timeouts=2, errors=1, n1=3)
        s = R.score_run(recs, GRID, 2., 6.)
        llm = s['services']['llm']
        self.assertEqual((llm['n'], llm['ok'], llm['error'], llm['drain_timeout'], llm['n_tokens_eq_1']), (20, 17, 1, 2, 3))
        self.assertAlmostEqual(s['attainment_llm']['0.5|0.02'], 17 / 20)       # failures stay in the denominator
        self.assertAlmostEqual(s['attainment_llm']['0.5|0.05'], 17 / 20)
        self.assertEqual(s['attainment_dit']['5.0'], 1.0)
        self.assertFalse(s['cells']['0.5|0.02|5.0|0.9'])                       # 0.85 < 0.9
        self.assertTrue(R.score_run(good_recs(), GRID, 2., 6.)['cells']['0.5|0.02|5.0|0.99'])
        # TPOT undefined for n=1 -> only TTFT applies; tight TPOT bound still passes those
        tight = dict(GRID, tpot_s=[0.001]); s2 = R.score_run(good_recs(n_llm=10, n1=10), tight, 2., 6.)
        self.assertEqual(s2['attainment_llm']['0.5|0.001'], 1.0)
        self.assertIsNone(s2['services']['llm']['success_only']['tpot_p90'])
        # a service with slow requests fails its bound while the other passes -> joint fails
        s3 = R.score_run(good_recs(slow_dit=3), GRID, 2., 6.)
        self.assertEqual(s3['attainment_dit']['5.0'], .7); self.assertFalse(s3['cells']['0.5|0.02|5.0|0.9']); self.assertTrue(s3['cells']['0.5|0.02|10.0|0.9'])

    def test_windows_and_backlog(self):
        recs = good_recs(n_llm=20, timeouts=2) + [rec('llm', 99, 7.9, 8.1, 8.6, 11)]      # one completion lands in the drain
        w = R.score_run(recs, GRID, 2., 6.)['windows']['llm']
        self.assertEqual(sum(x['planned_arrivals'] for x in w['windows']), 21)
        self.assertEqual(w['drain_timeouts'], 2)
        self.assertGreaterEqual(w['windows'][0]['inflight_at_end'], 0)
        self.assertGreaterEqual(w['completed_during_drain'], 1)

    def test_replay_integrity_rejects_lost_rows_and_short_outputs(self):
        recs = good_recs()
        plan = dict(warmup_s=2., duration_s=6., requests=[
            dict(kind=r['kind'], idx=r['idx'], at_s=r['planned_s'], measured=r['measured'], out_len=r['out_len']) for r in recs])
        manifest = dict(requests=len(recs), warmup_s=2., duration_s=6., measured={
            k: dict(n=sum(r['kind'] == k and r['measured'] for r in recs)) for k in ('llm', 'dit')})
        self.assertTrue(R.validate_records(recs, manifest, plan)['ok'])
        self.assertFalse(R.validate_records(recs[:-1], dict(manifest, requests=len(recs)-1), plan)['ok'])
        self.assertFalse(R.validate_records(recs+[recs[0]], dict(manifest, requests=len(recs)+1), plan)['ok'])
        changed = [dict(r) for r in recs]; changed[2]['out_tokens'] -= 1
        self.assertFalse(R.validate_records(changed, manifest, plan)['ok'])

    def test_strict_kv_gate(self):
        with tempfile.TemporaryDirectory() as t:
            case = Path(t) / 'c'
            write_run(case, good_recs(), 'C_4090')
            self.assertTrue(R.strict_kv_check(case, 'C', 37440, 4294967296)['ok'])
            self.assertFalse(R.strict_kv_check(case, 'C', 37440, 1)['ok'])
            write_run(case, good_recs(), 'C_4090', servers={'serve_8100': 37440})          # second engine missing
            self.assertIn('serve_8200', R.strict_kv_check(case, 'C', 37440, 4294967296)['reason'])
            write_run(case, good_recs(), 'C_4090', servers={'serve_8100': 37440, 'serve_8200': None})   # unparsed
            self.assertFalse(R.strict_kv_check(case, 'C', 37440, 4294967296)['ok'])
            write_run(case, good_recs(), 'S4_4090', servers={})
            self.assertFalse(R.strict_kv_check(case, 'S4', 37440, 4294967296)['ok'])       # empty is not a pass
            (case / 'deployment.json').unlink()
            self.assertFalse(R.strict_kv_check(case, 'S4', 37440, 4294967296)['ok'])

    def test_strict_parity_rejects_empty(self):
        g = {'both': [dict(plan_sha256='a', prompt_sha256='b', columns=['x'], denominators={'llm': 1})] * 2}
        self.assertTrue(R.strict_parity(g)['both']['ok'])
        g = {'both': [dict(plan_sha256=None, prompt_sha256='b', columns=['x'], denominators={'llm': 1})] * 2}
        self.assertFalse(R.strict_parity(g)['both']['ok'])
        g = {'both': [dict(plan_sha256='a', prompt_sha256='b', columns=['x'], denominators={})]}
        self.assertFalse(R.strict_parity(g)['both']['ok'])


class ProtocolAndReportTests(unittest.TestCase):
    def make_protocol(self, t):
        out = Path(t) / 'out'; plans = Path(t) / 'plans'
        trace = Path(t) / 'trace.csv'
        trace.write_text('TIMESTAMP,ContextTokens,GeneratedTokens\n' + '\n'.join(f'x,{c},{g}' for c, g in [(4, 10), (512, 64), (7000, 1000), (3000, 1899)] * 50) + '\n')
        args = SimpleNamespace(plans_dir=plans, out=out, trace=trace)
        buf = io.StringIO()
        with redirect_stdout(buf):
            proto = P.prepare(args, tok=FakeTok())
        return out, plans, proto, buf.getvalue()

    def test_prepare_validates_tokens_and_freezes(self):
        with tempfile.TemporaryDirectory() as t:
            out, plans, proto, text = self.make_protocol(t)
            self.assertEqual(len(proto['runs']), 28)
            self.assertEqual(len(plans.glob and list(plans.glob('alpha*_seed901.json'))), 4)
            for p in proto['plans']:
                self.assertTrue(p['ok']); self.assertLessEqual(p['max_in_plus_out_actual'], 8192); self.assertGreaterEqual(p['n_dit_measured'], 60)
                self.assertGreater(p['trace_rows_clamped_by_prepare'], 0)                  # ctx=4 rows clamped to 8 by prepare, reported
            self.assertEqual({r['candidate'] for r in proto['runs']}, set(P.CANDIDATES))
            alphas = [r['alpha'] for r in proto['runs']]
            self.assertEqual(alphas, sorted(alphas))                                        # alpha ascending, candidates shuffled within
            # re-prepare is idempotent; a changed protocol constant is refused
            with redirect_stdout(io.StringIO()):
                again = P.prepare(SimpleNamespace(plans_dir=plans, out=out, trace=Path(t) / 'trace.csv'), tok=FakeTok())
            self.assertEqual(again['protocol_sha256'], proto['protocol_sha256'])
            old = P.KV_TOKENS_EXPECTED; P.KV_TOKENS_EXPECTED = 1
            try:
                with self.assertRaises(SystemExit), redirect_stdout(io.StringIO()):
                    P.prepare(SimpleNamespace(plans_dir=plans, out=out, trace=Path(t) / 'trace.csv'), tok=FakeTok())
            finally:
                P.KV_TOKENS_EXPECTED = old

    def test_prepare_refuses_overlong_requests(self):
        with tempfile.TemporaryDirectory() as t:
            trace = Path(t) / 'trace.csv'; trace.write_text('TIMESTAMP,ContextTokens,GeneratedTokens\nx,7000,1500\n')
            with self.assertRaises(SystemExit), redirect_stdout(io.StringIO()):
                P.prepare(SimpleNamespace(plans_dir=Path(t) / 'plans', out=Path(t) / 'out', trace=trace), tok=FakeTok())

    def test_dry_run_command_and_env_isolation(self):
        with tempfile.TemporaryDirectory() as t:
            out, plans, proto, _ = self.make_protocol(t)
            env = dict(os.environ, REPLAY_ONLY='llm', CONTROLLED_PLAN='/nope', DRAIN_TIMEOUT='5', KV_BYTES='0', PATH=os.environ.get('PATH', ''))
            r = subprocess.run([sys.executable, str(HERE / 'run_stage1_pilot.py'), 'dry-run', '--out', str(out), '--skip-smoke-gate'], capture_output=True, text=True, env=env)
            self.assertEqual(r.returncode, 0, r.stderr)
            lines = [l for l in r.stdout.splitlines() if 'cmd=bash bench/hetero/run_mixed_modal.sh' in l]
            self.assertEqual(len(lines), 28)
            self.assertTrue(all('KV_BYTES=4294967296' in l and 'DRAIN_TIMEOUT=300' in l and 'GPU_UTIL=0.5' in l for l in lines))
            self.assertIn("REPLAY_ONLY", r.stdout.split('inherited env that would be stripped:')[1])
            e = P.run_env(Path('/p'), Path('/c'))
            self.assertNotIn('REPLAY_ONLY', e); self.assertEqual(e['KV_BYTES'], '4294967296'); self.assertEqual(e['CONTROLLED_PLAN'], '/p')
            self.assertEqual(P.command_for(proto['runs'][0])[:2], ['bash', 'bench/hetero/run_mixed_modal.sh'])

    def test_report_alpha_max_nonmonotone_untested_and_kv(self):
        with tempfile.TemporaryDirectory() as t:
            out, plans, proto, _ = self.make_protocol(t)
            # Match the synthetic replay to synthetic plans, preserving the production identity check.
            from hashlib import sha256
            for plan_info in proto['plans']:
                path = Path(plan_info['path'])
                plan = json.loads(path.read_text())
                plan.update(warmup_s=2., duration_s=6., requests=[
                    dict(kind=r['kind'], idx=r['idx'], at_s=r['planned_s'], measured=r['measured'], out_len=r['out_len'])
                    for r in good_recs()])
                path.write_text(json.dumps(plan))
                for job in proto['runs']:
                    if job['plan_stem'] == plan_info['plan_stem']:
                        job['plan_sha256'] = sha256(path.read_bytes()).hexdigest()
            (out / 'protocol.json').write_text(json.dumps(proto))
            jobs = {(r['candidate'], r['alpha']): r for r in proto['runs']}
            def put(cand, alpha, recs, **kw):
                j = jobs[(cand, alpha)]
                write_run(out / cand / j['plan_stem'], recs, cand, plan_sha=kw.pop('plan_sha', j['plan_sha256']), prompt_sha=kw.pop('prompt_sha', j['prompt_sha256']), **kw)
            put('S4_4090', 0.5, good_recs()); put('S4_4090', 1.0, good_recs(slow_llm=5)); put('S4_4090', 1.5, good_recs()); put('S4_4090', 2.0, good_recs(slow_llm=10))
            put('S3_3090', 0.5, good_recs()); put('S3_3090', 1.0, good_recs()); put('S3_3090', 1.5, good_recs()); put('S3_3090', 2.0, good_recs())
            put('C_4090', 0.5, good_recs(), servers={'serve_8100': 37440})                 # KV missing -> invalid, never a pass
            put('S4_3090', 0.5, good_recs(), plan_sha='wrong')                              # plan mismatch -> invalid
            put('C_split', 0.5, good_recs(), done=False)                                    # incomplete
            with redirect_stdout(io.StringIO()):
                S = R.build(out)
            cap = S['capacity']
            self.assertEqual(len(cap['S4_4090']), 192)                                      # full protocol grid, not the test grid
            key = '0.5|0.02|5.0|0.9'
            x = cap['S4_4090'][key]
            self.assertEqual(x['verdicts'], {'0.5': True, '1.0': False, '1.5': True, '2.0': False})
            self.assertEqual(x['alpha_max_tested'], 1.5); self.assertTrue(x['non_monotone']); self.assertEqual(x['bound'], 'observed')
            y = cap['S3_3090'][key]; self.assertEqual((y['alpha_max_tested'], y['bound']), (2.0, 'observed')); self.assertTrue(y['upper_unmeasured'])
            z = cap['C_4090'][key]; self.assertIsNone(z['alpha_max_tested']); self.assertEqual(z['untested'], [0.5, 1.0, 1.5, 2.0])
            self.assertFalse(next(r for r in S['runs'] if r['run_id'] == 'C_4090__alpha0.5_seed901')['valid'])
            self.assertFalse(next(r for r in S['runs'] if r['run_id'] == 'S4_3090__alpha0.5_seed901')['valid'])
            self.assertEqual(next(r for r in S['runs'] if r['run_id'] == 'C_split__alpha0.5_seed901')['state'], 'incomplete')
            self.assertEqual(S['cost']['S4_4090']['by_ratio']['0.5'], dict(occupied_only=1.0, idle_billed=1.5))
            self.assertEqual(S['cost']['C_split']['by_ratio']['0.94']['occupied_only'], 1.94)
            e = next(r for r in S['runs'] if r['run_id'] == 'S3_3090__alpha0.5_seed901')['resources']['energy']
            self.assertAlmostEqual(e['0']['observation_wh'], 360 * 6 / 3600, places=3)
            self.assertTrue((out / 'stage1_summary.md').exists() and (out / 'stage1_cells.csv').exists())
            md = (out / 'stage1_summary.md').read_text(); self.assertIn('NM', md); self.assertIn('upper-unmeasured', md)

    def test_smoke_gate_requires_all_13_and_kv(self):
        with tempfile.TemporaryDirectory() as t:
            smoke = Path(t) / 'smoke'
            for name in P.SMOKE_EXPECTED:
                only = 'llm' if name.endswith('_only_llm') else 'dit' if name.endswith('_only_dit') else 'both'
                recs = [r for r in good_recs() if only == 'both' or r['kind'] == only]
                write_run(smoke / name / P.SMOKE_PLAN, recs, name.split('_only')[0])
                m = smoke / name / P.SMOKE_PLAN / 'replay/manifest.json'; d = json.loads(m.read_text()); d['only'] = only; m.write_text(json.dumps(d))
            (smoke / 'batch_status.json').write_text(json.dumps(dict(status='complete', completed_jobs=13, failures=[])))
            self.assertTrue(P.smoke_gate(smoke)['ok'], P.smoke_gate(smoke)['problems'])
            (smoke / 'batch_status.json').write_text(json.dumps(dict(status='running', completed_jobs=12)))
            self.assertFalse(P.smoke_gate(smoke)['ok'])
            (smoke / 'batch_status.json').write_text(json.dumps(dict(status='complete', completed_jobs=13, failures=[])))
            dep = smoke / 'C_split_only_dit' / P.SMOKE_PLAN / 'deployment.json'
            dep.write_text(json.dumps(dict(kv_cache_memory_bytes=4294967296, kv_cache_tokens={'serve_8100': 37440, 'serve_8200': None})))
            g = P.smoke_gate(smoke); self.assertFalse(g['ok']); self.assertTrue(any('serve_8200' in p for p in g['problems']))
            dep.write_text(json.dumps(dict(kv_cache_memory_bytes=4294967296, kv_cache_tokens={'serve_8100': 37440, 'serve_8200': 37440})))
            (smoke / 'S4_4090' / P.SMOKE_PLAN / 'done').unlink()
            self.assertFalse(P.smoke_gate(smoke)['ok'])


if __name__ == '__main__':
    unittest.main()
