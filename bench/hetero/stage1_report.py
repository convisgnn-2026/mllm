#!/usr/bin/env python3
"""Stage 1 pilot: incremental analysis of the fixed-cost capacity map (joint SLO attainment,
alpha_max per SLO cell, stability windows, GPU-time/energy, assumed price ratios).

  python bench/hetero/stage1_report.py results/hetero/stage1_pilot

Reads <out>/protocol.json (frozen before the batch) and every run listed there. Runs that are
missing, incomplete or failed are reported as *untested* cells, never as passes. Definitions:
  TTFT  = first content chunk time - planned arrival (client send lag included, kept separately).
  TPOT  = (end - first) / (completion_tokens - 1); undefined for completion_tokens == 1 (no
          inter-token gap) -> only the TTFT bound applies to that request, count reported.
  LLM request attains a cell iff no error, TTFT <= T and (TPOT <= P or TPOT undefined).
  DiT request attains iff no error and E2E (end - planned arrival) <= D.
  Attainment denominator = all measured requests of the service, including errors, drain
  timeouts and unfinished ones. Cell passes iff BOTH services reach the target fraction.
  alpha_max(cell) = largest *tested* alpha that passed; pass/fail at every alpha is listed, and
  non-monotone patterns (fail at lower alpha, pass at higher) are flagged, not smoothed.
Success-only p50/p90 are reported separately and are not the attainment statistic.
"""
import csv
import datetime
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


# ----------------------------------------------------------------------------- strict gates
def expected_servers(llm):
    return ['serve_8100', 'serve_8200'] if llm == 'C' else ['serve_8100']


def strict_kv_check(case, llm, expected_tokens, expected_bytes):
    """Every required LLM server must report exactly `expected_tokens` KV tokens, and the
    deployment must record the fixed byte budget. A missing/unparsed value is a failure."""
    dep_path = Path(case) / 'deployment.json'
    if not dep_path.exists():
        return dict(ok=False, reason='deployment.json missing', servers={})
    dep = json.loads(dep_path.read_text())
    servers = dep.get('kv_cache_tokens') or {}
    problems = []
    if dep.get('kv_cache_memory_bytes') != expected_bytes:
        problems.append(f"kv_cache_memory_bytes={dep.get('kv_cache_memory_bytes')} != {expected_bytes}")
    for name in expected_servers(llm):
        value = servers.get(name)
        if value is None:
            problems.append(f'{name}: KV tokens missing/unparsed')
        elif value != expected_tokens:
            problems.append(f'{name}: {value} != {expected_tokens}')
    extra = sorted(set(servers) - set(expected_servers(llm)))
    if extra:
        problems.append(f'unexpected servers {extra}')
    return dict(ok=not problems, reason='; '.join(problems), servers=servers,
                kv_total_gib=round(expected_bytes * len(expected_servers(llm)) / 2**30, 2))


def strict_parity(groups):
    """groups: {key: [dict(plan_sha256, prompt_sha256, columns, denominators)]}. Each field must
    have exactly one distinct, non-empty value per group."""
    result = {}
    for key, group in groups.items():
        fields = {}
        for field in ('plan_sha256', 'prompt_sha256', 'columns', 'denominators'):
            values = {json.dumps(r.get(field), sort_keys=True) for r in group}
            fields[field] = len(values) == 1 and next(iter(values)) not in ('null', '""', '{}', '[]')
        result[key] = dict(n=len(group), ok=all(fields.values()), fields=fields)
    return result


# ----------------------------------------------------------------------------- loading
def fnum(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def load_requests(case):
    recs = []
    for kind in ('llm', 'dit'):
        f = Path(case) / 'replay' / kind / 'requests.csv'
        if not f.exists():
            continue
        with f.open() as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                recs.append(dict(kind=row['kind'], idx=int(row['idx']), measured=row['measured'] == 'True',
                                 planned_s=fnum(row['planned_s']), send_s=fnum(row['send_s']), send_lag_s=fnum(row['send_lag_s']),
                                 first_s=fnum(row['first_s']), end_s=fnum(row['end_s']), out_tokens=fnum(row['out_tokens']),
                                 out_len=fnum(row['out_len']), err=row['err'] or '', columns=tuple(reader.fieldnames)))
    return recs


def classify(rec):
    if rec['err'] == 'drain_timeout':
        return 'drain_timeout'
    if rec['err']:
        return 'error'
    if rec['end_s'] is None or (rec['kind'] == 'llm' and rec['first_s'] is None):
        return 'unfinished'
    return 'ok'


def validate_records(recs, manifest, plan):
    """Reject lost/duplicated rows and altered workload rather than rewarding a smaller denominator."""
    problems = []
    expected = {(r['kind'], r['idx']): r for r in plan['requests']}
    keys = [(r['kind'], r['idx']) for r in recs]
    if len(keys) != len(set(keys)) or set(keys) != set(expected):
        problems.append('request identities missing, unexpected or duplicated')
    if len(recs) != manifest.get('requests'):
        problems.append('CSV count differs from manifest')
    if manifest.get('warmup_s') != plan['warmup_s'] or manifest.get('duration_s') != plan['duration_s']:
        problems.append('observation window differs from plan')
    for kind in ('llm', 'dit'):
        count = sum(r['kind'] == kind and r['measured'] for r in recs)
        expected_n = sum(r['kind'] == kind and r['measured'] for r in plan['requests'])
        if count != expected_n or manifest.get('measured', {}).get(kind, {}).get('n') != expected_n:
            problems.append(f'{kind} denominator differs from plan')
    for r in recs:
        p = expected.get((r['kind'], r['idx']))
        if p is None:
            continue
        if r['planned_s'] != p['at_s'] or r['measured'] != p['measured']:
            problems.append('request arrival/measurement differs from plan'); break
        if r['kind'] == 'llm' and classify(r) == 'ok':
            n = r['out_tokens']
            if n is None or n != p['out_len'] or r['out_len'] != p['out_len']:
                problems.append('successful LLM output length differs from plan'); break
    return dict(ok=not problems, problems=problems)


def derive(rec):
    """Return (status, ttft, tpot, e2e, n_tokens). tpot None when undefined (n==1) or failed."""
    status = classify(rec)
    if status != 'ok':
        return status, None, None, None, None
    e2e = rec['end_s'] - rec['planned_s']
    if rec['kind'] == 'dit':
        return status, None, None, e2e, None
    n = int(rec['out_tokens'] or 0)
    ttft = rec['first_s'] - rec['planned_s']
    tpot = (rec['end_s'] - rec['first_s']) / (n - 1) if n >= 2 else None
    return status, ttft, tpot, e2e, n


def pct(values, q):
    v = sorted(x for x in values if x is not None)
    if not v:
        return None
    return v[min(len(v) - 1, int(math.ceil(q * len(v)) - 1))] if q < 1 else v[-1]


# ----------------------------------------------------------------------------- scoring
def score_run(recs, grid, warmup, duration):
    meas = [r for r in recs if r['measured']]
    per = {}
    for kind in ('llm', 'dit'):
        rows = [r for r in meas if r['kind'] == kind]
        derived = [derive(r) for r in rows]
        statuses = {s: sum(d[0] == s for d in derived) for s in ('ok', 'error', 'drain_timeout', 'unfinished')}
        ok = [d for d in derived if d[0] == 'ok']
        n1 = sum(1 for d in ok if d[4] == 1)
        mismatch = sum(1 for r, d in zip(rows, derived) if d[0] == 'ok' and kind == 'llm' and r['out_len'] is not None and d[4] != int(r['out_len']))
        per[kind] = dict(n=len(rows), **statuses, n_tokens_eq_1=n1 if kind == 'llm' else None,
                         out_tokens_mismatch=mismatch if kind == 'llm' else None,
                         send_lag_p99_s=pct([r['send_lag_s'] for r in rows], .99),
                         success_only=dict(ttft_p50=pct([d[1] for d in ok], .5), ttft_p90=pct([d[1] for d in ok], .9),
                                           tpot_p50=pct([d[2] for d in ok], .5), tpot_p90=pct([d[2] for d in ok], .9),
                                           e2e_p50=pct([d[3] for d in ok], .5), e2e_p90=pct([d[3] for d in ok], .9)),
                         _derived=derived)
    # attainment vectors
    att_llm = {}
    for T in grid['ttft_s']:
        for P in grid['tpot_s']:
            d = per['llm']['_derived']
            passed = sum(1 for s, ttft, tpot, _, _ in d if s == 'ok' and ttft <= T and (tpot is None or tpot <= P))
            att_llm[f'{T}|{P}'] = passed / len(d) if d else None
    att_dit = {}
    for D in grid['dit_e2e_s']:
        d = per['dit']['_derived']
        att_dit[str(D)] = sum(1 for s, _, _, e2e, _ in d if s == 'ok' and e2e <= D) / len(d) if d else None
    cells = {}
    for T in grid['ttft_s']:
        for P in grid['tpot_s']:
            for D in grid['dit_e2e_s']:
                for g in grid['target']:
                    a, b = att_llm[f'{T}|{P}'], att_dit[str(D)]
                    cells[f'{T}|{P}|{D}|{g}'] = None if a is None or b is None else bool(a >= g and b >= g)
    for kind in per:
        per[kind].pop('_derived')
    return dict(services=per, attainment_llm=att_llm, attainment_dit=att_dit, cells=cells,
                windows=windows(recs, warmup, duration))


def windows(recs, warmup, duration, n_windows=3):
    """Per 200 s window: planned (measured) arrivals, successful completions whose end falls in the
    window, errors/timeouts ending in the window, and in-flight at the window end (sent - ended,
    all requests of the kind incl. warmup). Also backlog at observation end and after drain."""
    w = duration / n_windows
    stop = warmup + duration
    out = {}
    for kind in ('llm', 'dit'):
        rows = [r for r in recs if r['kind'] == kind]
        wins = []
        for k in range(n_windows):
            lo, hi = warmup + k * w, warmup + (k + 1) * w
            arrived = sum(1 for r in rows if r['measured'] and lo <= r['planned_s'] < hi)
            done_ok = sum(1 for r in rows if classify(r) == 'ok' and lo <= r['end_s'] < hi)
            done_bad = sum(1 for r in rows if classify(r) != 'ok' and r['end_s'] is not None and lo <= r['end_s'] < hi)
            inflight = sum(1 for r in rows if r['send_s'] is not None and r['send_s'] <= hi) - sum(1 for r in rows if r['end_s'] is not None and r['end_s'] <= hi)
            wins.append(dict(window=k, start_s=lo, end_s=hi, planned_arrivals=arrived, completed_ok=done_ok, ended_failed=done_bad, inflight_at_end=inflight))
        last_end = max((r['end_s'] for r in rows if r['end_s'] is not None), default=stop)
        completed_in_drain = sum(1 for r in rows if classify(r) == 'ok' and r['end_s'] >= stop)
        out[kind] = dict(windows=wins, backlog_at_observation_end=wins[-1]['inflight_at_end'] if wins else None,
                         completed_during_drain=completed_in_drain, drain_timeouts=sum(1 for r in rows if r['err'] == 'drain_timeout'),
                         drain_used_s=max(0., last_end - stop),
                         saturation_suspect=bool(wins and wins[-1]['inflight_at_end'] > wins[0]['inflight_at_end'] and any(r['err'] == 'drain_timeout' for r in rows)))
    return out


# ----------------------------------------------------------------------------- resources
def parse_gpu_csv(case):
    rows = []
    f = Path(case) / 'gpu.csv'
    if not f.exists():
        return rows
    with f.open() as fh:
        for row in csv.reader(fh):
            try:
                ts = datetime.datetime.strptime(row[0].strip(), '%Y/%m/%d %H:%M:%S.%f').timestamp()
                rows.append((ts, int(row[1]), float(row[4].split()[0]), int(row[3].split()[0])))
            except (ValueError, IndexError):
                continue
    return rows


def energy(case, start_unix, warmup, duration):
    """Wh per GPU index over the observation window and over the whole replay (load tail + drain)."""
    rows = parse_gpu_csv(case)
    out = {}
    for idx in sorted({r[1] for r in rows}):
        series = sorted((t, p, m) for t, i, p, m in rows if i == idx)
        def integrate(lo, hi):
            wh = 0.
            for (t0, p0, _), (t1, _, _) in zip(series, series[1:]):
                a, b = max(t0, lo), min(t1, hi)
                if b > a:
                    wh += p0 * (b - a) / 3600
            return wh
        obs = integrate(start_unix + warmup, start_unix + warmup + duration)
        full = integrate(series[0][0], series[-1][0]) if series else 0.
        out[str(idx)] = dict(observation_wh=round(obs, 2), full_replay_wh=round(full, 2), peak_mem_mib=max((m for _, _, m in series), default=None),
                             span_s=round(series[-1][0] - series[0][0], 1) if series else None, samples=len(series))
    return out


def gpus_of(candidate):
    """Physically occupied cards (both models resident). PD/split candidates occupy both cards."""
    llm, dit = candidate.split('_', 1)
    cards = set()
    cards |= {'4090'} if llm == 'S4' else {'3090'} if llm == 'S3' else {'4090', '3090'}
    cards |= {'4090'} if dit == '4090' else {'3090'} if dit == '3090' else {'4090', '3090'}
    return sorted(cards)


def cost_rate(cards, r, idle_billed):
    n4, n3 = ('4090' in cards), ('3090' in cards)
    if idle_billed:
        n4, n3 = 1, 1
    return n4 + r * n3  # units: p_4090 per hour (p_4090 normalised to 1)


# ----------------------------------------------------------------------------- report
def build(out_dir):
    out_dir = Path(out_dir)
    proto = json.loads((out_dir / 'protocol.json').read_text())
    grid, price = proto['slo_grid'], proto['price_model']
    runs = []
    groups = {}
    for job in proto['runs']:
        case = out_dir / job['candidate'] / job['plan_stem']
        entry = dict(job, case=str(case.relative_to(ROOT)) if case.is_relative_to(ROOT) else str(case), state='untested')
        status_file = out_dir / 'status' / f"{job['run_id']}.json"
        if status_file.exists():
            entry['runner_status'] = json.loads(status_file.read_text())
        manifest = case / 'replay' / 'manifest.json'
        if not (case / 'done').exists() or not manifest.exists():
            entry['state'] = 'incomplete' if case.exists() else 'untested'
            runs.append(entry); continue
        rep = json.loads(manifest.read_text())
        recs = load_requests(case)
        plan = json.loads((ROOT / job['plan']).read_text())
        integrity = validate_records(recs, rep, plan)
        kv = strict_kv_check(case, job['candidate'].split('_')[0], proto['kv_tokens_expected'], proto['kv_bytes'])
        plan_ok = rep['plan_sha256'] == job['plan_sha256'] and rep['prompt_sha256'] == job['prompt_sha256'] and rep['only'] == 'both'
        entry.update(state='done', kv=kv, plan_match=plan_ok, manifest=dict(requests=rep['requests'], errors=rep['errors'], drain_timeout_s=rep['drain_timeout_s']),
                     score=score_run(recs, grid, rep['warmup_s'], rep['duration_s']),
                     resources=dict(cards=gpus_of(job['candidate']), kv_total_gib=kv.get('kv_total_gib'),
                                    gpu_hours_observation=len(gpus_of(job['candidate'])) * rep['duration_s'] / 3600,
                                    energy=energy(case, rep['start_unix_s'], rep['warmup_s'], rep['duration_s']),
                                    mem_after_load_mib=(case / 'mem_after_load.txt').read_text().strip() if (case / 'mem_after_load.txt').exists() else None))
        entry['integrity'] = integrity
        entry['valid'] = bool(kv['ok'] and plan_ok and integrity['ok'])
        cols = sorted({r['columns'] for r in recs}, key=str)
        if entry['valid']:
            groups.setdefault(job['plan_stem'], []).append(dict(plan_sha256=rep['plan_sha256'], prompt_sha256=rep['prompt_sha256'], columns=cols,
                                                                denominators={k: v['n'] for k, v in entry['score']['services'].items()}))
        runs.append(entry)
    parity = strict_parity(groups)
    for entry in runs:
        if entry.get('valid') and not parity[entry['plan_stem']]['ok']:
            entry['valid'] = False
            entry['integrity']['problems'].append('cross-candidate parity mismatch')
    # alpha_max per candidate per cell
    alphas = proto['alphas']
    cands = proto['candidates']
    by = {(r['candidate'], r['alpha']): r for r in runs}
    cell_keys = [f'{T}|{P}|{D}|{g}' for T in grid['ttft_s'] for P in grid['tpot_s'] for D in grid['dit_e2e_s'] for g in grid['target']]
    capacity = {}
    for c in cands:
        capacity[c] = {}
        for key in cell_keys:
            verdicts = []
            for a in alphas:
                r = by.get((c, a))
                v = r['score']['cells'][key] if r and r['state'] == 'done' and r.get('valid') else None
                verdicts.append(v)
            passing = [a for a, v in zip(alphas, verdicts) if v]
            tested = [a for a, v in zip(alphas, verdicts) if v is not None]
            amax = max(passing) if passing else None
            nonmono = any(verdicts[i] is False and any(verdicts[j] for j in range(i + 1, len(verdicts))) for i in range(len(verdicts)))
            capacity[c][key] = dict(verdicts=dict(zip(map(str, alphas), verdicts)), alpha_max_tested=amax,
                                    bound='observed' if amax is not None else None,
                                    upper_unmeasured=bool(amax is not None and amax == max(alphas)),
                                    untested=[a for a in alphas if a not in tested], non_monotone=nonmono)
    # cost view
    cost = {}
    for c in cands:
        cards = gpus_of(c)
        cost[c] = dict(cards=cards, by_ratio={})
        for r in price['ratio_3090_over_4090']:
            cost[c]['by_ratio'][str(r)] = dict(occupied_only=cost_rate(cards, r, False), idle_billed=cost_rate(cards, r, True))
    summary = dict(protocol_sha256=proto['protocol_sha256'], runs=runs, parity=parity, capacity=capacity, cost=cost,
                   n_done=sum(r['state'] == 'done' for r in runs), n_valid=sum(bool(r.get('valid')) for r in runs), n_total=len(runs))
    (out_dir / 'stage1_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    (out_dir / 'stage1_summary.md').write_text(markdown(proto, summary))
    write_cells_csv(out_dir, proto, runs)
    return summary


def write_cells_csv(out_dir, proto, runs):
    grid = proto['slo_grid']
    with (out_dir / 'stage1_cells.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['candidate', 'alpha', 'ttft_s', 'tpot_s', 'dit_e2e_s', 'target', 'att_llm', 'att_dit', 'pass', 'valid'])
        for r in runs:
            if r['state'] != 'done':
                continue
            s = r['score']
            for T in grid['ttft_s']:
                for P in grid['tpot_s']:
                    for D in grid['dit_e2e_s']:
                        for g in grid['target']:
                            w.writerow([r['candidate'], r['alpha'], T, P, D, g, s['attainment_llm'][f'{T}|{P}'], s['attainment_dit'][str(D)],
                                        s['cells'][f'{T}|{P}|{D}|{g}'], r.get('valid')])


def fmt(x, nd=3):
    return '-' if x is None else f'{x:.{nd}f}'


def markdown(proto, S):
    grid = proto['slo_grid']; alphas = proto['alphas']
    L = ['# Stage 1 pilot: fixed-cost capacity map (exploratory, seed 901)', '',
         f"protocol {S['protocol_sha256'][:12]} · runs done {S['n_done']}/{S['n_total']} (valid {S['n_valid']}). "
         'Exploratory capacity map on one seed; not a selector validation. alpha_max is the largest *tested* alpha that passed '
         '(observed grid values only, never exact capacity bounds; no passing point means none observed). '
         'Untested cells are blank, never passes. Prices are research assumptions, not market quotes.', '',
         '## Runs', '', '| candidate | alpha | state | valid (KV+plan) | LLM n / ok / err / timeout / unfinished | n=1 tokens | DiT n / ok / err / timeout | '
         'LLM TTFT p90 (ok) | LLM TPOT p90 (ok) | DiT E2E p90 (ok) | backlog@end LLM/DiT | drain used s | saturation suspect | Wh obs 4090/3090 |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in S['runs']:
        if r['state'] != 'done':
            L.append(f"| {r['candidate']} | {r['alpha']} | {r['state']} | | | | | | | | | | | |"); continue
        s = r['score']; l, d = s['services']['llm'], s['services']['dit']; w = s['windows']; e = r['resources']['energy']
        L.append(f"| {r['candidate']} | {r['alpha']} | done | {r['valid']} | {l['n']}/{l['ok']}/{l['error']}/{l['drain_timeout']}/{l['unfinished']} | {l['n_tokens_eq_1']} | "
                 f"{d['n']}/{d['ok']}/{d['error']}/{d['drain_timeout']} | {fmt(l['success_only']['ttft_p90'])} | {fmt(l['success_only']['tpot_p90'],4)} | {fmt(d['success_only']['e2e_p90'])} | "
                 f"{w['llm']['backlog_at_observation_end']}/{w['dit']['backlog_at_observation_end']} | {w['llm']['drain_used_s']:.0f} | {w['llm']['saturation_suspect'] or w['dit']['saturation_suspect']} | "
                 f"{e.get('0',{}).get('observation_wh','-')}/{e.get('1',{}).get('observation_wh','-')} |")
    bad = [r for r in S['runs'] if r['state'] == 'done' and not r['valid']]
    if bad:
        L += ['', '**Invalid runs (excluded from alpha_max):** ' + '; '.join(f"{r['candidate']} α{r['alpha']}: {r['kv']['reason'] or r['integrity']['problems'] or 'plan/prompt hash mismatch'}" for r in bad)]
    L += ['', '## Stability windows (200 s × 3, measured arrivals / ok completions / failed ends / in-flight at window end)', '',
          '| candidate | alpha | service | w0 | w1 | w2 | completed in drain | drain timeouts |', '|---|---|---|---|---|---|---|---|']
    for r in S['runs']:
        if r['state'] != 'done':
            continue
        for kind in ('llm', 'dit'):
            w = r['score']['windows'][kind]
            cells = ' | '.join(f"{x['planned_arrivals']}/{x['completed_ok']}/{x['ended_failed']}/{x['inflight_at_end']}" for x in w['windows'])
            L.append(f"| {r['candidate']} | {r['alpha']} | {kind} | {cells} | {w['completed_during_drain']} | {w['drain_timeouts']} |")
    L += ['', '## alpha_max per SLO cell (all 192 cells; verdict per alpha, ✓ pass · ✗ fail · · untested)', '',
          'Attainment denominators include errors/timeouts/unfinished. A 99 % target with ~90 DiT samples is a point estimate (one failure breaks it).', '']
    hdr = '| TTFT s | TPOT ms | DiT E2E s | target | ' + ' | '.join(S['capacity'].keys()) + ' |'
    L += [hdr, '|' + '---|' * (4 + len(S['capacity']))]
    sym = {True: '✓', False: '✗', None: '·'}
    for T in grid['ttft_s']:
        for P in grid['tpot_s']:
            for D in grid['dit_e2e_s']:
                for g in grid['target']:
                    key = f'{T}|{P}|{D}|{g}'
                    cells = []
                    for c, cap in S['capacity'].items():
                        x = cap[key]
                        verd = ''.join(sym[x['verdicts'][str(a)]] for a in alphas)
                        am = '' if x['alpha_max_tested'] is None else f" observed={x['alpha_max_tested']}" + (' upper-unmeasured' if x['upper_unmeasured'] else '')
                        cells.append(verd + am + (' NM' if x['non_monotone'] else ''))
                    L.append(f'| {T} | {P*1000:g} | {D} | {g} | ' + ' | '.join(cells) + ' |')
    L += ['', '## Cost view (research assumptions)', '',
          f"Hourly cost in units of p_4090: n_4090 + r·n_3090, r ∈ {proto['price_model']['ratio_3090_over_4090']} "
          f"({proto['price_model']['ratio_note']}). Fixed-cost comparison is only between candidates with the same cards; 1-GPU candidates are compared via the price model.",
          '', '| candidate | cards | KV GiB total | ' + ' | '.join(f'r={r} occ / idle-billed' for r in proto['price_model']['ratio_3090_over_4090']) + ' |',
          '|---|---|---|' + '---|' * len(proto['price_model']['ratio_3090_over_4090'])]
    for c, x in S['cost'].items():
        kv = next((r['resources']['kv_total_gib'] for r in S['runs'] if r['candidate'] == c and r['state'] == 'done'), None)
        L.append(f"| {c} | {'+'.join(x['cards'])} | {kv or '-'} | " + ' | '.join(f"{v['occupied_only']:.2f} / {v['idle_billed']:.2f}" for v in x['by_ratio'].values()) + ' |')
    L += ['', 'alpha_max / cost for a given cell and r = alpha_max_tested(candidate, cell) ÷ cost(candidate, r); computed from the table above and `stage1_cells.csv`, '
          'is a tested-grid comparison only; preserve censoring and non-monotone flags rather than claiming exact capacity. GPU-time primary: cards × observation 600 s; Wh per card from gpu.csv '
          '(observation window and full replay incl. drain are separate columns in stage1_summary.json).', '',
          '## Parity (per plan: plan hash, prompt hash, CSV columns, denominators each exactly one non-empty value)', '']
    for k, p in S['parity'].items():
        L.append(f"- {k}: n={p['n']} ok={p['ok']} {p['fields']}")
    return '\n'.join(L) + '\n'


if __name__ == '__main__':
    build(sys.argv[1] if len(sys.argv) > 1 else ROOT / 'results/hetero/stage1_pilot')
    print('ok')
