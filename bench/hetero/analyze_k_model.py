#!/usr/bin/env python3
"""Can the interference ratio k be computed from solo profiles instead of measured in mixed runs?

  python3 bench/hetero/analyze_k_model.py [--out results/hetero/k_model]

Inputs (read-only): resident-solo runs (stage2_solo_*), mixed runs (stage1_pilot, stage1_refine), and the
1 Hz nvidia-smi samples (gpu.csv) of the solo runs. For every shared-card placement and alpha with a
measured k we compute, per metric, from the *solo* runs only:
  u_other : busy fraction of the other service on the GPU(s) the metric's stage runs on
  u_own   : busy fraction of the service itself on that GPU
  r       : L_other / L_own, ratio of the other service's step length to this stage's step length
            (decode step = solo TPOT p50; DiT step = solo DiT E2E p50 / 28; prefill = solo TTFT p50)
Stage -> GPU: TTFT uses the prefill GPU and, for P/D split, also the decode GPU (first token comes from the
decoder after the KV transfer); TPOT uses the decode GPU; DiT uses the denoise GPU.
Models (fitted on log k, leave-one-placement-out, one parameter set per metric):
  LIN : k = 1 + a u_other
  PS  : k = 1 / (1 - a u_other)                       time-sharing
  PSG : k = 1 / (1 - a u_other) + b u_other r         time-sharing + waiting behind the other service's long step
Outputs k_model.md/.json and fig_k_model.png. Exploratory; nothing here changes protocols or results.
"""
import argparse
import collections
import datetime as dt
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stage1_report as R  # noqa: E402
import stage2_report as S2  # noqa: E402

ROOT = HERE.parents[1]
STAGES = {  # GPU index per stage (0 = 4090, 1 = 3090)
    'S4_4090': dict(prefill=0, decode=0, dit=0), 'S3_3090': dict(prefill=1, decode=1, dit=1),
    'S4_3090': dict(prefill=0, decode=0, dit=1), 'S3_4090': dict(prefill=1, decode=1, dit=0),
    'C_4090': dict(prefill=0, decode=1, dit=0), 'C_3090': dict(prefill=0, decode=1, dit=1), 'C_split': dict(prefill=0, decode=1, dit=0),
}
SOLO = {'llm': ['results/hetero/stage2_solo_llm', 'results/hetero/stage2_solo_llm_b'], 'dit': ['results/hetero/stage2_solo_dit', 'results/hetero/stage2_solo_dit_b']}
MIXED = ['results/hetero/stage1_pilot', 'results/hetero/stage1_refine']
DIT_STEPS = 28
MODELS = {  # params -> prediction; grids searched
    'LIN': (lambda p, x: 1 + p[0] * x['u'], [(a / 100,) for a in range(1, 800)]),
    'PS': (lambda p, x: 1 / max(1e-3, 1 - p[0] * x['u']), [(a / 100,) for a in range(1, 200)]),
    'PSG': (lambda p, x: 1 / max(1e-3, 1 - p[0] * x['u']) + p[1] * x['u'] * x['r'], [(a / 50, b / 20) for a in range(0, 80) for b in range(0, 80)]),
}


def busy(case):
    rt = json.loads((case / 'replay' / 'runtime.json').read_text())
    t0 = rt['start_unix_s'] + rt['warmup_s']; t1 = t0 + rt['duration_s']
    acc = collections.defaultdict(list)
    for line in (case / 'gpu.csv').read_text().splitlines():
        p = [x.strip() for x in line.split(',')]
        if len(p) < 3:
            continue
        try:
            ts = dt.datetime.strptime(p[0], '%Y/%m/%d %H:%M:%S.%f').timestamp()
        except ValueError:
            continue
        if t0 <= ts <= t1:
            acc[int(p[1])].append(float(p[2].rstrip(' %')) / 100)
    return {g: sum(v) / len(v) for g, v in acc.items() if v}


def p50(case):
    pr = S2.per_request(case)
    ok_l = [x for x in pr['llm'] if x[0] == 'ok']; ok_d = [x for x in pr['dit'] if x[0] == 'ok']
    return dict(ttft=R.pct([x[1] for x in ok_l], .5), tpot=R.pct([x[2] for x in ok_l], .5), dit=R.pct([x[3] for x in ok_d], .5))


def runs():
    solo, mixed = {}, {}
    for kind, dirs in SOLO.items():
        for d in dirs:
            d = ROOT / d
            if (d / 'protocol.json').exists():
                for job in json.loads((d / 'protocol.json').read_text())['runs']:
                    case = d / job['candidate'] / job['plan_stem']
                    if (case / 'done').exists() and (case / 'gpu.csv').exists():
                        solo.setdefault((kind, job['candidate'], job['alpha']), case)
    for d in MIXED:
        d = ROOT / d
        valid = {(r['candidate'], r['alpha']) for r in json.loads((d / 'stage1_summary.json').read_text())['runs'] if r['state'] == 'done' and r.get('valid')}
        for job in json.loads((d / 'protocol.json').read_text())['runs']:
            if (job['candidate'], job['alpha']) in valid:
                mixed.setdefault((job['candidate'], job['alpha']), d / job['candidate'] / job['plan_stem'])
    return solo, mixed


def features(c, ul, ud, sl50, sd50):
    """per metric: dict(u=u_other, uo=u_own, r=L_other/L_own) from solo runs only."""
    st = STAGES[c]; gp, gd, gt = st['prefill'], st['decode'], st['dit']
    L_dec, L_pre, L_dit = sl50['tpot'], sl50['ttft'], sd50['dit'] / DIT_STEPS
    L_llm_on = lambda g: L_dec if gd == g else L_pre  # the LLM step length that the DiT GPU sees
    ttft_g = {gp, gd}
    return dict(
        ttft=dict(u=sum(ud.get(g, 0.) for g in ttft_g), uo=max(ul.get(g, 0.) for g in ttft_g), r=L_dit / L_pre),
        tpot=dict(u=ud.get(gd, 0.), uo=ul.get(gd, 0.), r=L_dit / L_dec),
        dit=dict(u=ul.get(gt, 0.), uo=ud.get(gt, 0.), r=L_llm_on(gt) / L_dit))


def fit(points, model):
    f, grid = MODELS[model]
    best = None
    for p in grid:
        err = sum((math.log(max(1e-6, f(p, x))) - math.log(x['k'])) ** 2 for x in points)
        if best is None or err < best[1]:
            best = (p, err)
    return best[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, default=ROOT / 'results/hetero/k_model')
    ap.add_argument('--max-alpha', type=float, default=1.25)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    solo, mixed = runs()
    rows = []
    for (c, al), mcase in sorted(mixed.items(), key=lambda x: (list(STAGES).index(x[0][0]), x[0][1])):
        if ('llm', c, al) not in solo or ('dit', c, al) not in solo:
            continue
        sl, sd = solo[('llm', c, al)], solo[('dit', c, al)]
        pm = S2.stats(S2.per_request(mcase)); pl = S2.stats(S2.per_request(sl)); pd_ = S2.stats(S2.per_request(sd))
        k = dict(ttft=pm['ttft'] / pl['ttft'], tpot=pm['tpot'] / pl['tpot'], dit=pm['dit'] / pd_['dit'])
        rows.append(dict(candidate=c, alpha=al, k=k, feat=features(c, busy(sl), busy(sd), p50(sl), p50(sd))))
    pts_all = [dict(c=r['candidate'], al=r['alpha'], m=m, k=r['k'][m], **r['feat'][m]) for r in rows for m in S2.METRICS if r['feat'][m]['u'] > 0.03]
    nonshared = [round(r['k'][m], 2) for r in rows for m in S2.METRICS if r['feat'][m]['u'] <= 0.03]
    pts = [p for p in pts_all if p['al'] <= a.max_alpha]; sat = [p for p in pts_all if p['al'] > a.max_alpha]
    cands = sorted({p['c'] for p in pts}, key=list(STAGES).index)
    res = dict(rows=rows, max_alpha=a.max_alpha, n_points=len(pts), nonshared_k=nonshared, fits={})
    for model in MODELS:
        f = MODELS[model][0]; par = {}; loo = []
        for m in S2.METRICS:
            ps = [p for p in pts if p['m'] == m]
            par[m] = fit(ps, model)
            for c in cands:
                train = [p for p in ps if p['c'] != c]; test = [p for p in ps if p['c'] == c]
                if train and test:
                    pc = fit(train, model)
                    loo += [dict(**p, k_pred=f(pc, p), rel_err=f(pc, p) / p['k'] - 1) for p in test]
        errs = sorted(abs(x['rel_err']) for x in loo)
        by_m = {m: sorted(abs(x['rel_err']) for x in loo if x['m'] == m) for m in S2.METRICS}
        res['fits'][model] = dict(params=par, loo=loo, n=len(errs), median=errs[len(errs) // 2], p90=errs[int(0.9 * (len(errs) - 1))], within25=sum(e <= 0.25 for e in errs),
                                  median_by_metric={m: (v[len(v) // 2] if v else None) for m, v in by_m.items()},
                                  saturated=[dict(**p, k_pred=f(par[p['m']], p)) for p in sat])
    (a.out / 'k_model.json').write_text(json.dumps(res, indent=2) + '\n')
    L = ['# k from solo profiles only (exploratory)', '',
         f'Features from the resident-solo runs: u_other / u_own = mean nvidia-smi busy fraction on the GPU(s) of the metric\'s stage; r = L_other / L_own (step-length ratio). k = mixed p90 / solo p90. Fits use alpha <= {a.max_alpha}; saturated alphas are extrapolation checks.', '',
         '| candidate | alpha | TTFT u/uo/r | TPOT u/uo/r | DiT u/uo/r | k TTFT | k TPOT | k DiT |', '|---|---|---|---|---|---|---|---|']
    for r in rows:
        L.append(f"| {r['candidate']} | {r['alpha']} | " + ' | '.join(f"{r['feat'][m]['u']:.2f}/{r['feat'][m]['uo']:.2f}/{r['feat'][m]['r']:.1f}" for m in S2.METRICS) + ' | ' + ' | '.join(f"{r['k'][m]:.2f}" for m in S2.METRICS) + ' |')
    L += ['', 'Non-shared stage/metric pairs (u_other <= 0.03): k = ' + ', '.join(f'{x:.2f}' for x in nonshared) + ' (expected 1.0)', '',
          f'## Fits per metric, leave-one-placement-out, alpha <= {a.max_alpha} ({len(pts)} points, {len(cands)} placements)', '',
          '| model | params (ttft / tpot / dit) | median abs rel err | p90 | within 25% | median by metric ttft / tpot / dit |', '|---|---|---|---|---|---|']
    for name, f in res['fits'].items():
        L.append(f"| {name} | " + ' / '.join(','.join(f'{v:.2f}' for v in f['params'][m]) for m in S2.METRICS) + f" | {f['median']:.0%} | {f['p90']:.0%} | {f['within25']}/{f['n']} | "
                 + ' / '.join(f"{f['median_by_metric'][m]:.0%}" for m in S2.METRICS) + ' |')
    best = min(res['fits'], key=lambda n: res['fits'][n]['median'])
    L += ['', f'## Leave-one-out predictions, {best}', '', '| candidate | alpha | metric | u_other | u_own | r | k measured | k predicted | rel err |', '|---|---|---|---|---|---|---|---|---|']
    for x in res['fits'][best]['loo']:
        L.append(f"| {x['c']} | {x['al']} | {x['m']} | {x['u']:.2f} | {x['uo']:.2f} | {x['r']:.1f} | {x['k']:.2f} | {x['k_pred']:.2f} | {x['rel_err']:+.0%} |")
    L += ['', f'## Saturated runs (alpha > {a.max_alpha}), {best} with all-data params', '', '| candidate | alpha | metric | u_other | u_own | k measured | k predicted |', '|---|---|---|---|---|---|---|']
    for x in res['fits'][best]['saturated']:
        L.append(f"| {x['c']} | {x['al']} | {x['m']} | {x['u']:.2f} | {x['uo']:.2f} | {x['k']:.2f} | {x['k_pred']:.2f} |")
    (a.out / 'k_model.md').write_text('\n'.join(L) + '\n')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        fig, axs = plt.subplots(1, 3, figsize=(10, 3.3))
        col = {c: plt.cm.tab10(i) for i, c in enumerate(list(STAGES))}
        for ax, m in zip(axs, S2.METRICS):
            for name, mk in (('PS', 'o'), ('PSG', 's')):
                for x in res['fits'][name]['loo']:
                    if x['m'] == m:
                        ax.plot(x['k'], x['k_pred'], mk, color=col[x['c']], ms=5, mfc='none' if name == 'PS' else col[x['c']])
            lim = [0.9, max(x['k'] for x in pts if x['m'] == m) * 1.3]
            ax.plot(lim, lim, 'k-', lw=0.6); ax.plot(lim, [l * 1.25 for l in lim], 'k:', lw=0.5); ax.plot(lim, [l / 1.25 for l in lim], 'k:', lw=0.5)
            ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(lim); ax.set_ylim(lim)
            ax.set_title(f"{m.upper()}: LOO median  PS {res['fits']['PS']['median_by_metric'][m]:.0%}, PSG {res['fits']['PSG']['median_by_metric'][m]:.0%}", fontsize=8)
            ax.set_xlabel('k measured (mixed / solo p90)'); ax.spines[['top', 'right']].set_visible(False)
        axs[0].set_ylabel('k predicted from solo profiles (leave-one-placement-out)')
        h = [Line2D([], [], color=col[c], marker='s', ls='', label=c) for c in cands] + [Line2D([], [], color='k', marker='o', mfc='none', ls='', label='PS (time-sharing)'), Line2D([], [], color='k', marker='s', ls='', label='PSG (+ step-length term)')]
        fig.legend(handles=h, loc='upper center', ncol=len(h), frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.04))
        fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(a.out / 'fig_k_model.png', dpi=200, bbox_inches='tight')
    except Exception as ex:
        print('figure failed:', ex, file=sys.stderr)
    i = L.index(f'## Fits per metric, leave-one-placement-out, alpha <= {a.max_alpha} ({len(pts)} points, {len(cands)} placements)')
    print('\n'.join(L[i:]))


if __name__ == '__main__':
    main()
