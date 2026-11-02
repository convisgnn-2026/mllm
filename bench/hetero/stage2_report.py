#!/usr/bin/env python3
"""Stage 2: resident-solo profiles vs mixed runs, and selector comparison.

  python3 bench/hetero/stage2_report.py [--out results/hetero/stage2_report]

Inputs (read-only):
  mixed, profiling seed : results/hetero/stage1_pilot              (7 candidates x alpha 0.5/1.0 used)
  solo,  profiling seed : results/hetero/stage2_solo_llm, stage2_solo_dit (same plans, one service replayed)
  mixed, unseen seeds   : results/hetero/stage1_seed902, stage1_seed903  (4 candidates)

Per request we re-derive TTFT / mean TPOT / DiT E2E with stage1_report.derive (same definitions,
denominators include failures). A cell (TTFT, TPOT, DiT E2E, target) passes when both services reach
the target attainment.

Predictors of the mixed outcome:
  solo      : attainment of the solo runs, unchanged (assumes no runtime interference).
  solo+k    : solo per-request latencies scaled by k[c, metric] = mixed p90 / solo p90 fitted on the
              profiling seed. For the profiling seed this is evaluated leave-one-alpha-out (k from the
              other alpha); for unseen seeds k is fitted on seed 901 at the same alpha.
  oracle    : the mixed run itself.
Selectors per SLO cell:
  max-alpha : candidate with the largest predicted passing alpha (ties: lower hourly cost at r=0.75,
              occupied cards, then candidate order); realised = that candidate's mixed alpha_max.
  min-cost  : for each alpha, cheapest candidate predicted to pass; outcome is checked on mixed.
  fixed     : always C_4090, always S4_3090 (no model).
Nothing here is a capacity bound: alpha is the tested grid {0.5, 1.0} only.
"""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stage1_report as R  # noqa: E402

ROOT = HERE.parents[1]
CANDS = ['S4_4090', 'S3_3090', 'S4_3090', 'S3_4090', 'C_4090', 'C_3090', 'C_split']
ALPHAS = [0.5, 1.0]
R_PRICE = 0.75
METRICS = ('ttft', 'tpot', 'dit')


def cost(c):
    return R.cost_rate(R.gpus_of(c), R_PRICE, False)


# ----------------------------------------------------------------------------- loading
def per_request(case):
    recs = [r for r in R.load_requests(case) if r['measured']]
    llm, dit = [], []
    for r in recs:
        s, ttft, tpot, e2e, _ = R.derive(r)
        (llm if r['kind'] == 'llm' else dit).append((s, ttft, tpot, e2e))
    return dict(llm=llm, dit=dit)


def load_batch(d, solo_kind=None):
    """{(candidate, alpha): per_request dict} for valid done runs of a batch directory."""
    d = ROOT / d
    if not (d / 'protocol.json').exists():
        return {}
    proto = json.loads((d / 'protocol.json').read_text())
    valid = {}
    if solo_kind is None and (d / 'stage1_summary.json').exists():
        valid = {(r['candidate'], r['alpha']): bool(r.get('valid')) for r in json.loads((d / 'stage1_summary.json').read_text())['runs'] if r['state'] == 'done'}
    out = {}
    for job in proto['runs']:
        case = d / job['candidate'] / job['plan_stem']
        if job['alpha'] not in ALPHAS or not (case / 'done').exists():
            continue
        if solo_kind is None:
            if not valid.get((job['candidate'], job['alpha'])):
                continue
        else:
            st = d / 'status' / f"{job['run_id']}.json"
            st = json.loads(st.read_text()) if st.exists() else {}
            if not (st.get('outcome') == 'done' and st.get('kv_ok') and st.get('plan_match') and st.get('records_ok')):
                continue
        out[(job['candidate'], job['alpha'])] = per_request(case)
    return out


# ----------------------------------------------------------------------------- scoring
def p90(v):
    return R.pct(v, .9)


def stats(pr):
    ok_l = [x for x in pr['llm'] if x[0] == 'ok']; ok_d = [x for x in pr['dit'] if x[0] == 'ok']
    return dict(ttft=p90([x[1] for x in ok_l]), tpot=p90([x[2] for x in ok_l]), dit=p90([x[3] for x in ok_d]))


def att_llm(llm, T, P, k=(1., 1.)):
    if not llm:
        return None
    return sum(1 for s, ttft, tpot, _ in llm if s == 'ok' and ttft * k[0] <= T and (tpot is None or tpot * k[1] <= P)) / len(llm)


def att_dit(dit, D, k=1.):
    if not dit:
        return None
    return sum(1 for s, _, _, e2e in dit if s == 'ok' and e2e * k <= D) / len(dit)


def cells(llm, dit, grid, k=None):
    k = k or dict(ttft=1., tpot=1., dit=1.)
    out = {}
    for T in grid['ttft_s']:
        for P in grid['tpot_s']:
            a = att_llm(llm, T, P, (k['ttft'], k['tpot']))
            for D in grid['dit_e2e_s']:
                b = att_dit(dit, D, k['dit'])
                for g in grid['target']:
                    out[(T, P, D, g)] = None if a is None or b is None else (a >= g and b >= g)
    return out


def amax(table, c, key):
    passing = [a for a in ALPHAS if table.get((c, a), {}).get(key)]
    return max(passing) if passing else 0.


# ----------------------------------------------------------------------------- selectors
def pick_max_alpha(pred, cands, key):
    best = max(cands, key=lambda c: (amax(pred, c, key), -cost(c), -CANDS.index(c)))
    return best, amax(pred, best, key)


def eval_selectors(pred_tables, mixed, cands, grid):
    keys = list(next(iter(mixed.values())).keys())
    res = {}
    for name, pred in pred_tables.items():
        r = dict(cells=0, exact=0, regret_sum=0., overpromise=0, under=0,
                 mc_cases=0, mc_ok=0, mc_violation=0, mc_overpay=0, mc_missed=0, mc_cost_regret=0., mc_false_feasible=0)
        for key in keys:
            oracle = max(amax(mixed, c, key) for c in cands)
            if name.startswith('fixed:'):
                pick = name.split(':', 1)[1]; promised = None
            elif name == 'oracle':
                pick, promised = pick_max_alpha(mixed, cands, key)
            else:
                pick, promised = pick_max_alpha(pred, cands, key)
            real = amax(mixed, pick, key)
            if oracle > 0 or (promised or 0) > 0:
                r['cells'] += 1
                r['exact'] += real == oracle
                r['regret_sum'] += oracle - real
                if promised is not None:
                    r['overpromise'] += promised > real
                    r['under'] += promised < real
            for a in ALPHAS:
                feas = [c for c in cands if mixed.get((c, a), {}).get(key)]
                o_cost = min((cost(c) for c in feas), default=None)
                if name.startswith('fixed:'):
                    choice = name.split(':', 1)[1]
                elif name == 'oracle':
                    choice = min(feas, key=lambda c: (cost(c), CANDS.index(c))) if feas else None
                else:
                    pf = [c for c in cands if pred.get((c, a), {}).get(key)]
                    choice = min(pf, key=lambda c: (cost(c), CANDS.index(c))) if pf else None
                if o_cost is None:
                    r['mc_false_feasible'] += choice is not None  # nothing passes in mixed, selector still deploys
                    continue
                r['mc_cases'] += 1
                if choice is None:
                    r['mc_missed'] += 1; continue
                if not mixed.get((choice, a), {}).get(key):
                    r['mc_violation'] += 1; continue
                r['mc_ok'] += 1
                if cost(choice) > o_cost + 1e-9:
                    r['mc_overpay'] += 1; r['mc_cost_regret'] += cost(choice) - o_cost
        res[name] = r
    return res


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, default=ROOT / 'results/hetero/stage2_report')
    ap.add_argument('--mixed', default='results/hetero/stage1_pilot')
    ap.add_argument('--solo-llm', default='results/hetero/stage2_solo_llm')
    ap.add_argument('--solo-dit', default='results/hetero/stage2_solo_dit')
    ap.add_argument('--unseen', nargs='*', default=['results/hetero/stage1_seed902', 'results/hetero/stage1_seed903'])
    a = ap.parse_args()
    grid = json.loads((ROOT / a.mixed / 'protocol.json').read_text())['slo_grid']
    mixed_pr = load_batch(a.mixed)
    solo_l = load_batch(a.solo_llm, 'llm'); solo_d = load_batch(a.solo_dit, 'dit')
    pairs = sorted(set(mixed_pr) & set(solo_l) & set(solo_d), key=lambda x: (CANDS.index(x[0]), x[1]))
    cands = [c for c in CANDS if all((c, al) in pairs for al in ALPHAS)]
    S = dict(pairs=[list(p) for p in pairs], candidates_complete=cands, r_price=R_PRICE, alphas=ALPHAS)

    # interference: mixed vs solo p90 per (candidate, alpha)
    infl = {}
    for c, al in pairs:
        m = stats(mixed_pr[(c, al)]); s = dict(ttft=stats(solo_l[(c, al)])['ttft'], tpot=stats(solo_l[(c, al)])['tpot'], dit=stats(solo_d[(c, al)])['dit'])
        infl[(c, al)] = dict(mixed=m, solo=s, ratio={k: (m[k] / s[k] if m[k] and s[k] else None) for k in METRICS})
    S['inflation'] = [dict(candidate=c, alpha=al, **v) for (c, al), v in infl.items()]

    # predictor tables on the profiling seed
    mixed = {p: cells(mixed_pr[p]['llm'], mixed_pr[p]['dit'], grid) for p in pairs}
    solo = {p: cells(solo_l[p]['llm'], solo_d[p]['dit'], grid) for p in pairs}
    solok = {}
    for c, al in pairs:
        other = [x for x in ALPHAS if x != al]
        src = infl.get((c, other[0])) if other else None
        k = {m: (src['ratio'][m] or 1.) for m in METRICS} if src else dict(ttft=1., tpot=1., dit=1.)
        solok[(c, al)] = cells(solo_l[(c, al)]['llm'], solo_d[(c, al)]['dit'], grid, k)
    conf = {}
    for name, pred in (('solo', solo), ('solo+k (LOAO)', solok)):
        for c in cands:
            t = dict(tp=0, fp=0, fn=0, tn=0)
            for al in ALPHAS:
                for key, mv in mixed[(c, al)].items():
                    pv = pred[(c, al)][key]
                    t['tp' if pv and mv else 'fp' if pv and not mv else 'fn' if mv and not pv else 'tn'] += 1
            conf.setdefault(name, {})[c] = t
    S['confusion_profiling_seed'] = conf
    sel_names = lambda: {'solo': solo, 'solo+k (LOAO)': solok, 'oracle': mixed, 'fixed:C_4090': None, 'fixed:S4_3090': None}
    S['selectors_profiling_seed'] = eval_selectors(sel_names(), mixed, cands, grid) if cands else {}

    # unseen seeds: predictors from seed-901 profiles, oracle = that seed's mixed runs
    S['selectors_unseen'] = {}
    for d in a.unseen:
        um_pr = load_batch(d)
        if not um_pr:
            continue
        uc = [c for c in cands if all((c, al) in um_pr for al in ALPHAS)]
        um = {(c, al): cells(um_pr[(c, al)]['llm'], um_pr[(c, al)]['dit'], grid) for c in uc for al in ALPHAS}
        k_same = {(c, al): {m: (infl[(c, al)]['ratio'][m] or 1.) for m in METRICS} for c in uc for al in ALPHAS}
        us = {(c, al): solo[(c, al)] for c in uc for al in ALPHAS}
        uk = {(c, al): cells(solo_l[(c, al)]['llm'], solo_d[(c, al)]['dit'], grid, k_same[(c, al)]) for c in uc for al in ALPHAS}
        um901 = {(c, al): mixed[(c, al)] for c in uc for al in ALPHAS}
        tables = {'solo': us, 'solo+k (seed 901)': uk, 'mixed seed 901': um901, 'oracle': um, 'fixed:C_4090': None, 'fixed:S4_3090': None}
        S['selectors_unseen'][Path(d).name] = dict(candidates=uc, result=eval_selectors(tables, um, uc, grid))

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'stage2_summary.json').write_text(json.dumps(S, indent=2, default=str) + '\n')
    (a.out / 'stage2_summary.md').write_text(markdown(S))
    try:
        figures(S, a.out)
    except Exception as ex:  # figures are optional
        print('figures failed:', ex, file=sys.stderr)
    print(f'pairs {len(pairs)}, complete candidates {cands} -> {a.out}')


def markdown(S):
    L = ['# Stage 2: resident-solo profiles vs mixed, selector comparison', '',
         f"Profiling seed 901, alpha ∈ {S['alphas']} (tested grid only). Cost r = {S['r_price']} (p_3090 / p_4090, research assumption), occupied cards only.", '',
         '## Interference: mixed p90 / solo p90', '', '| candidate | alpha | TTFT solo→mixed s | TPOT solo→mixed ms | DiT E2E solo→mixed s | ratios TTFT/TPOT/DiT |', '|---|---|---|---|---|---|']
    f = lambda x, s=1, nd=3: '-' if x is None else f'{x * s:.{nd}f}'
    for v in S['inflation']:
        s, m, r = v['solo'], v['mixed'], v['ratio']
        L.append(f"| {v['candidate']} | {v['alpha']} | {f(s['ttft'])} → {f(m['ttft'])} | {f(s['tpot'], 1000, 1)} → {f(m['tpot'], 1000, 1)} | {f(s['dit'])} → {f(m['dit'])} | "
                 f"{f(r['ttft'], 1, 2)} / {f(r['tpot'], 1, 2)} / {f(r['dit'], 1, 2)} |")
    L += ['', '## Cell prediction vs mixed (profiling seed, 192 cells × 2 alphas per candidate)', '',
          'fp = predictor says pass, mixed fails (SLO violation risk); fn = predictor says fail, mixed passes (lost capacity).', '',
          '| predictor | candidate | tp | fp | fn | tn |', '|---|---|---|---|---|---|']
    for name, per in S['confusion_profiling_seed'].items():
        for c, t in per.items():
            L.append(f"| {name} | {c} | {t['tp']} | {t['fp']} | {t['fn']} | {t['tn']} |")

    def sel_table(res, title):
        out = ['', f'## {title}', '',
               'max-alpha: over SLO cells where someone passes or is predicted to pass. exact = realised alpha_max equals oracle; regret = Σ(oracle − realised); overpromise = predicted alpha above realised.',
               'min-cost: per (cell, alpha) where at least one candidate passes in mixed. violation = chosen candidate fails in mixed; overpay = passes but costs more than the cheapest passing one; missed = selector finds none. false-feasible = (cell, alpha) where nothing passes in mixed but the selector still deploys (counted separately).', '',
               '| selector | max-α cells | exact | regret Σα | overpromise | min-cost cases | ok (incl. overpay) | violation | overpay | missed | cost regret Σ | false-feasible |', '|---|---|---|---|---|---|---|---|---|---|---|---|']
        for n, r in res.items():
            out.append(f"| {n} | {r['cells']} | {r['exact']} | {r['regret_sum']:.2f} | {r['overpromise']} | {r['mc_cases']} | {r['mc_ok']} | {r['mc_violation']} | {r['mc_overpay']} | {r['mc_missed']} | {r['mc_cost_regret']:.2f} | {r['mc_false_feasible']} |")
        return out
    if S['selectors_profiling_seed']:
        L += sel_table(S['selectors_profiling_seed'], f"Selectors on the profiling seed (candidates: {', '.join(S['candidates_complete'])})")
    for d, v in S['selectors_unseen'].items():
        L += sel_table(v['result'], f"Selectors on unseen {d} (candidates: {', '.join(v['candidates'])}; profiles and k from seed 901)")
    return '\n'.join(L) + '\n'


def figures(S, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    color = {'S4_4090': '#777777', 'S3_3090': '#bbbbbb', 'S4_3090': '#0072B2', 'S3_4090': '#56B4E9', 'C_4090': '#D55E00', 'C_3090': '#E69F00', 'C_split': '#009E73'}
    inf = S['inflation']
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, (m, t) in zip(axs, [('ttft', 'LLM TTFT p90'), ('tpot', 'LLM TPOT p90'), ('dit', 'DiT E2E p90')]):
        labels = [f"{v['candidate']} α{v['alpha']:g}" for v in inf]
        vals = [v['ratio'][m] or np.nan for v in inf]
        ax.bar(range(len(inf)), vals, color=[color[v['candidate']] for v in inf])
        ax.axhline(1, color='k', lw=.6, ls=':')
        ax.set_yscale('log'); ax.set_xticks(range(len(inf)), labels, fontsize=5.5, rotation=90); ax.set_title(f'{t}: mixed / solo', fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Runtime interference: mixed p90 divided by resident-solo p90 (same deployment, same plan, seed 901)', fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, .92))
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(out / f'fig_stage2_interference.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)
    blocks = [('seed 901 (profiling)', S['selectors_profiling_seed'])] + [(d, v['result']) for d, v in S['selectors_unseen'].items()]
    blocks = [b for b in blocks if b[1]]
    if not blocks:
        return
    fig, axs = plt.subplots(2, len(blocks), figsize=(4 * len(blocks), 5.2), squeeze=False)
    for j, (title, res) in enumerate(blocks):
        names = list(res)
        ax = axs[0, j]
        ax.bar(range(len(names)), [res[n]['exact'] / max(res[n]['cells'], 1) for n in names], color='#0072B2')
        ax.set_ylim(0, 1.05); ax.set_title(f'{title}\nmax-α: share of cells matching oracle', fontsize=8); ax.set_xticks(range(len(names)), names, fontsize=6, rotation=25, ha='right')
        ax = axs[1, j]
        bottom = np.zeros(len(names))
        for k, col, lab in (('mc_ok', '#009E73', 'ok'), ('mc_overpay', '#E69F00', 'ok but overpays'), ('mc_violation', '#D55E00', 'SLO violated in mixed'), ('mc_missed', '#bbbbbb', 'missed feasible')):
            v = np.array([res[n][k] - (res[n]['mc_overpay'] if k == 'mc_ok' else 0) for n in names], float)
            ax.bar(range(len(names)), v, bottom=bottom, color=col, label=lab); bottom += v
        ax.set_title('min-cost: outcome per (cell, α)', fontsize=8); ax.set_xticks(range(len(names)), names, fontsize=6, rotation=25, ha='right')
        if j == 0:
            ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(out / f'fig_stage2_selectors.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
