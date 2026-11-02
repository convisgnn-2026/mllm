#!/usr/bin/env python3
"""Stage 2b: solver with / without comparison on unseen seeds, using only inputs a solver can have.

  python3 bench/hetero/stage2b_report.py [--out results/hetero/stage2b_report]

Inputs (read-only):
  mixed, seed 901 : results/hetero/stage1_pilot + stage1_refine         (all tested alphas)
  solo,  seed 901 : results/hetero/stage2_solo_{llm,dit} + stage2_solo_{llm,dit}_b
  mixed, unseen   : results/hetero/stage1_seed902, stage1_seed903        (4 candidates x alpha 0.5..1.25)

Candidates: the four decision-map winners present on the unseen seeds. Two share a card between the
services (S4_4090, C_4090: interference measured in Stage 2), two keep them on separate cards
(S4_3090, S3_4090: mixed/solo p90 ratio 0.99-1.07 in Stage 2, so k = 1 and the seed-901 mixed run
stands in for the solo profile).

Profile rule: a candidate's prediction at alpha uses its seed-901 profile at that alpha if one
exists, otherwise the profile at the nearest *larger* profiled alpha (conservative: capacity is
treated as monotone in alpha, the same assumption behind alpha_max).

Interference models for the shared candidates (k per metric = mixed p90 / solo p90):
  k-linear  : straight line through the two anchors alpha 0.5 and 1.0, clipped at >= 1   <- the solver
  k-const   : k of the nearest anchor (0.5 for alpha <= 0.75, else 1.0)
  k-measured: k measured at the same alpha (needs a mixed run there; upper bound of the k approach)
Selectors without a solver: fixed S4_4090 / S4_3090 / C_4090, and 'solo only' (k = 1 everywhere).
Oracle = the unseen seed's own mixed runs. Evaluation alphas: anchors {0.5, 1.0} and held-out
{0.6, 0.75, 1.1, 1.25}; the held-out set is the fair test (alpha unseen by k, seed unseen by all).
"""
import argparse
import collections
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stage1_report as R  # noqa: E402
import stage2_report as S2  # noqa: E402

ROOT = HERE.parents[1]
CANDS = ['S4_4090', 'S4_3090', 'S3_4090', 'C_4090']
SHARED = ['S4_4090', 'C_4090']
ANCHORS = [0.5, 1.0]
EVAL = [0.5, 0.6, 0.75, 1.0, 1.1, 1.25]
HELD = [a for a in EVAL if a not in ANCHORS]
METRICS = S2.METRICS
LABEL = {'S4_4090': 'One GPU: 4090', 'S4_3090': 'LLM 4090 / DiT 3090', 'S3_4090': 'LLM 3090 / DiT 4090', 'C_4090': 'P/D split + DiT 4090'}
cost = S2.cost


# ----------------------------------------------------------------------------- loading
def load_mixed(dirs):
    out = {}
    for d in dirs:
        d = ROOT / d
        if not (d / 'stage1_summary.json').exists():
            continue
        proto = json.loads((d / 'protocol.json').read_text())
        valid = {(r['candidate'], r['alpha']) for r in json.loads((d / 'stage1_summary.json').read_text())['runs'] if r['state'] == 'done' and r.get('valid')}
        for job in proto['runs']:
            key = (job['candidate'], job['alpha'])
            if key in valid and key not in out and job['candidate'] in CANDS:
                out[key] = S2.per_request(d / job['candidate'] / job['plan_stem'])
    return out


def load_solo(dirs):
    out = {}
    for d in dirs:
        d = ROOT / d
        if not (d / 'protocol.json').exists():
            continue
        for job in json.loads((d / 'protocol.json').read_text())['runs']:
            st = d / 'status' / f"{job['run_id']}.json"
            st = json.loads(st.read_text()) if st.exists() else {}
            case = d / job['candidate'] / job['plan_stem']
            if (case / 'done').exists() and st.get('outcome') == 'done' and st.get('kv_ok') and st.get('plan_match') and st.get('records_ok') and job['candidate'] in CANDS:
                out.setdefault((job['candidate'], job['alpha']), S2.per_request(case))
    return out


# ----------------------------------------------------------------------------- interference models
def k_table(mixed, solo_l, solo_d):
    """measured k[(c, alpha)][metric] for shared candidates where both mixed and solo exist."""
    k = {}
    for c in SHARED:
        for (cc, al) in sorted(mixed):
            if cc != c or (c, al) not in solo_l or (c, al) not in solo_d:
                continue
            m = S2.stats(mixed[(c, al)])
            s = dict(ttft=S2.stats(solo_l[(c, al)])['ttft'], tpot=S2.stats(solo_l[(c, al)])['tpot'], dit=S2.stats(solo_d[(c, al)])['dit'])
            k[(c, al)] = {x: (m[x] / s[x] if m[x] and s[x] else None) for x in METRICS}
    return k


def k_model(kt, c, al, mode):
    one = dict(ttft=1., tpot=1., dit=1.)
    if c not in SHARED:
        return one
    a0, a1 = ANCHORS
    k0, k1 = kt.get((c, a0)), kt.get((c, a1))
    if mode == 'measured':
        src = kt.get((c, al))
        return {m: (src[m] or 1.) for m in METRICS} if src else None
    if not k0 or not k1:
        return None
    if mode == 'const':
        src = k0 if al <= 0.75 else k1
        return {m: (src[m] or 1.) for m in METRICS}
    if mode == 'linear':
        out = {}
        for m in METRICS:
            y0, y1 = k0[m] or 1., k1[m] or 1.
            out[m] = max(1., y0 + (y1 - y0) * (al - a0) / (a1 - a0))
        return out
    raise ValueError(mode)


# ----------------------------------------------------------------------------- predictions
def profile_alpha(avail, al):
    """alpha of the profile to use for alpha al: exact if present, else nearest larger (None if none)."""
    if al in avail:
        return al
    larger = [a for a in avail if a > al]
    return min(larger) if larger else None


def predict(c, al, grid, mixed901, solo_l, solo_d, kt, kmode):
    """cells dict predicted for (c, al), or None when no usable profile. kmode None -> k = 1 (solo only)."""
    if c in SHARED:
        avail = sorted({a for (cc, a) in solo_l if cc == c} & {a for (cc, a) in solo_d if cc == c})
        ap = profile_alpha(avail, al)
        if ap is None:
            return None
        k = dict(ttft=1., tpot=1., dit=1.) if kmode is None else k_model(kt, c, ap, kmode)
        if k is None:
            return None
        return S2.cells(solo_l[(c, ap)]['llm'], solo_d[(c, ap)]['dit'], grid, k)
    avail = sorted(a for (cc, a) in mixed901 if cc == c)
    ap = profile_alpha(avail, al)
    if ap is None:
        return None
    return S2.cells(mixed901[(c, ap)]['llm'], mixed901[(c, ap)]['dit'], grid)


def mixed_table(mixed, cands, alphas, grid):
    return {(c, a): S2.cells(mixed[(c, a)]['llm'], mixed[(c, a)]['dit'], grid) for c in cands for a in alphas if (c, a) in mixed}


# ----------------------------------------------------------------------------- evaluation
def amax(table, c, key, alphas):
    passing = [a for a in alphas if table.get((c, a)) and table[(c, a)].get(key)]
    return max(passing) if passing else 0.


def evaluate(pred_tables, oracle, cands, alphas):
    keys = list(next(iter(oracle.values())).keys())
    res = {}
    for name, pred in pred_tables.items():
        r = dict(cells=0, exact=0, regret_sum=0., overpromise=0, mc_cases=0, mc_ok=0, mc_violation=0, mc_overpay=0, mc_missed=0, mc_cost_regret=0., mc_false_feasible=0, picks=collections.Counter())
        for key in keys:
            o_best = max(amax(oracle, c, key, alphas) for c in cands)
            if name.startswith('fixed:'):
                pick, promised = name.split(':', 1)[1], None
            else:
                src = oracle if name == 'oracle' else pred
                pick = max(cands, key=lambda c: (amax(src, c, key, alphas), -cost(c), -CANDS.index(c)))
                promised = amax(src, pick, key, alphas)
            real = amax(oracle, pick, key, alphas)
            if o_best > 0 or (promised or 0) > 0:
                r['cells'] += 1; r['exact'] += real == o_best; r['regret_sum'] += o_best - real
                if promised is not None:
                    r['overpromise'] += promised > real
            for a in alphas:
                feas = [c for c in cands if oracle.get((c, a), {}).get(key)]
                o_cost = min((cost(c) for c in feas), default=None)
                if name.startswith('fixed:'):
                    choice = name.split(':', 1)[1]
                elif name == 'oracle':
                    choice = min(feas, key=lambda c: (cost(c), CANDS.index(c))) if feas else None
                else:
                    pf = [c for c in cands if pred.get((c, a)) and pred[(c, a)].get(key)]
                    choice = min(pf, key=lambda c: (cost(c), CANDS.index(c))) if pf else None
                if o_cost is None:
                    r['mc_false_feasible'] += choice is not None
                    continue
                r['mc_cases'] += 1
                if choice is None:
                    r['mc_missed'] += 1; continue
                r['picks'][choice] += 1
                if not oracle.get((choice, a), {}).get(key):
                    r['mc_violation'] += 1; continue
                r['mc_ok'] += 1
                if cost(choice) > o_cost + 1e-9:
                    r['mc_overpay'] += 1; r['mc_cost_regret'] += cost(choice) - o_cost
        r['picks'] = dict(r['picks'])
        res[name] = r
    return res


def confusion(pred, oracle, pairs):
    t = dict(tp=0, fp=0, fn=0, tn=0)
    for p in pairs:
        if not pred.get(p) or not oracle.get(p):
            continue
        for key, mv in oracle[p].items():
            pv = pred[p][key]
            t['tp' if pv and mv else 'fp' if pv and not mv else 'fn' if mv and not pv else 'tn'] += 1
    return t


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, default=ROOT / 'results/hetero/stage2b_report')
    ap.add_argument('--mixed901', nargs='*', default=['results/hetero/stage1_pilot', 'results/hetero/stage1_refine'])
    ap.add_argument('--solo-llm', nargs='*', default=['results/hetero/stage2_solo_llm', 'results/hetero/stage2_solo_llm_b'])
    ap.add_argument('--solo-dit', nargs='*', default=['results/hetero/stage2_solo_dit', 'results/hetero/stage2_solo_dit_b'])
    ap.add_argument('--unseen', nargs='*', default=['results/hetero/stage1_seed902', 'results/hetero/stage1_seed903'])
    a = ap.parse_args()
    grid = json.loads((ROOT / a.mixed901[0] / 'protocol.json').read_text())['slo_grid']
    mixed901 = load_mixed(a.mixed901); solo_l = load_solo(a.solo_llm); solo_d = load_solo(a.solo_dit)
    kt = k_table(mixed901, solo_l, solo_d)
    S = dict(candidates=CANDS, shared=SHARED, anchors=ANCHORS, eval_alphas=EVAL, held_out_alphas=HELD, r_price=S2.R_PRICE,
             solo_alphas={c: sorted({al for (cc, al) in solo_l if cc == c} & {al for (cc, al) in solo_d if cc == c}) for c in SHARED},
             mixed901_alphas={c: sorted(al for (cc, al) in mixed901 if cc == c) for c in CANDS})

    # measured k vs the two-anchor models, on seed 901
    S['k_measured'] = [dict(candidate=c, alpha=al, k=v) for (c, al), v in sorted(kt.items())]
    S['k_model_error'] = []
    for (c, al), v in sorted(kt.items()):
        if al in ANCHORS:
            continue
        row = dict(candidate=c, alpha=al)
        for mode in ('linear', 'const'):
            km = k_model(kt, c, al, mode)
            row[mode] = {m: (km[m] / v[m] - 1 if km and v[m] else None) for m in METRICS}
        S['k_model_error'].append(row)
    # seed-901 held-out cells for shared candidates: predicted with k models vs the mixed run at the same alpha
    pairs901 = [(c, al) for (c, al) in sorted(kt) if al not in ANCHORS]
    or901 = mixed_table(mixed901, SHARED, sorted({al for _, al in pairs901}), grid)
    S['confusion_901_heldout'] = {}
    for name, mode in (('solo only', None), ('solo+k-linear', 'linear'), ('solo+k-const', 'const'), ('solo+k-measured', 'measured')):
        pred = {p: predict(p[0], p[1], grid, mixed901, solo_l, solo_d, kt, mode) for p in pairs901}
        S['confusion_901_heldout'][name] = {c: confusion(pred, or901, [p for p in pairs901 if p[0] == c]) for c in SHARED}

    # unseen seeds
    def tables(alphas):
        t = {}
        t['solo only'] = {(c, al): predict(c, al, grid, mixed901, solo_l, solo_d, kt, None) for c in CANDS for al in alphas}
        t['solo+k-const (2 anchors)'] = {(c, al): predict(c, al, grid, mixed901, solo_l, solo_d, kt, 'const') for c in CANDS for al in alphas}
        t['solo+k-linear (2 anchors)'] = {(c, al): predict(c, al, grid, mixed901, solo_l, solo_d, kt, 'linear') for c in CANDS for al in alphas}
        t['solo+k-measured (same alpha)'] = {(c, al): predict(c, al, grid, mixed901, solo_l, solo_d, kt, 'measured') for c in CANDS for al in alphas}
        m901 = {}
        for c in CANDS:
            avail = sorted(al for (cc, al) in mixed901 if cc == c)
            for al in alphas:
                ap_ = profile_alpha(avail, al)
                m901[(c, al)] = S2.cells(mixed901[(c, ap_)]['llm'], mixed901[(c, ap_)]['dit'], grid) if ap_ is not None else None
        t['mixed seed 901'] = m901
        t['oracle'] = None
        for c in CANDS:
            t[f'fixed:{c}'] = None
        return t
    S['unseen'] = {}
    pooled = {}
    for d in a.unseen:
        um = load_mixed([d])
        if not um:
            continue
        cands = [c for c in CANDS if all((c, al) in um for al in EVAL)]
        res = {}
        for tag, alphas in (('all', EVAL), ('held_out', HELD), ('anchors', ANCHORS)):
            oracle = mixed_table(um, cands, alphas, grid)
            res[tag] = evaluate(tables(alphas), oracle, cands, alphas)
        S['unseen'][Path(d).name] = dict(candidates=cands, result=res)
    # pooled held-out over seeds (sum of counts)
    for tag in ('all', 'held_out'):
        agg = {}
        for v in S['unseen'].values():
            for name, r in v['result'][tag].items():
                g = agg.setdefault(name, collections.Counter())
                for k_, x in r.items():
                    if k_ != 'picks':
                        g[k_] += x
        pooled[tag] = {n: dict(r) for n, r in agg.items()}
    S['pooled'] = pooled

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'stage2b_summary.json').write_text(json.dumps(S, indent=2, default=str) + '\n')
    (a.out / 'stage2b_summary.md').write_text(markdown(S))
    try:
        figures(S, a.out)
    except Exception as ex:
        print('figures failed:', ex, file=sys.stderr)
    print(f"solo alphas {S['solo_alphas']} -> {a.out}")


def markdown(S):
    f2 = lambda x: '-' if x is None else f'{x:.2f}'
    L = ['# Stage 2b: solver with / without, unseen seeds', '',
         f"Candidates {S['candidates']}; shared-card candidates {S['shared']} use solo profiles + k; the others use k = 1 with the seed-901 mixed run as profile. "
         f"Anchors {S['anchors']}; held-out alphas {S['held_out_alphas']}. Cost r = {S['r_price']}, occupied cards.", '',
         f"Solo profiles on seed 901: {S['solo_alphas']}", '',
         '## Measured k (mixed p90 / solo p90, seed 901)', '', '| candidate | alpha | k TTFT | k TPOT | k DiT |', '|---|---|---|---|---|']
    for r in S['k_measured']:
        L.append(f"| {r['candidate']} | {r['alpha']} | {f2(r['k']['ttft'])} | {f2(r['k']['tpot'])} | {f2(r['k']['dit'])} |")
    L += ['', '## Two-anchor k models vs measured k (relative error, held-out alphas)', '', '| candidate | alpha | linear TTFT/TPOT/DiT | const TTFT/TPOT/DiT |', '|---|---|---|---|']
    for r in S['k_model_error']:
        L.append(f"| {r['candidate']} | {r['alpha']} | " + ' / '.join('-' if r['linear'][m] is None else f"{r['linear'][m]:+.0%}" for m in METRICS) + ' | '
                 + ' / '.join('-' if r['const'][m] is None else f"{r['const'][m]:+.0%}" for m in METRICS) + ' |')
    L += ['', '## Seed 901, held-out alphas, shared candidates: predicted cell vs mixed cell', '', 'fp = predicted pass, mixed fails; fn = predicted fail, mixed passes.', '',
          '| predictor | candidate | tp | fp | fn | tn |', '|---|---|---|---|---|---|']
    for name, per in S['confusion_901_heldout'].items():
        for c, t in per.items():
            L.append(f"| {name} | {c} | {t['tp']} | {t['fp']} | {t['fn']} | {t['tn']} |")

    def sel(res, title):
        out = ['', f'## {title}', '',
               '| selector | max-α cells | exact | regret Σα | overpromise | min-cost cases | ok | violation | overpay | missed | cost regret Σ | false-feasible |', '|---|---|---|---|---|---|---|---|---|---|---|---|']
        for n, r in res.items():
            out.append(f"| {n} | {r['cells']} | {r['exact']} | {r['regret_sum']:.2f} | {r['overpromise']} | {r['mc_cases']} | {r['mc_ok']} | {r['mc_violation']} | {r['mc_overpay']} | {r['mc_missed']} | {r['mc_cost_regret']:.2f} | {r['mc_false_feasible']} |")
        return out
    L += sel(S['pooled']['held_out'], f"Pooled over unseen seeds, held-out alphas {S['held_out_alphas']} (the fair test)")
    L += sel(S['pooled']['all'], f"Pooled over unseen seeds, all alphas {S['eval_alphas']}")
    for d, v in S['unseen'].items():
        for tag in ('held_out', 'anchors'):
            L += sel(v['result'][tag], f"{d}, {tag} alphas (candidates {', '.join(v['candidates'])})")
    return '\n'.join(L) + '\n'


def figures(S, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    # k vs alpha
    fig, axs = plt.subplots(len(SHARED), 3, figsize=(10, 2.6 * len(SHARED)), squeeze=False)
    for i, c in enumerate(SHARED):
        rows = [r for r in S['k_measured'] if r['candidate'] == c]
        for j, m in enumerate(METRICS):
            ax = axs[i, j]
            xs = [r['alpha'] for r in rows]; ys = [r['k'][m] or np.nan for r in rows]
            ax.plot(xs, ys, 'o', color='k', ms=4, label='measured')
            an = {r['alpha']: r['k'][m] for r in rows if r['alpha'] in ANCHORS}
            if len(an) == 2:
                (a0, y0), (a1, y1) = sorted(an.items())
                g = np.linspace(0.45, 1.55, 50)
                ax.plot(g, np.maximum(1, y0 + (y1 - y0) * (g - a0) / (a1 - a0)), '-', color='#D55E00', lw=1, label='linear, 2 anchors')
                ax.step(g, np.where(g <= 0.75, y0, y1), '--', color='#0072B2', lw=1, label='const, nearest anchor')
                ax.plot([a0, a1], [y0, y1], 'o', mfc='none', mec='#D55E00', ms=9)
            ax.axhline(1, color='k', lw=.5, ls=':'); ax.set_yscale('log'); ax.set_title(f'{LABEL[c]}: k {m.upper()}', fontsize=8)
            for xx, yy in zip(xs, ys):
                if xx not in ANCHORS and np.isfinite(yy):
                    ax.annotate(f'{yy:.2f}', (xx, yy), textcoords='offset points', xytext=(0, 5), ha='center', fontsize=6)
            ax.set_xlabel(r'$\alpha$', fontsize=8); ax.spines[['top', 'right']].set_visible(False)
    axs[0, 0].legend(fontsize=6, frameon=False)
    fig.suptitle('Interference ratio k = mixed p90 / resident-solo p90 (seed 901); circles mark the two anchors a solver would measure', fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, .95))
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(out / f'fig_stage2b_k.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)
    # selectors, held-out pooled + per seed
    blocks = [('pooled seeds 902+903', S['pooled']['held_out'])] + [(d, v['result']['held_out']) for d, v in S['unseen'].items()]
    fig, axs = plt.subplots(2, len(blocks), figsize=(4.2 * len(blocks), 5.4), squeeze=False)
    for j, (title, res) in enumerate(blocks):
        names = list(res); ax = axs[0, j]
        ax.bar(range(len(names)), [res[n]['exact'] / res[n]['cells'] if res[n]['cells'] else 0 for n in names], color=['#D55E00' if 'linear' in n else '#999999' if n.startswith('fixed') or n == 'solo only' else '#0072B2' for n in names])
        ax.set_ylim(0, 1.05); ax.set_title(f'{title}\nmax-α: share of cells matching oracle', fontsize=8); ax.set_xticks(range(len(names)), names, fontsize=6, rotation=30, ha='right')
        ax.spines[['top', 'right']].set_visible(False)
        ax = axs[1, j]; bottom = np.zeros(len(names))
        for key, lab, col in (('mc_ok', 'ok', '#009E73'), ('mc_overpay', 'ok but overpays', '#E69F00'), ('mc_violation', 'SLO violated in mixed', '#D55E00'), ('mc_missed', 'missed feasible', '#bbbbbb')):
            v = np.array([res[n][key] - (res[n]['mc_overpay'] if key == 'mc_ok' else 0) for n in names], dtype=float)
            ax.bar(range(len(names)), v, bottom=bottom, color=col, label=lab if j == 0 else None); bottom += v
        ax.set_title(f'min-cost: outcome per (cell, α), held-out α {HELD}', fontsize=8); ax.set_xticks(range(len(names)), names, fontsize=6, rotation=30, ha='right')
        ax.spines[['top', 'right']].set_visible(False)
    axs[1, 0].legend(fontsize=6, frameon=False, loc='lower left')
    fig.tight_layout()
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(out / f'fig_stage2b_selectors.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
