#!/usr/bin/env python3
"""Seed validation of the Stage 1 capacity map (seed 901 = pilot + refinement, 902/903 = validation).

  python3 bench/hetero/plot_stage1_seeds.py [--out results/hetero/stage1_validation]

For each candidate only the alphas tested on *every* seed are used (no interpolation, no assumed
failure above the largest tested alpha). Outputs:
  fig_seed_latency.png   p90 TTFT / TPOT / DiT E2E vs alpha, one line style per seed
  fig_seed_winners.png   per SLO cell: winner (largest common-grid alpha_max among the 4 candidates)
                         for each seed, and whether the three seeds agree
  seed_validation.md / .json  agreement counts, per-candidate alpha_max changes between seeds
"""
import argparse
import collections
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CANDS = ['S4_4090', 'S4_3090', 'S3_4090', 'C_4090']
LABEL = {'S4_4090': 'One GPU: 4090', 'S4_3090': 'LLM 4090 / DiT 3090', 'S3_4090': 'LLM 3090 / DiT 4090', 'C_4090': 'P/D split + DiT 4090'}
COLOR = {'S4_4090': '#777777', 'S4_3090': '#0072B2', 'S3_4090': '#56B4E9', 'C_4090': '#D55E00'}
SEEDS = {901: ['results/hetero/stage1_pilot', 'results/hetero/stage1_refine'], 902: ['results/hetero/stage1_seed902'], 903: ['results/hetero/stage1_seed903']}
LS = {901: '-', 902: '--', 903: ':'}
COST = {'S4_4090': 1.0, 'S4_3090': 1.75, 'S3_4090': 1.75, 'C_4090': 1.75}  # r = 0.75, occupied cards


def load(dirs):
    runs, cells = [], []
    for d in dirs:
        d = ROOT / d
        if not (d / 'stage1_summary.json').exists():
            continue
        runs += [r for r in json.loads((d / 'stage1_summary.json').read_text())['runs'] if r['state'] == 'done' and r.get('valid') and r['candidate'] in CANDS]
        cells += [r for r in csv.DictReader((d / 'stage1_cells.csv').open()) if r['valid'] == 'True' and r['candidate'] in CANDS]
    return runs, cells


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, default=ROOT / 'results/hetero/stage1_validation')
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    data = {s: load(d) for s, d in SEEDS.items()}
    data = {s: v for s, v in data.items() if v[0]}
    seeds = sorted(data)
    tested = {s: collections.defaultdict(set) for s in seeds}
    for s in seeds:
        for r in data[s][0]:
            tested[s][r['candidate']].add(r['alpha'])
    common = {c: sorted(set.intersection(*[tested[s][c] for s in seeds])) if all(tested[s][c] for s in seeds) else [] for c in CANDS}

    # latency figure
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.0))
    for s in seeds:
        for c in CANDS:
            rs = sorted([r for r in data[s][0] if r['candidate'] == c], key=lambda r: r['alpha'])
            if not rs:
                continue
            for ax, (svc, key, sc) in zip(axs, [('llm', 'ttft_p90', 1), ('llm', 'tpot_p90', 1000), ('dit', 'e2e_p90', 1)]):
                ax.plot([r['alpha'] for r in rs], [r['score']['services'][svc]['success_only'][key] * sc for r in rs], color=COLOR[c], ls=LS[s], marker='o', ms=2.5,
                        label=f'{LABEL[c]}' if s == seeds[0] else None)
    for ax, t, yl in zip(axs, ['LLM TTFT p90', 'LLM TPOT p90', 'DiT E2E p90'], ['s', 'ms', 's']):
        ax.set_yscale('log'); ax.set_title(t, fontsize=8); ax.set_ylabel(yl); ax.set_xlabel(r'demand multiplier $\alpha$'); ax.grid(axis='y', alpha=.2); ax.spines[['top', 'right']].set_visible(False)
    h, l = axs[0].get_legend_handles_labels()
    h += [plt.Line2D([], [], color='k', ls=LS[s]) for s in seeds]; l += [f'seed {s}' for s in seeds]
    fig.legend(h, l, loc='upper center', ncol=len(h), frameon=False, fontsize=7, bbox_to_anchor=(.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, .9))
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(a.out / f'fig_seed_latency.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)

    # alpha_max on the common grid
    am = {}
    keys = sorted({(r['ttft_s'], r['tpot_s'], r['dit_e2e_s'], r['target']) for s in seeds for r in data[s][1]}, key=lambda k: tuple(map(float, k)))
    for s in seeds:
        for r in data[s][1]:
            al = float(r['alpha']); c = r['candidate']; k = (r['ttft_s'], r['tpot_s'], r['dit_e2e_s'], r['target'])
            if al in common[c]:
                am.setdefault((s, c, k), 0.)
                if r['pass'] == 'True':
                    am[(s, c, k)] = max(am[(s, c, k)], al)
    def winner(s, k, by_cost=False):
        v = {c: am.get((s, c, k), 0.) / (COST[c] if by_cost else 1) for c in CANDS if common[c]}
        b = max(v.values()) if v else 0
        if b == 0:
            return 'none'
        ws = [c for c in v if abs(v[c] - b) < 1e-9]
        return ws[0] if len(ws) == 1 else 'tie:' + '+'.join(ws)
    res = {}
    for by_cost in (False, True):
        W = {k: tuple(winner(s, k, by_cost) for s in seeds) for k in keys}
        agree = sum(1 for w in W.values() if len(set(w)) == 1)
        agree_nonempty = sum(1 for w in W.values() if len(set(w)) == 1 and w[0] != 'none')
        any_pass = sum(1 for w in W.values() if any(x != 'none' for x in w))
        changes = collections.Counter(w for w in W.values() if len(set(w)) > 1)
        res['cost' if by_cost else 'alpha'] = dict(cells=len(keys), agree_all_seeds=agree, agree_with_a_winner=agree_nonempty, cells_with_any_winner=any_pass,
                                                    top_disagreements=[dict(winners=dict(zip(map(str, seeds), w)), n=n) for w, n in changes.most_common(10)])
        res['W_' + ('cost' if by_cost else 'alpha')] = W
    shifts = {}
    for c in CANDS:
        if not common[c]:
            continue
        d = collections.Counter()
        for k in keys:
            vals = tuple(am.get((s, c, k), 0.) for s in seeds)
            d['same' if len(set(vals)) == 1 else 'differs'] += 1
        shifts[c] = dict(common_alphas=common[c], **d)
    res['alpha_max_stability'] = shifts

    # winners figure
    T = sorted({k[0] for k in keys}, key=float); P_ = sorted({k[1] for k in keys}, key=float); D = sorted({k[2] for k in keys}, key=float); G = sorted({k[3] for k in keys}, key=float)
    rows = [(t, p) for t in T for p in P_]; cols = [(d, g) for d in D for g in G]
    pal = {c: COLOR[c] for c in CANDS}; pal['none'] = '#ffffff'
    fig, axs = plt.subplots(2, len(seeds) + 1, figsize=(4.2 * (len(seeds) + 1), 9.5))
    for row, mode in enumerate(['alpha', 'cost']):
        W = res['W_' + mode]
        for j, s in enumerate(seeds):
            ax = axs[row, j]
            img = np.ones((len(rows), len(cols), 3))
            for i, (t, p) in enumerate(rows):
                for jj, (d, g) in enumerate(cols):
                    w = W[(t, p, d, g)][j]
                    img[i, jj] = matplotlib.colors.to_rgb('#f0e442' if w.startswith('tie') else pal[w])
            ax.imshow(img, aspect='auto'); ax.set_title(f'seed {s}: winner by ' + ('α_max' if mode == 'alpha' else 'α_max / cost (r=0.75)'), fontsize=8)
            ax.set_xticks(range(len(cols)), [f'{float(d):g}s\n{int(float(g) * 100)}%' for d, g in cols], fontsize=5)
            ax.set_yticks(range(len(rows)), [f'TTFT {float(t):g} / TPOT {int(float(p) * 1000)}' for t, p in rows] if j == 0 else [], fontsize=5)
            for x in range(len(G), len(cols), len(G)):
                ax.axvline(x - .5, color='k', lw=.3)
            for y in range(len(P_), len(rows), len(P_)):
                ax.axhline(y - .5, color='k', lw=.3)
        ax = axs[row, -1]
        img = np.ones((len(rows), len(cols), 3))
        for i, (t, p) in enumerate(rows):
            for jj, (d, g) in enumerate(cols):
                w = W[(t, p, d, g)]
                img[i, jj] = matplotlib.colors.to_rgb('#ffffff' if all(x == 'none' for x in w) else '#009E73' if len(set(w)) == 1 else '#CC79A7')
        ax.imshow(img, aspect='auto'); r = res[mode]
        ax.set_title(f'agreement across seeds: {r["agree_with_a_winner"]}/{r["cells_with_any_winner"]} cells\n(green = same winner, pink = differs)', fontsize=8)
        ax.set_xticks(range(len(cols)), [f'{float(d):g}s\n{int(float(g) * 100)}%' for d, g in cols], fontsize=5); ax.set_yticks([])
        for x in range(len(G), len(cols), len(G)):
            ax.axvline(x - .5, color='k', lw=.3)
        for y in range(len(P_), len(rows), len(P_)):
            ax.axhline(y - .5, color='k', lw=.3)
    handles = [Patch(color=COLOR[c], label=LABEL[c]) for c in CANDS] + [Patch(color='#f0e442', label='tie'), Patch(facecolor='white', edgecolor='k', label='no candidate passes')]
    fig.legend(handles=handles, loc='lower center', ncol=6, frameon=False, fontsize=7)
    fig.suptitle('Seed validation: winner per SLO cell among 4 candidates, alphas tested on every seed only ('
                 + '; '.join(f'{c}: {",".join(f"{x:g}" for x in common[c])}' for c in CANDS if common[c]) + ')', fontsize=8)
    fig.tight_layout(rect=(0, .03, 1, .97))
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(a.out / f'fig_seed_winners.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)

    out = {k: v for k, v in res.items() if not k.startswith('W_')}
    out['seeds'] = seeds; out['common_alphas'] = common
    (a.out / 'seed_validation.json').write_text(json.dumps(out, indent=2) + '\n')
    L = ['# Seed validation (Stage 1)', '', f'seeds {seeds}; candidates {CANDS}; alphas common to all seeds per candidate: ' + '; '.join(f'{c} {common[c]}' for c in CANDS), '',
         '| objective | cells with a winner on some seed | same winner on all seeds |', '|---|---|---|']
    for m in ('alpha', 'cost'):
        r = res[m]; L.append(f"| {'α_max' if m == 'alpha' else 'α_max / cost (r=0.75)'} | {r['cells_with_any_winner']} | {r['agree_with_a_winner']} |")
    L += ['', '| candidate | cells with identical α_max on all seeds | cells that differ |', '|---|---|---|']
    for c, v in shifts.items():
        L.append(f"| {c} | {v.get('same', 0)} | {v.get('differs', 0)} |")
    L += ['', 'Most frequent disagreements (α_max objective):', '']
    for d in res['alpha']['top_disagreements']:
        L.append(f"- {d['winners']}: {d['n']} cells")
    (a.out / 'seed_validation.md').write_text('\n'.join(L) + '\n')
    print(f'seeds {seeds}; common {common} -> {a.out}')


if __name__ == '__main__':
    main()
