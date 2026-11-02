#!/usr/bin/env python3
"""Stage 1 capacity figures merged over several batch directories (pilot + refinement + KV controls).

  python3 bench/hetero/plot_stage1_map.py [--dirs results/hetero/stage1_pilot results/hetero/stage1_refine ...]
                                          [--kv-dirs results/hetero/stage1_kv2g results/hetero/stage1_kv8g]
                                          [--out results/hetero/stage1_pilot/figures]

Figures (all from stage1_summary.json / stage1_cells.csv written by stage1_report.py; nothing re-scored):
  fig_stage1_capacity   : (a-c) p90 latency vs alpha, all 7 candidates, grid thresholds as dotted lines;
                          (d) number of SLO cells passed vs alpha.
  fig_stage1_decision   : per SLO cell, the candidate(s) with the largest tested alpha_max (ties shown),
                          and the same with alpha_max divided by hourly cost at r = 0.75 (occupied cards).
  fig_stage1_kv_control : KV-size control runs (if present) next to the pilot runs they compare with.
alpha_max is the largest *tested* alpha that passed; untested alphas are not interpolated.
"""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CANDS = ['S4_4090', 'S3_3090', 'S4_3090', 'S3_4090', 'C_4090', 'C_3090', 'C_split']
LABEL = {'S4_4090': 'One GPU: 4090', 'S3_3090': 'One GPU: 3090', 'S4_3090': 'LLM 4090 / DiT 3090', 'S3_4090': 'LLM 3090 / DiT 4090',
         'C_4090': 'P/D split + DiT 4090', 'C_3090': 'P/D split + DiT 3090', 'C_split': 'P/D split + DiT split'}
COLOR = {'S4_4090': '#777777', 'S3_3090': '#bbbbbb', 'S4_3090': '#0072B2', 'S3_4090': '#56B4E9', 'C_4090': '#D55E00', 'C_3090': '#E69F00', 'C_split': '#009E73'}
MARK = {'S4_4090': 'o', 'S3_3090': 'o', 'S4_3090': 's', 'S3_4090': 's', 'C_4090': '^', 'C_3090': '^', 'C_split': 'D'}
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
                     'axes.linewidth': .6, 'lines.linewidth': 1.2, 'lines.markersize': 4, 'pdf.fonttype': 42})


def load(dirs):
    runs, cells, grid, price = [], [], None, None
    for d in dirs:
        d = Path(d)
        if not (d / 'stage1_summary.json').exists():
            continue
        S = json.loads((d / 'stage1_summary.json').read_text()); proto = json.loads((d / 'protocol.json').read_text())
        grid = grid or proto['slo_grid']; price = price or proto['price_model']
        for r in S['runs']:
            if r['state'] == 'done' and r.get('valid'):
                r['batch'] = d.name; runs.append(r)
        for row in csv.DictReader((d / 'stage1_cells.csv').open()):
            if row['valid'] == 'True':
                row['batch'] = d.name; cells.append(row)
    return runs, cells, grid, price


def alpha_max(cells):
    """(candidate, cell) -> largest tested passing alpha (0 if none); also set of tested alphas."""
    am, tested = {}, {}
    for r in cells:
        k = (r['candidate'], (r['ttft_s'], r['tpot_s'], r['dit_e2e_s'], r['target'])); a = float(r['alpha'])
        tested.setdefault(k, set()).add(a)
        if r['pass'] == 'True':
            am[k] = max(am.get(k, 0), a)
    return am, tested


def cost_of(c, r=0.75):
    from stage1_report import gpus_of, cost_rate
    return cost_rate(gpus_of(c), r, False)


def fig_capacity(runs, cells, grid, out):
    fig, axs = plt.subplots(1, 4, figsize=(11, 2.7))
    for c in CANDS:
        rs = sorted([r for r in runs if r['candidate'] == c], key=lambda r: r['alpha'])
        if not rs:
            continue
        a = [r['alpha'] for r in rs]
        for ax, (svc, key, scale) in zip(axs, [('llm', 'ttft_p90', 1), ('llm', 'tpot_p90', 1000), ('dit', 'e2e_p90', 1)]):
            ax.plot(a, [r['score']['services'][svc]['success_only'][key] * scale for r in rs], color=COLOR[c], marker=MARK[c], label=LABEL[c])
        npass = []
        for al in a:
            npass.append(sum(1 for x in cells if x['candidate'] == c and float(x['alpha']) == al and x['pass'] == 'True'))
        axs[3].plot(a, npass, color=COLOR[c], marker=MARK[c], label=LABEL[c])
    for ax, th in zip(axs, [grid['ttft_s'], [t * 1000 for t in grid['tpot_s']], grid['dit_e2e_s']]):
        for t in th:
            ax.axhline(t, color='k', ls=':', lw=.5, alpha=.5)
        ax.set_yscale('log')
    for ax, t, yl in zip(axs, ['(a) LLM TTFT p90', '(b) LLM TPOT p90', '(c) DiT E2E p90', '(d) SLO cells passed (of 192)'], ['s', 'ms', 's', 'cells']):
        ax.set_title(t, fontsize=8); ax.set_ylabel(yl); ax.set_xlabel(r'demand multiplier $\alpha$'); ax.grid(axis='y', alpha=.2); ax.spines[['top', 'right']].set_visible(False)
    axs[3].set_ylim(bottom=-2)
    fig.legend(*axs[0].get_legend_handles_labels(), loc='upper center', ncol=7, frameon=False, bbox_to_anchor=(.5, 1.04))
    fig.suptitle('Stage 1 (seed 901, 600 s, grid thresholds dotted): success-request p90 vs offered load', y=1.10, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, .93))
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(out / f'fig_stage1_capacity.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)


def fig_decision(cells, grid, out, r_price=0.75):
    am, tested = alpha_max(cells)
    rows = [(T, P) for T in grid['ttft_s'] for P in grid['tpot_s']]; cols = [(D, g) for D in grid['dit_e2e_s'] for g in grid['target']]
    idx = {c: i for i, c in enumerate(CANDS)}
    fig, axs = plt.subplots(1, 2, figsize=(12, 6.2))
    cmap = matplotlib.colors.ListedColormap(['white'] + [COLOR[c] for c in CANDS] + ['#f0e442'])
    for ax, mode in zip(axs, ['alpha_max', 'alpha_max / cost']):
        img = np.zeros((len(rows), len(cols))); txt = {}
        for i, (T, P) in enumerate(rows):
            for j, (D, g) in enumerate(cols):
                key = (str(T), str(P), str(D), str(g))
                val = {c: am.get((c, key), 0) / (cost_of(c, r_price) if mode != 'alpha_max' else 1) for c in CANDS}
                best = max(val.values())
                if best == 0:
                    img[i, j] = 0; txt[i, j] = ''; continue
                ws = [c for c in CANDS if abs(val[c] - best) < 1e-9]
                img[i, j] = idx[ws[0]] + 1 if len(ws) == 1 else len(CANDS) + 1
                a_best = max(am.get((c, key), 0) for c in ws)
                txt[i, j] = (f'{a_best:g}' if len(ws) == 1 else f'{a_best:g}\n×{len(ws)}')
        ax.imshow(img, cmap=cmap, vmin=0, vmax=len(CANDS) + 1, aspect='auto')
        for (i, j), s in txt.items():
            ax.text(j, i, s, ha='center', va='center', fontsize=6, color='white' if img[i, j] not in (0, len(CANDS) + 1) else 'black')
        ax.set_xticks(range(len(cols)), [f'{D:g}s\n{int(g*100)}%' for D, g in cols], fontsize=6)
        ax.set_yticks(range(len(rows)), [f'TTFT {T:g}s / TPOT {int(P*1000)}ms' for T, P in rows], fontsize=6)
        ax.set_xlabel('DiT E2E limit / attainment target'); ax.tick_params(length=0)
        for x in range(len(grid['target']), len(cols), len(grid['target'])):
            ax.axvline(x - .5, color='k', lw=.4)
        for y in range(len(grid['tpot_s']), len(rows), len(grid['tpot_s'])):
            ax.axhline(y - .5, color='k', lw=.4)
        ax.set_title('winner by tested ' + (r'$\alpha_{max}$' if mode == 'alpha_max' else rf'$\alpha_{{max}}$ / hourly cost (r = {r_price}, occupied cards)'), fontsize=9)
    handles = [Patch(color=COLOR[c], label=LABEL[c]) for c in CANDS] + [Patch(color='#f0e442', label='tie (×n candidates)'), Patch(facecolor='white', edgecolor='k', label='no candidate passes at any tested α')]
    fig.legend(handles=handles, loc='lower center', ncol=5, frameon=False, bbox_to_anchor=(.5, -.02))
    alphas = sorted({float(x['alpha']) for x in cells})
    fig.suptitle(f'Stage 1 decision map (seed 901; tested α = {", ".join(f"{a:g}" for a in alphas)}; cell text = winner α_max, censored at the largest tested α)', fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .96))
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(out / f'fig_stage1_decision.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)


def fig_kv(runs, kv_runs, out):
    if not kv_runs:
        return
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.0))
    groups = {}
    for r in kv_runs:
        groups.setdefault(r['candidate'], []).append(r)
    xs, labels, colors = [], [], []
    series = {k: [] for k in ('ttft', 'tpot', 'dit')}
    for c, rs in groups.items():
        base = [r for r in runs if r['candidate'] == c and r['alpha'] == rs[0]['alpha']]
        for r in base + rs:
            kv = r['resources'].get('kv_total_gib')
            labels.append(f"{kv:g} GiB\n({'pilot' if r['batch'] == 'stage1_pilot' else 'control'})"); colors.append(COLOR[c]); s = r['score']['services']
            series['ttft'].append(s['llm']['success_only']['ttft_p90']); series['tpot'].append(s['llm']['success_only']['tpot_p90'] * 1000); series['dit'].append(s['dit']['success_only']['e2e_p90'])
    x = np.arange(len(labels))
    for ax, (k, t, yl) in zip(axs, [('ttft', 'LLM TTFT p90', 's'), ('tpot', 'LLM TPOT p90', 'ms'), ('dit', 'DiT E2E p90', 's')]):
        ax.bar(x, series[k], color=colors)
        for xi, v in zip(x, series[k]):
            ax.text(xi, v, f'{v:.2f}' if k != 'tpot' else f'{v:.1f}', ha='center', va='bottom', fontsize=6)
        ax.set_xticks(x, labels, fontsize=6); ax.set_xlabel('KV cache total per candidate'); ax.set_title(t, fontsize=8); ax.set_ylabel(yl); ax.spines[['top', 'right']].set_visible(False)
    fig.legend(handles=[Patch(color=COLOR[c], label=LABEL[c]) for c in groups], loc='upper center', ncol=len(groups), frameon=False, bbox_to_anchor=(.5, .97))
    fig.suptitle(f'KV-size control at α = {kv_runs[0]["alpha"]:g} (pilot runs vs control runs; same plan, same seed)', fontsize=9, y=1.03)
    fig.tight_layout(rect=(0, 0, 1, .90))
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(out / f'fig_stage1_kv_control.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dirs', nargs='*', default=['results/hetero/stage1_pilot', 'results/hetero/stage1_refine'])
    ap.add_argument('--kv-dirs', nargs='*', default=['results/hetero/stage1_kv2g', 'results/hetero/stage1_kv8g'])
    ap.add_argument('--out', type=Path, default=ROOT / 'results/hetero/stage1_pilot/figures')
    a = ap.parse_args()
    import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
    runs, cells, grid, price = load([ROOT / d for d in a.dirs])
    kv_runs, _, _, _ = load([ROOT / d for d in a.kv_dirs])
    a.out.mkdir(parents=True, exist_ok=True)
    fig_capacity(runs, cells, grid, a.out); fig_decision(cells, grid, a.out); fig_kv(runs, kv_runs, a.out)
    print(f'{len(runs)} runs, {len(kv_runs)} KV-control runs, alphas {sorted({r["alpha"] for r in runs})} -> {a.out}')


if __name__ == '__main__':
    main()
