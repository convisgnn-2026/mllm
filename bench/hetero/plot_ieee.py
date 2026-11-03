#!/usr/bin/env python3
"""IEEE-style figures for the Stage 2b result (reads stage2b_summary.json only).

  python3 bench/hetero/plot_ieee.py [--summary results/hetero/stage2b_report/stage2b_summary.json] [--out DIR]

fig_ieee_k          : interference ratio k = mixed p90 / solo p90 vs demand multiplier, per metric;
                      two shared-card placements; two-anchor linear fit (alpha 0.5, 1.0).
fig_ieee_selectors  : (a) realised outcome of the min-cost choice on unseen seeds / held-out alphas,
                      (b) share of SLO cells where the max-alpha choice matches the oracle.
"""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ANCHORS = [0.5, 1.0]
METRICS = [('ttft', 'LLM TTFT'), ('tpot', 'LLM TPOT'), ('dit', 'DiT E2E')]
CAND = {'S4_4090': ('Co-located (one GPU)', 'o', 'k'), 'C_4090': ('P/D split, DiT on prefill GPU', 's', '0.45')}
SEL = [('solo only', 'Solo\nonly'), ('fixed:C_4090', 'Fixed\nP/D split'), ('fixed:S4_3090', 'Fixed\nseparate'),
       ('solo+k-linear (2 anchors)', 'Solver\n(2 anchors)'), ('oracle', 'Oracle')]
OUT = [('mc_ok', 'SLO met, cheapest', '0.3', ''), ('mc_overpay', 'SLO met, overpaid', '0.62', ''),
       ('mc_violation', 'SLO violated', '1.0', '///'), ('mc_missed', 'Feasible, declined', '0.88', '')]


def style():
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman', 'Nimbus Roman', 'Liberation Serif', 'STIX Two Text'],
        'mathtext.fontset': 'stix', 'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'legend.fontsize': 7,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.direction': 'in', 'ytick.direction': 'in', 'lines.linewidth': 0.9, 'lines.markersize': 3.5,
        'legend.frameon': False, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42})


def fig_k(S, out):
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 1.9))
    for ax, (m, title) in zip(axs, METRICS):
        for c, (lab, mk, col) in CAND.items():
            rows = sorted([r for r in S['k_measured'] if r['candidate'] == c], key=lambda r: r['alpha'])
            xs = np.array([r['alpha'] for r in rows]); ys = np.array([r['k'][m] or np.nan for r in rows])
            ax.plot(xs, ys, mk, color=col, mfc='white' if c == 'C_4090' else col, label=lab if m == 'ttft' else None)
            an = {r['alpha']: r['k'][m] for r in rows if r['alpha'] in ANCHORS}
            if len(an) == 2:
                (a0, y0), (a1, y1) = sorted(an.items()); g = np.linspace(0.5, 1.5, 50)
                ax.plot(g, np.maximum(1, y0 + (y1 - y0) * (g - a0) / (a1 - a0)), '-', color=col, lw=0.8, alpha=0.9,
                        label='Linear fit through anchors' if (m == 'ttft' and c == 'S4_4090') else None)
                ax.plot([a0, a1], [y0, y1], mk, color=col, mfc='none', ms=8, mew=0.8)
        ax.axhline(1, color='k', lw=0.5, ls=':'); ax.set_yscale('log'); ax.set_title(title)
        ax.set_xlabel(r'Demand multiplier $\alpha$'); ax.set_xticks([0.5, 0.75, 1.0, 1.25, 1.5])
        ax.axvspan(1.25, 1.55, color='0.93', zorder=0, lw=0)
    axs[0].set_ylabel(r'Interference ratio $k$ (mixed / solo p90)')
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(w_pad=1.0)
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'fig_ieee_k.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)


def fig_selectors(S, out):
    res = S['pooled']['held_out']
    names = [k for k, _ in SEL if k in res]; labels = [l for k, l in SEL if k in res]
    x = np.arange(len(names))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.2), gridspec_kw=dict(width_ratios=[1.25, 1]))
    # (a) min-cost outcome, fraction of cases
    n = res[names[0]]['mc_cases']; bottom = np.zeros(len(names))
    for key, lab, col, hatch in OUT:
        v = np.array([res[k][key] - (res[k]['mc_overpay'] if key == 'mc_ok' else 0) for k in names], float) / n
        ax1.bar(x, v, bottom=bottom, width=0.62, color=col, edgecolor='k', linewidth=0.5, hatch=hatch, label=lab)
        for i, (vi, bi) in enumerate(zip(v, bottom)):
            if vi >= 0.08:
                ax1.text(i, bi + vi / 2, f'{vi:.0%}', ha='center', va='center', fontsize=6.5, color='white' if col == '0.3' else 'k',
                         bbox=dict(fc='white', ec='none', pad=0.6) if hatch else None)
        bottom += v
    ax1.set_xticks(x, labels); ax1.set_ylim(0, 1); ax1.set_ylabel('Fraction of (SLO cell, $\\alpha$) cases')
    ax1.set_title(f'(a) Min-cost choice, realised outcome ($n$ = {n})')
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.28), ncol=4, handlelength=1.4, columnspacing=1.0)
    # (b) max-alpha exact share
    share = [res[k]['exact'] / res[k]['cells'] for k in names]
    ax2.bar(x, share, width=0.62, color='0.45', edgecolor='k', linewidth=0.5)
    for i, (k, s) in enumerate(zip(names, share)):
        ax2.text(i, s + 0.02, f"{res[k]['exact']}/{res[k]['cells']}", ha='center', fontsize=6.5)
    ax2.set_xticks(x, labels); ax2.set_ylim(0, 1.12); ax2.set_ylabel('Share of SLO cells')
    ax2.set_title('(b) Max-$\\alpha$ choice equals oracle')
    fig.tight_layout(w_pad=1.5)
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'fig_ieee_selectors.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--summary', type=Path, default=ROOT / 'results/hetero/stage2b_report/stage2b_summary.json')
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args(); out = a.out or a.summary.parent
    style(); S = json.loads(a.summary.read_text())
    fig_k(S, out); fig_selectors(S, out)
    print(out / 'fig_ieee_k.pdf', out / 'fig_ieee_selectors.pdf')


if __name__ == '__main__':
    main()
