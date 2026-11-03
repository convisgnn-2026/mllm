#!/usr/bin/env python3
"""IEEE-style two-panel figure: (a) why P/D split helps (LLM only, no DiT traffic), (b) where the benefit survives
or disappears when DiT shares a GPU with one of the two stages (mixed, alpha 1.0). Reads per-request records.

  python3 bench/hetero/plot_ieee_pd.py [--out results/hetero/stage2b_report/fig_ieee_pd]
"""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOLO = ['results/hetero/stage2_solo_llm', 'results/hetero/stage2_solo_llm_b']
MIXED = ROOT / 'results/hetero/stage1_pilot'
STEM = {0.5: 'alpha0.5_seed901', 1.0: 'alpha1_seed901', 1.5: 'alpha1.5_seed901'}
C = dict(blue='#0072B2', orange='#D55E00', green='#009E73', grey='#777777', sky='#56B4E9')


def recs(case):
    return [json.loads(l) for l in (case / 'replay' / 'requests.live.jsonl').open() if l.strip()]


def pct(v, q):
    v = sorted(x for x in v if x is not None); return v[int(q * (len(v) - 1))]


def llm_stats(case):
    ok = [r for r in recs(case) if r['kind'] == 'llm' and r['measured'] and not r['err']]
    return dict(tpot90=pct([r['tpot'] for r in ok], .9) * 1000, tpot50=pct([r['tpot'] for r in ok], .5) * 1000, ttft90=pct([r['ttft'] for r in ok], .9))


def solo_case(cand, al):
    for d in SOLO:
        c = ROOT / d / cand / STEM[al]
        if (c / 'done').exists():
            return c
    return None


def style():
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Nimbus Roman', 'Liberation Serif'], 'mathtext.fontset': 'stix',
                         'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                         'axes.linewidth': 0.6, 'xtick.direction': 'in', 'ytick.direction': 'in', 'lines.linewidth': 1.0, 'lines.markersize': 4,
                         'legend.frameon': False, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, default=ROOT / 'results/hetero/stage2b_report/fig_ieee_pd')
    a = ap.parse_args(); style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.3), gridspec_kw=dict(width_ratios=[1, 1.15]))

    # (a) LLM only: TPOT p90 vs alpha
    series = [('S4_4090', 'Single engine, RTX 4090', C['grey'], 'o'), ('S3_4090', 'Single engine, RTX 3090', C['sky'], '^'), ('C_4090', 'P/D split (prefill 4090 → decode 3090)', C['orange'], 's')]
    for cand, lab, col, mk in series:
        xs, ys = [], []
        for al in (0.5, 1.0, 1.5):
            c = solo_case(cand, al)
            if c:
                xs.append(al); ys.append(llm_stats(c)['tpot90'])
        ax1.plot(xs, ys, marker=mk, color=col, label=lab)
        for x, y in zip(xs, ys):
            ax1.annotate(f'{y:.0f}', (x, y), textcoords='offset points', xytext=(0, -9 if cand == 'C_4090' else 4), ha='center', fontsize=6.5, color=col)
    ax1.set_xlabel(r'Demand multiplier $\alpha$ (LLM only, DiT idle)'); ax1.set_ylabel('LLM TPOT p90 (ms)'); ax1.set_xticks([0.5, 1.0, 1.5])
    ax1.set_ylim(0, 75); ax1.set_title('(a) Why split: P/D isolates decode')
    ax1.legend(loc='upper left', handlelength=1.6)

    # (b) mixed alpha 1.0: TPOT p90 for PD variants and the co-located reference
    bars = [('P/D split,\nLLM only', solo_case('C_4090', 1.0), C['orange'], ''),
            ('P/D split +\nDiT on prefill GPU', MIXED / 'C_4090' / STEM[1.0], C['orange'], ''),
            ('P/D split +\nDiT on decode GPU', MIXED / 'C_3090' / STEM[1.0], C['orange'], '///'),
            ('Single engine +\nDiT on same GPU', MIXED / 'S4_4090' / STEM[1.0], C['grey'], '///')]
    vals = [llm_stats(c)['tpot90'] for _, c, _, _ in bars]
    x = np.arange(len(bars))
    for i, ((lab, _, col, hatch), v) in enumerate(zip(bars, vals)):
        ax2.bar(i, v, width=0.6, color=col if not hatch else 'white', edgecolor=col, linewidth=1.0, hatch=hatch)
        ax2.text(i, v + 1.2, f'{v:.1f}', ha='center', fontsize=7)
    ax2.axhline(vals[0], color=C['orange'], lw=0.6, ls=':')
    ax2.set_xticks(x, [b[0] for b in bars]); ax2.set_ylabel('LLM TPOT p90 (ms)'); ax2.set_ylim(0, 62)
    ax2.set_title(r'(b) Mixed with DiT, $\alpha$ = 1.0: benefit kept or lost')
    ax2.annotate('decode GPU\nshared with DiT', xy=(2.3, vals[2]), xytext=(2.55, 42), fontsize=6.5, ha='center', arrowprops=dict(arrowstyle='-', lw=0.5))
    ax2.annotate('decode GPU\nnot shared', xy=(1.3, vals[1]), xytext=(1.45, 24), fontsize=6.5, ha='center', arrowprops=dict(arrowstyle='-', lw=0.5))
    fig.tight_layout(w_pad=1.5)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{a.out}.{ext}', dpi=300, bbox_inches='tight')
    print(a.out, [f'{v:.1f}' for v in vals])


if __name__ == '__main__':
    main()
