#!/usr/bin/env python3
"""Timeline snippet from a profile_mixed_nsys.sh capture: decode graph executions and DiT kernels on the RTX 3090.

  python3 bench/hetero/plot_nsys_timeline.py results/hetero/nsys_c3090/mixed [--t0 30 --span 0.5] [--out FILE]

Two lanes (decode vLLM, dit_server); each box is one CUDA graph (decode) or one kernel (DiT) as recorded by CUPTI,
i.e. wall time from launch on the GPU to completion. Boxes of the two lanes overlapping in time means the two
contexts were resident at the same time (time-sliced or preempted), not that both executed simultaneously.
"""
import argparse
from pathlib import Path
import sqlite3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run', type=Path); ap.add_argument('--t0', type=float, default=30.0, help='offset into the capture, s')
    ap.add_argument('--span', type=float, default=0.5, help='window length, s'); ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args(); c = sqlite3.connect(a.run / 'report.sqlite')
    lo = c.execute('select min(start) from CUPTI_ACTIVITY_KIND_KERNEL').fetchone()[0]
    w0 = lo + int(a.t0 * 1e9); w1 = w0 + int(a.span * 1e9)
    dec = [r[0] for r in c.execute('select globalPid from CUPTI_ACTIVITY_KIND_GRAPH_TRACE group by globalPid')]
    dec = dec[0] if dec else None
    cnt = dict(c.execute('select globalPid, count(*) from CUPTI_ACTIVITY_KIND_KERNEL group by globalPid').fetchall())
    dit = max((g for g in cnt if g != dec), key=lambda g: cnt[g], default=None)
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Nimbus Roman', 'Liberation Serif'], 'font.size': 8})
    fig, ax = plt.subplots(figsize=(7.16, 1.6))
    lanes = []
    if dec is not None:
        g = c.execute('select start, end from CUPTI_ACTIVITY_KIND_GRAPH_TRACE where globalPid=? and end>? and start<?', (dec, w0, w1)).fetchall()
        k = c.execute('select start, end from CUPTI_ACTIVITY_KIND_KERNEL where globalPid=? and graphId is null and end>? and start<?', (dec, w0, w1)).fetchall()
        lanes.append(('Decode vLLM\n(CUDA graphs)', g, '#D55E00')); lanes.append(('Decode vLLM\n(other kernels)', k, '#E69F00'))
    if dit is not None:
        d = c.execute('select start, end from CUPTI_ACTIVITY_KIND_KERNEL where globalPid=? and end>? and start<?', (dit, w0, w1)).fetchall()
        lanes.append(('DiT server\n(kernels)', d, '#0072B2'))
    for i, (name, iv, col) in enumerate(lanes):
        y = len(lanes) - 1 - i
        for s, e in iv:
            ax.add_patch(Rectangle(((max(s, w0) - w0) / 1e6, y + 0.1), (min(e, w1) - max(s, w0)) / 1e6, 0.8, color=col, lw=0))
    ax.set_xlim(0, a.span * 1e3); ax.set_ylim(0, len(lanes)); ax.set_yticks([len(lanes) - 1 - i + 0.5 for i in range(len(lanes))], [n for n, _, _ in lanes], fontsize=7)
    ax.set_xlabel(f'time (ms), {a.span * 1e3:.0f} ms window at +{a.t0:.0f} s of the capture, {a.run.name}')
    ax.spines[['top', 'right']].set_visible(False); fig.tight_layout()
    out = a.out or a.run / f'timeline_{a.run.name}.png'
    fig.savefig(out, dpi=250, bbox_inches='tight'); print(out)


if __name__ == '__main__':
    main()
