#!/usr/bin/env python
"""
Fairness frontier: aggregate SAR vs worst-class SAR, four queue policies,
Full GPUs vs 2-way MIG partition, same trace, same load.

The point of the figure: the four Full points trace one trade-off curve
(pay aggregate to buy the worst class); partition moves every policy off
that curve, outward.  Policies move *along* a frontier; partition moves
*the frontier*.

Reads results/fair/{2F,2x2way}_{fcfs,sjf,edf,vtc}_r<rate>.json.
"""
import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != "/" and not _os.path.isdir(_os.path.join(_d, "results")):
    _d = _os.path.dirname(_d)
_os.chdir(_d)

import argparse
import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

USER = {"480x272x33": "A", "832x480x33": "B", "832x480x49": "C", "1280x720x49": "D"}
POLICIES = ["fcfs", "sjf", "edf", "vtc"]
CFG = {"2F": "Full ×2", "2x2way": "2-way ×2"}
# reference palette, slots 1 and 2 (validated adjacent pair)
COLOR = {"2F": "#2a78d6", "2x2way": "#eb6834"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
CAP_2F = 0.687


def score(path, slo):
    d = json.load(open(path))
    per = defaultdict(list)
    for r in d["requests"]:
        per[USER[r["tpl"]]].append((r["end_s"] - r["arrival_s"]) <= slo[r["tpl"]])
    sar = {k: sum(v) / len(v) for k, v in per.items()}
    n = sum(len(v) for v in per.values())
    agg = sum(sum(v) for v in per.values()) / n
    worst = min(sar, key=sar.get)
    return agg, sar[worst], worst, len(per["D"])


def hull_upper_right(pts):
    """Points not dominated by any other (higher x and higher y)."""
    keep = []
    for i, (x, y) in enumerate(pts):
        dominated = any(x2 >= x and y2 >= y and (x2 > x or y2 > y)
                        for j, (x2, y2) in enumerate(pts) if j != i)
        if not dominated:
            keep.append((x, y))
    return sorted(keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", default="0.85")
    ap.add_argument("--slo", default="video_3.0x")
    ap.add_argument("-o", "--out", default="results/fig_frontier.png")
    a = ap.parse_args()
    slo = {k: float(v) for k, v in json.load(open(f"slo_configs/{a.slo}.json")).items()}

    P = {}
    for cfg in CFG:
        for q in POLICIES:
            P[(cfg, q)] = score(f"results/fair/{cfg}_{q}_r{a.rate}.json", slo)
    n_d = P[("2F", "sjf")][3]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2})
    fig, ax = plt.subplots(figsize=(6.4, 5.0), dpi=160)
    ax.set_facecolor("#fcfcfb"); fig.patch.set_facecolor("#fcfcfb")

    # arrows Full -> 2-way, one per policy (drawn first so marks sit on top)
    for q in POLICIES:
        x0, y0 = P[("2F", q)][:2]
        x1, y1 = P[("2x2way", q)][:2]
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#b3b2ad",
                                    shrinkA=6, shrinkB=6))

    # dashed frontier through the non-dominated points of each config
    for cfg in CFG:
        pts = [P[(cfg, q)][:2] for q in POLICIES]
        h = hull_upper_right(pts)
        ax.plot([x for x, _ in h], [y for _, y in h], ls="--", lw=1.4,
                color=COLOR[cfg], alpha=0.75, zorder=2)

    # marks + direct labels
    OFF = {("2F", "fcfs"): (6, -10), ("2F", "sjf"): (6, -10),
           ("2F", "edf"): (6, -10), ("2F", "vtc"): (-8, 8),
           ("2x2way", "fcfs"): (6, 6), ("2x2way", "sjf"): (6, -4),
           ("2x2way", "edf"): (6, 4), ("2x2way", "vtc"): (-8, 8)}
    for cfg in CFG:
        for q in POLICIES:
            x, y = P[(cfg, q)][:2]
            ax.scatter(x, y, s=64, color=COLOR[cfg], edgecolor="#fcfcfb",
                       linewidth=1.5, zorder=4)
            dx, dy = OFF[(cfg, q)]
            ax.annotate(q, (x, y), xytext=(dx, dy), textcoords="offset points",
                        fontsize=9.5, color=INK, zorder=5,
                        ha="right" if dx < 0 else "left")

    ax.set_xlim(0.15, 0.95); ax.set_ylim(0.05, 0.56)
    ax.set_xlabel("Aggregate SAR  (all 200 requests)")
    ax.set_ylabel("Worst-class SAR  (min over 4 classes)")
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # legend: two series
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="--", color=COLOR[c], markersize=7,
                      label=CFG[c]) for c in CFG]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=9.5,
              title="Same 2 GPUs, same trace", title_fontsize=9)

    ax.text(0.98, 0.02,
            "load %d%% of Full capacity  ·  SLO %s\nworst class is 1280p (n = %d) except fcfs (480p)  ·  one y-tick = 1/%d = %.3f"
            % (round(float(a.rate) / CAP_2F * 100), a.slo.split("_")[1], n_d,
                                       n_d, 1 / n_d),
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color=INK2)
    ax.set_title("Partition moves the whole fairness frontier, not a point on it",
                 fontsize=11, color=INK, loc="left", pad=10)

    fig.tight_layout()
    fig.savefig(a.out)
    print(f"wrote {a.out}")
    for cfg in CFG:
        for q in POLICIES:
            x, y, w, _ = P[(cfg, q)]
            print(f"  {CFG[cfg]:>9} {q:>5}  agg {x:.2f}  worst {y:.2f} [{w}]")


if __name__ == "__main__":
    main()
