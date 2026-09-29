#!/usr/bin/env python3
"""SLO 축을 포함한 결정 지도.
  fig6_decision_map.png   n 별 패널: 행 = 프리필 비율, 열 = TPOT SLO(50/75/100/none), 칸 색 = goodput 1위 구성, 칸 글자 = 값과 2위 대비 배수
  fig7_goodput_vs_slo.png 대표 셀 3개: x = TPOT SLO 임계값(30~150 ms), y = goodput, 구성별 선
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_goodput as G

CONFIGS, COLOR, LABEL = G.CONFIGS, G.COLOR, G.LABEL
INK, INK2, GRID, SURF = G.INK, G.INK2, G.GRID, G.SURF
SLOS = [50, 75, 100, None]
SLO_LABEL = ["≤50 ms", "≤75 ms", "≤100 ms", "no SLO"]


def goodput(d, key, T):
    v = d.get(key)
    if v is None:
        return None
    ok = len(v["tpot"]) if T is None else sum(1 for t in v["tpot"] if t <= T)
    return ok / v["makespan"]


def decision_map(d):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6)); fig.patch.set_facecolor(SURF)
    for ax, n in zip(axes, G.NS):
        ax.set_facecolor(SURF); ax.set_title(f"n = {n}", loc="left", fontsize=11, color=INK, pad=8)
        rows = [(name, frac) for name, frac in G.MIX_ORDER]
        for i, (name, frac) in enumerate(rows):
            for j, T in enumerate(SLOS):
                vals = {c: goodput(d, (c, name, n), T) for c in CONFIGS}
                vals = {c: v for c, v in vals.items() if v is not None}
                if not vals:
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, color=GRID)); ax.text(j + 0.5, i + 0.5, "–", ha="center", va="center", color=INK2); continue
                best = max(vals, key=vals.get); ranked = sorted(vals.values(), reverse=True)
                margin = ranked[0] / ranked[1] if len(ranked) > 1 and ranked[1] > 0 else float("inf")
                ax.add_patch(plt.Rectangle((j, i), 1, 1, color=COLOR[best], alpha=0.9, ec=SURF, lw=2))
                txt = f"{best}\n{ranked[0]:.2f}" + (f"\n×{margin:.1f}" if margin < 99 else "")
                ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center", fontsize=8, color="white" if best != "D" else INK)
        ax.set_xlim(0, len(SLOS)); ax.set_ylim(0, len(rows)); ax.invert_yaxis()
        ax.set_xticks([j + 0.5 for j in range(len(SLOS))]); ax.set_xticklabels(SLO_LABEL, fontsize=9, color=INK2)
        ax.set_yticks([i + 0.5 for i in range(len(rows))]); ax.set_yticklabels([f"{f:.0%} prefill" for _, f in rows], fontsize=9, color=INK2)
        ax.set_xlabel("TPOT SLO", color=INK2, fontsize=9); ax.tick_params(length=0)
        for s in ax.spines.values(): s.set_visible(False)
    axes[0].set_ylabel("workload mix (share of prefill-heavy)", color=INK2, fontsize=9)
    handles = [Patch(color=COLOR[c], label=LABEL[c]) for c in CONFIGS]
    fig.legend(handles=handles, frameon=False, fontsize=9, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Winning configuration by workload mix × TPOT SLO  (cell = winner, goodput req/s, margin over runner-up)", x=0.01, ha="left", fontsize=10, color=INK2)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95)); p = os.path.join(G.FIG, "fig6_decision_map.png"); fig.savefig(p, dpi=160); plt.close(fig); return p


def goodput_vs_slo(d):
    cells = [("prefill", 200, "Prefill 100%, n=200"), ("p75", 200, "Prefill 75%, n=200"), ("decode", 100, "Decode 100%, n=100")]
    Ts = list(range(30, 155, 5))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True); fig.patch.set_facecolor(SURF)
    for ax, (name, n, title) in zip(axes, cells):
        ax.set_facecolor(SURF); ax.set_title(title, loc="left", fontsize=11, color=INK, pad=8)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=9); ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
        for c in CONFIGS:
            ys = [goodput(d, (c, name, n), T) for T in Ts]
            if ys[0] is None:
                continue
            ax.plot(Ts, ys, "-", color=COLOR[c], lw=2, label=LABEL[c])
            ax.text(Ts[-1] + 2, ys[-1], c, color=INK2, fontsize=8, va="center")
        for T in (50, 75, 100):
            ax.axvline(T, color=GRID, lw=1, ls="--")
        ax.set_xlabel("TPOT SLO threshold (ms)", color=INK2, fontsize=9)
    axes[0].set_ylabel("goodput (req/s meeting SLO)", color=INK2, fontsize=9)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=9, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.06, 1, 1)); p = os.path.join(G.FIG, "fig7_goodput_vs_slo.png"); fig.savefig(p, dpi=160); plt.close(fig); return p


if __name__ == "__main__":
    d = G.load()
    print(decision_map(d)); print(goodput_vs_slo(d))
