#!/usr/bin/env python3
"""1단계 결정 지도 그림: 워크로드(행) × SLO Scale(열) × 도착률(칸) → 가장 싼 통과 구성 (SAR ≥ 90%).
비용 = 상각 $/h (4090 $1599, 3090 $1499, 4년) + 전력 요금(실측 W × PUE 1.3 × $0.12/kWh). 동률은 전력 요금으로 갈림."""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from map_score import cell, ROOT, WL, RATES, SCALES, SAR_MIN, TPOT_MULT, TAG

CFGS = ["S3", "S4", "A", "B", "C", "D"]
LABEL = {"S3": "3090 only", "S4": "4090 only", "A": "A  both, RR 1:1", "B": "B  both, weighted 2:1", "C": "C  PD 4090→3090", "D": "D  PD 3090→4090"}
COLOR = {"S3": "#8e8e8e", "S4": "#2a2a2a", "A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "D": "#eda100"}
AMORT = {"S3": 1499 / (4 * 8760), "S4": 1599 / (4 * 8760)}; AMORT.update({c: AMORT["S3"] + AMORT["S4"] for c in "ABCD"})
KWH, PUE = 0.12, 1.3
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


def cost_h(c, x):
    return AMORT[c] + (x.get("w", 0) / 1000) * PUE * KWH


def main():
    D = {(c, w, r): cell(c, w, r) for c in CFGS for w in WL for r in RATES}
    fig, axes = plt.subplots(len(WL), 1, figsize=(11, 6.2)); fig.patch.set_facecolor(SURF)
    rows = []
    for ax, w in zip(axes, WL):
        ax.set_facecolor(SURF); ax.set_xlim(-0.5, len(RATES) - 0.5); ax.set_ylim(-0.5, len(SCALES) - 0.5)
        for s in ax.spines.values(): s.set_visible(False)
        ax.set_yticks(range(len(SCALES))); ax.set_yticklabels([f"SLO ×{s}" for s in SCALES], fontsize=9, color=INK2)
        ax.set_xticks(range(len(RATES))); ax.set_xticklabels([f"{r}" for r in RATES], fontsize=9, color=INK2); ax.tick_params(length=0)
        ax.set_title({"code": "Azure code trace (long input, short output)", "conv": "Azure conversation trace", "decode": "synthetic decode-heavy (128 in / 1024 out)"}[w], loc="left", fontsize=10.5, color=INK, pad=6)
        for i, sc in enumerate(SCALES):
            for j, r in enumerate(RATES):
                cand = [(cost_h(c, D[(c, w, r)]), c) for c in CFGS if D[(c, w, r)] and D[(c, w, r)]["sar"][sc] >= SAR_MIN]
                if cand:
                    cost, best = min(cand); x = D[(best, w, r)]
                    ax.add_patch(plt.Rectangle((j - 0.48, i - 0.46), 0.96, 0.92, color=COLOR[best], alpha=0.9, lw=0))
                    ax.text(j, i, f"{best}\n{x['sar'][sc]*100:.0f}%  {x.get('w', 0):.0f} W", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
                    rows.append((w, sc, r, best, x["sar"][sc], x.get("w"), cost, [c for _, c in sorted(cand)]))
                else:
                    ax.add_patch(plt.Rectangle((j - 0.48, i - 0.46), 0.96, 0.92, facecolor="none", edgecolor=GRID, lw=1, hatch="///"))
                    ax.text(j, i, "none", ha="center", va="center", fontsize=8, color=INK2); rows.append((w, sc, r, None, None, None, None, []))
    axes[-1].set_xlabel("arrival rate (req/s)", color=INK2, fontsize=9.5)
    fig.legend(handles=[Patch(color=COLOR[c], label=LABEL[c]) for c in CFGS], frameon=False, fontsize=8.5, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(f"Cheapest configuration meeting the SLO (SAR ≥ 90%). 4090 + 3090, Llama-3.1-8B. SLO = 3090-uncontended × (TTFT 2, TPOT {TPOT_MULT}) × scale" + (f", TTFT ref floor {float(os.environ.get('TTFT_FLOOR',0))*1e3:.0f} ms" if os.environ.get("TTFT_FLOOR") else ""), x=0.01, ha="left", fontsize=9.5, color=INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96)); p = os.path.join(ROOT, f"fig_map{TAG}.png"); fig.savefig(p, dpi=160); print(p)
    L = ["# 결정 지도 — 가장 싼 통과 구성 (SAR ≥ 90%)", "", "비용/h = 상각(4년) + 전력 요금(실측 W × PUE 1.3 × $0.12/kWh). 통과 구성은 싼 순.", "",
         "| 워크로드 | Scale | 도착률 | 최선 | SAR | W | $/h | 통과 구성 (싼 순) |", "|---|---|---|---|---|---|---|---|"]
    for w, sc, r, best, sar, wt, cost, order in rows:
        L.append(f"| {w} | ×{sc} | {r} | {best or '–'} | {sar*100:.0f}% | {wt:.0f} | {cost:.4f} | {' > '.join(order)} |" if best else f"| {w} | ×{sc} | {r} | – | | | | |")
    open(os.path.join(ROOT, f"map_decision{TAG}.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
