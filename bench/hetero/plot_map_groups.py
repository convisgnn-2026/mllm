#!/usr/bin/env python3
"""단일 GPU vs PD 분할이 드러나는 지도. 구성을 세 묶음으로 접는다: 단일(S3/S4) · 복제(A/B) · PD(C/D).
칸 = 가장 싼 통과 묶음(색) + 구체 구성 + 2위 묶음 대비 전력 절감. 환경변수 TTFT_FLOOR / TPOT_MULT / MAP_TAG 는 map_score 와 공유."""
import os, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from map_score import cell, ROOT, WL, RATES, SCALES, SAR_MIN, TPOT_MULT, TAG
from plot_map import cost_h, CFGS

GROUP = {"S3": "single", "S4": "single", "A": "replicate", "B": "replicate", "C": "PD", "D": "PD"}
GCOLOR = {"single": "#2a2a2a", "replicate": "#2a78d6", "PD": "#1baf7a"}
GLABEL = {"single": "single GPU (3090 or 4090)", "replicate": "two GPUs, replicated (A/B)", "PD": "two GPUs, prefill/decode split (C/D)"}
TITLE = {"code": "code trace — prefill-heavy (in 1.5K, out 13)", "conv": "conversation trace — mixed (in 1.0K, out 128)", "decode": "synthetic — decode-heavy (in 128, out 1024)"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


def main():
    D = {(c, w, r): cell(c, w, r) for c in CFGS for w in WL for r in RATES}
    fig, axes = plt.subplots(1, len(SCALES), figsize=(16, 5.0), sharey=True); fig.patch.set_facecolor(SURF)
    YL = {"code": "prefill-heavy\n(code trace: in 1.5K, out 13)", "conv": "mixed\n(conv trace: in 1.0K, out 128)", "decode": "decode-heavy\n(synthetic: in 128, out 1024)"}
    for k, sc in enumerate(SCALES):
        ax = axes[k]; ax.set_facecolor(SURF); ax.set_xlim(-0.5, len(RATES) - 0.5); ax.set_ylim(-0.5, len(WL) - 0.5)
        for s_ in ax.spines.values(): s_.set_visible(False)
        ax.set_xticks(range(len(RATES))); ax.set_xticklabels(RATES, fontsize=9, color=INK2); ax.tick_params(length=0)
        ax.set_yticks(range(len(WL))); ax.set_yticklabels([YL[w] for w in WL], fontsize=8.5, color=INK)
        ax.set_title(f"SLO scale ×{sc}" + ("  (tight)" if sc == 1 else "  (loose)" if sc == 3 else ""), loc="left", fontsize=11, color=INK, pad=8)
        ax.set_xlabel("arrival rate (req/s)", fontsize=9, color=INK2)
        for i, w in enumerate(WL):
            for j, r in enumerate(RATES):
                passed = sorted((cost_h(c, D[(c, w, r)]), c) for c in CFGS if D[(c, w, r)] and D[(c, w, r)]["sar"][sc] >= SAR_MIN)
                if not passed:
                    ax.add_patch(plt.Rectangle((j - 0.47, i - 0.44), 0.94, 0.88, facecolor="none", edgecolor=GRID, lw=1, hatch="///")); ax.text(j, i, "none", ha="center", va="center", fontsize=8, color=INK2); continue
                cost, best = passed[0]; g = GROUP[best]; other = [(cc, c) for cc, c in passed if GROUP[c] != g]
                ax.add_patch(plt.Rectangle((j - 0.47, i - 0.44), 0.94, 0.88, color=GCOLOR[g], alpha=0.9, lw=0))
                ax.text(j, i + 0.12, best, ha="center", va="center", fontsize=10, color="white", fontweight="bold")
                if other:
                    w_b = D[(best, w, r)].get("w", 0); w_o = D[(other[0][1], w, r)].get("w", 0)
                    ax.text(j, i - 0.2, f"{w_b - w_o:+.0f} W vs {other[0][1]}", ha="center", va="center", fontsize=7, color="white")
                else:
                    ax.text(j, i - 0.2, "only group", ha="center", va="center", fontsize=7, color="white")
    fig.legend(handles=[Patch(color=GCOLOR[g], label=GLABEL[g]) for g in GCOLOR], frameon=False, fontsize=9.5, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(f"Single GPU vs PD split — cheapest group meeting SLO (SAR ≥ 90%). 4090 + 3090, Llama-3.1-8B. SLO = 3090-uncontended × (TTFT 2, TPOT {TPOT_MULT}) × scale"
                 + (f", TTFT floor {float(os.environ.get('TTFT_FLOOR', 0))*1e3:.0f} ms" if os.environ.get("TTFT_FLOOR") else "") + ". Small text: power vs cheapest config of another group", x=0.01, ha="left", fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94)); p = os.path.join(ROOT, f"fig_map_groups{TAG}.png"); fig.savefig(p, dpi=160); print(p)


if __name__ == "__main__":
    main()
