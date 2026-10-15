#!/usr/bin/env python3
"""Splitwise 식 iso-demand 비용 비교, "3090 은 이미 있고 4090 을 새로 산다" 관점.
수요(워크로드·도착률·Scale)마다 SLO(SAR ≥ 90%)를 지키는 구성 중 '새로 사야 하는 4090 수'가 가장 적은 것, 같으면 전력이 적은 것.
구성: S3(구매 0) · S4(4090 1장, 3090 은 놀림) · A/B/C/D(4090 1장 + 기존 3090) · S4x2(4090 2장, 추정: S4 의 λ/2 결과).
지표: 구매 수, 전력, SLO 만족 요청당 Wh.   환경변수 TTFT_FLOOR / TPOT_MULT / MAP_TAG 공유."""
import json, math, os, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from map_score import cell, ROOT, WL, RATES, SCALES, SAR_MIN, TPOT_MULT, TAG

CFGS = ["S3", "S4", "A", "B", "C", "D", "S4x2"]
BUY = {"S3": 0, "S4": 1, "A": 1, "B": 1, "C": 1, "D": 1, "S4x2": 2}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
BCOLOR = {0: "#8e8e8e", 1: "#2a78d6", 2: "#c0392b"}; BLABEL = {0: "buy nothing (3090 only)", 1: "buy one 4090 (alone, or with the existing 3090)", 2: "buy two 4090s (estimated from 4090-only at λ/2)"}
YL = {"code": "prefill-heavy (code trace)", "conv": "mixed (conv trace)", "decode": "decode-heavy (synthetic)"}


def get(c, w, r):
    if c != "S4x2": return cell(c, w, r)
    half = float(r) / 2; grid = [float(x) for x in RATES]; up = min([g for g in grid if g >= half - 1e-9], default=None)
    if up is None: return None
    x = cell("S4", w, f"{up:.1f}")
    if not x: return None
    return dict(thr=2 * x["thr"], sar=x["sar"], w=2 * x.get("w", 0), est=True)


def main():
    D = {(c, w, r): get(c, w, r) for c in CFGS for w in WL for r in RATES}
    L = [f"# iso-demand 비용 비교 — 3090 보유, 4090 신규 구매 관점 (SLO: TTFT 2 / TPOT {TPOT_MULT} × Scale, 하한 {os.environ.get('TTFT_FLOOR','0')} s)", "",
         "칸 = 통과 구성 중 구매 수 최소 → 전력 최소. S4x2 는 추정(4090 단독의 λ/2 결과, 보수적 올림).", ""]
    fig, axes = plt.subplots(1, len(SCALES), figsize=(16, 4.8), sharey=True); fig.patch.set_facecolor(SURF)
    for k, sc in enumerate(SCALES):
        ax = axes[k]; ax.set_facecolor(SURF); ax.set_xlim(-0.5, len(RATES) - 0.5); ax.set_ylim(-0.5, len(WL) - 0.5)
        for s_ in ax.spines.values(): s_.set_visible(False)
        ax.set_xticks(range(len(RATES))); ax.set_xticklabels(RATES, fontsize=9, color=INK2); ax.tick_params(length=0)
        ax.set_yticks(range(len(WL))); ax.set_yticklabels([YL[w] for w in WL], fontsize=8.5, color=INK)
        ax.set_title(f"SLO scale ×{sc}", loc="left", fontsize=11, color=INK, pad=8); ax.set_xlabel("arrival rate (req/s)", fontsize=9, color=INK2)
        L += [f"## Scale ×{sc}", "", "| 워크로드 | 도착률 | 최선 | 구매 | W | Wh/SLO 요청 | 통과 구성 (구매수, W) |", "|---|---|---|---|---|---|---|"]
        for i, w in enumerate(WL):
            for j, r in enumerate(RATES):
                passed = [c for c in CFGS if D[(c, w, r)] and D[(c, w, r)]["sar"][sc] >= SAR_MIN]
                if not passed:
                    ax.add_patch(plt.Rectangle((j - 0.47, i - 0.44), 0.94, 0.88, facecolor="none", edgecolor=GRID, lw=1, hatch="///")); ax.text(j, i, "none", ha="center", va="center", fontsize=8, color=INK2)
                    L.append(f"| {w} | {r} | – | | | | |"); continue
                order = sorted(passed, key=lambda c: (BUY[c], D[(c, w, r)].get("w", 0))); best = order[0]; x = D[(best, w, r)]
                wh = (x.get("w", 0) * (1 / x["thr"] if x["thr"] else 0)) / 3600 / x["sar"][sc] if x["thr"] else 0
                ax.add_patch(plt.Rectangle((j - 0.47, i - 0.44), 0.94, 0.88, color=BCOLOR[BUY[best]], alpha=0.9, lw=0))
                ax.text(j, i + 0.12, best + ("*" if x.get("est") else ""), ha="center", va="center", fontsize=10, color="white", fontweight="bold")
                ax.text(j, i - 0.2, f"{x.get('w', 0):.0f} W", ha="center", va="center", fontsize=7.5, color="white")
                s4 = D[("S4", w, r)]; saved = best in "ABCD" and not (s4 and s4["sar"][sc] >= SAR_MIN)      # 기존 3090 덕에 4090 한 장을 덜 삼
                if saved: ax.add_patch(plt.Rectangle((j - 0.47, i - 0.44), 0.94, 0.88, facecolor="none", edgecolor="#1baf7a", lw=3))
                L.append(f"| {w} | {r} | {best}{'*' if x.get('est') else ''} | {BUY[best]} | {x.get('w', 0):.0f} | {wh:.3f} | " + ", ".join(f"{c}({BUY[c]}, {D[(c, w, r)].get('w', 0):.0f})" for c in order) + " |")
        L.append("")
    fig.legend(handles=[Patch(color=BCOLOR[b], label=BLABEL[b]) for b in BCOLOR] + [Patch(facecolor="none", edgecolor="#1baf7a", lw=3, label="existing 3090 saves a second 4090")], frameon=False, fontsize=9.5, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(f"What must be bought to meet the SLO (SAR ≥ 90%) when a 3090 is already owned — cell: config (* = estimated), power.  SLO = 3090-uncontended × (TTFT 2, TPOT {TPOT_MULT}) × scale", x=0.01, ha="left", fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94)); p = os.path.join(ROOT, f"fig_iso_cost{TAG}.png"); fig.savefig(p, dpi=160); print(p)
    open(os.path.join(ROOT, f"iso_cost{TAG}.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
