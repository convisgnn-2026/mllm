#!/usr/bin/env python
"""
The threshold figure: where does slicing a GPU start to pay off?

Top row  -- SLO attainment vs SLO looseness, at a fixed 3-GPU budget.
            Three strategies: don't slice / slice well (2-way) / slice badly (7-way).
Bottom   -- the same comparison as a difference against the un-sliced baseline,
            for both a 3-GPU and a 2-GPU budget, so the zero crossing is explicit.

The point of the figure is that the crossing exists, and that it MOVES with the
workload -- which is why the choice has to be solved rather than fixed.
"""

import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != "/" and not _os.path.isdir(_os.path.join(_d, "results")):
    _d = _os.path.dirname(_d)
_sys.path[:0] = [_os.path.join(_d, "solver"), _os.path.join(_d, "bench")]
_os.chdir(_d)            # results/ and slo_configs/ are addressed from the repo root

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

for _f in ("/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Medium.ttc",
           "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Bold.ttc"):
    try:
        fm.fontManager.addfont(_f)
    except Exception:
        pass
plt.rcParams["font.family"] = "Noto Sans CJK KR"
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
from matplotlib.lines import Line2D

# validated categorical slots 1-3 (see dataviz references/palette.md)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8983"
SURF, GRID = "#fcfcfb", "#e6e5e0"

SLOS = [0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0]
KEYS = [f"video_{x}x" for x in SLOS]
NAME = {"C1_4full": "4F", "C1f_4full_fcfs": "4Ff", "C2_3full": "3F", "C3_2full": "2F",
        "C4_2full_1x2way": "2F+1x2", "C5_2full_1x7way": "2F+1x7",
        "C6_1full_1x2way": "1F+1x2", "A2_1gpu_2way": "0F+1x2", "A3_1gpu_7way": "0F+1x7"}


def load():
    s = json.load(open("results/scoresB.json"))
    out = {}
    for f, d in s.items():
        wl = "evenly" if "evenly" in f else "skewed"
        out[(wl, NAME[d["policy"]])] = {
            "sar": [d["scores"][k]["sar"] for k in KEYS],
            "ce": [d["scores"][k]["ce_req_per_gpu_h"] for k in KEYS],
            "gpu_s": d["gpu_seconds"], "n_gpus": d["n_gpus"]}
    return out


def crossing(x, d):
    """First x where d goes from <=0 to >0, linearly interpolated."""
    for i in range(len(d) - 1):
        if d[i] <= 0 < d[i + 1]:
            t = -d[i] / (d[i + 1] - d[i])
            lx = np.log(x[i]) + t * (np.log(x[i + 1]) - np.log(x[i]))
            return float(np.exp(lx))
        if d[i] == 0 and d[i + 1] >= 0 and i > 0 and d[i - 1] < 0:
            return x[i]
    return None


def style(ax, ylabel, ylim):
    ax.set_facecolor(SURF)
    ax.set_xscale("log")
    ax.set_xticks(SLOS)
    ax.set_xticklabels([f"{v:g}x" for v in SLOS], fontsize=9, color=INK2)
    ax.set_xlim(0.72, 6.4)
    ax.set_ylim(*ylim)
    ax.grid(True, which="major", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.set_ylabel(ylabel, fontsize=10, color=INK2)


def main():
    D = load()
    fig = plt.figure(figsize=(12.6, 9.4), facecolor=SURF)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.62], hspace=0.46,
                          wspace=0.14, top=0.845, bottom=0.10,
                          left=0.075, right=0.985)

    titles = {"evenly": "균등 분포  (네 크기가 25%씩)",
              "skewed": "큰 요청 편중  (1280p 40.5%)"}
    thresholds = []

    # ---------- top row: SAR at a fixed 3-GPU budget ----------
    for col, wl in enumerate(["evenly", "skewed"]):
        ax = fig.add_subplot(gs[0, col])
        style(ax, "SLO 달성률 (SAR)" if col == 0 else "", (0, 1.08))
        for key, c in [("3F", BLUE), ("2F+1x2", ORANGE), ("2F+1x7", AQUA)]:
            ax.plot(SLOS, D[(wl, key)]["sar"], color=c, lw=2, marker="o", ms=5.5,
                    mec=SURF, mew=1.4, zorder=4)

        d = [a - b for a, b in zip(D[(wl, "2F+1x2")]["sar"], D[(wl, "3F")]["sar"])]
        xc = crossing(SLOS, d)
        if xc:
            ax.axvline(xc, color=MUTED, lw=1.1, ls=(0, (4, 3)), zorder=2)
            ax.axvspan(xc, 6.4, color=ORANGE, alpha=0.05, zorder=1)
            ax.annotate(f"교차점 {xc:.2f}x", xy=(xc, 0.06),
                        xytext=(7, 0), textcoords="offset points",
                        fontsize=10, color=INK, fontweight="bold", va="bottom")
            # the shaded band is about the 2-way line only; 7-way never wins
            ax.text(np.sqrt(xc * 6.4), 1.028, "2분할이 통짜를 앞섬", ha="center",
                    fontsize=9.5, color=ORANGE, fontweight="bold")
        ax.set_title(titles[wl], fontsize=12, color=INK, pad=10, loc="left")
        ax.set_xlabel("SLO 여유도   (마감 = 기준 × 배수)", fontsize=10, color=INK2)

    for wl in ["evenly", "skewed"]:
        for key, base, lab in [("2F+1x2", "3F", "GPU 3장"), ("1F+1x2", "2F", "GPU 2장")]:
            d = [a - b for a, b in zip(D[(wl, key)]["sar"], D[(wl, base)]["sar"])]
            thresholds.append((wl, lab, crossing(SLOS, d), max(d)))

    # ---------- bottom: where each threshold lands, on one axis ----------
    ax = fig.add_subplot(gs[1, :])
    style(ax, "", (-0.7, 3.7))
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)

    xs = [t[2] for t in thresholds if t[2]]
    lo, hi = min(xs), max(xs)
    ax.axvspan(lo, hi, color=ORANGE, alpha=0.09, zorder=1)
    ax.axvspan(hi, 6.4, color=ORANGE, alpha=0.045, zorder=1)

    order = [("evenly", "GPU 3장"), ("evenly", "GPU 2장"),
             ("skewed", "GPU 3장"), ("skewed", "GPU 2장")]
    lut = {(w, b): (x, g) for w, b, x, g in thresholds}
    for i, (wl, bud) in enumerate(reversed(order)):
        x, gain = lut[(wl, bud)]
        ax.plot([0.72, x], [i, i], color=GRID, lw=1.6, zorder=2)
        ax.plot([x], [i], marker="o", ms=11, color=ORANGE, mec=SURF, mew=2, zorder=5)
        ax.annotate(f"{x:.2f}x", xy=(x, i), xytext=(0, 13),
                    textcoords="offset points", ha="center", fontsize=10.5,
                    color=INK, fontweight="bold")
        lbl = f"{titles[wl].split('  ')[0]} · {bud}"
        ax.annotate(lbl, xy=(0.735, i), xytext=(0, 12), textcoords="offset points",
                    ha="left", fontsize=10, color=INK2)

    ax.annotate(f"{lo:.2f}x  ~  {hi:.2f}x", xy=(np.sqrt(lo * hi), 3.45),
                ha="center", fontsize=10.5, color=ORANGE, fontweight="bold")
    ax.annotate("같은 하드웨어인데 임계점이 2.2배 범위로 움직인다\n"
                "→ 고정 규칙으로 정할 수 없다",
                xy=(np.sqrt(hi * 6.4), 2.6), ha="center", va="top",
                fontsize=10.5, color=INK, linespacing=1.5)
    ax.text(0.735, -0.45, "← 마감이 빡빡함", fontsize=9.5, color=INK2)
    ax.text(6.3, -0.45, "마감이 넉넉함 →", fontsize=9.5, color=INK2, ha="right")
    ax.set_xlabel("2분할이 통짜를 앞서기 시작하는 SLO 여유도", fontsize=10.5,
                  color=INK2)
    ax.set_title("임계점은 워크로드와 GPU 예산에 따라 이동한다",
                 fontsize=12, color=INK, pad=12, loc="left")

    handles = [Line2D([], [], color=c, lw=2, marker="o", ms=5.5, mec=SURF, mew=1.4)
               for c in (BLUE, ORANGE, AQUA)]
    labels = ["통짜 3장  (분할 안 함)", "통짜 2장 + 1장을 2분할  (잘 쪼갬)",
              "통짜 2장 + 1장을 7분할  (잘못 쪼갬)"]
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.918),
               ncol=3, frameon=False, fontsize=10.5, labelcolor=INK2,
               columnspacing=2.6, handletextpad=0.7)

    fig.suptitle("GPU를 쪼개는 것이 언제 이득이 되는가 — 같은 GPU 예산에서의 SLO 달성률",
                 fontsize=15, color=INK, y=0.972, fontweight="bold")
    fig.text(0.075, 0.022,
             "Wan2.1-T2V-1.3B · B300 · 요청 150개 · 감마 도착(CV 1.0) · "
             "통짜 3장 기준 부하 77~78% · 마감은 템플릿별 절대 초를 배수로 스윕",
             fontsize=8.5, color=MUTED)

    fig.savefig("results/fig_threshold.png", dpi=170, facecolor=SURF)
    print("wrote results/fig_threshold.png")
    print(f"\n{'workload':>10} {'budget':>9} {'임계점':>9} {'최대 이득':>9}")
    for wl, bud, x, g in thresholds:
        print(f"{wl:>10} {bud:>9} {x:>8.2f}x {g:>+9.3f}")


if __name__ == "__main__":
    main()
