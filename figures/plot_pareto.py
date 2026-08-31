#!/usr/bin/env python
"""
Cost-SAR Pareto frontier -- the figure that matches the objective.

`min resource s.t. SAR >= beta` is a question about the (cost, SAR) plane, so
that is where the figure belongs.  Holding the GPU count fixed and varying only
the partition barely moves SAR (simulated regret against the envelope: 0.002),
which is why the load-vs-SAR view has nothing for a solver to do.  Cost is where
the configurations actually separate.

What the frontier has to show to justify a solver:
  1. different configuration TYPES own different stretches of it
     (sliced at the cheap end, un-sliced at the expensive end), and
  2. common-sense choices sit visibly off it.

All points are measured (experiment B, 150 requests, evenly + skewed).
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
           "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Bold.ttc"):
    try:
        fm.fontManager.addfont(_f)
    except Exception:
        pass
plt.rcParams["font.family"] = "Noto Sans CJK KR"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8983"
SURF, GRID = "#fcfcfb", "#e6e5e0"

NAME = {"C1_4full": "Full 4장", "C2_3full": "Full 3장", "C3_2full": "Full 2장",
        "C4_2full_1x2way": "Full 2 + 2분할 1", "C5_2full_1x7way": "Full 2 + 7분할 1",
        "C6_1full_1x2way": "Full 1 + 2분할 1", "A2_1gpu_2way": "2분할 1장",
        "A3_1gpu_7way": "7분할 1장"}
SLICED = {"Full 2 + 2분할 1", "Full 2 + 7분할 1", "Full 1 + 2분할 1",
          "2분할 1장", "7분할 1장"}
# hand-tuned label offsets, to keep text off the frontier line
DXY = {"Full 2장": (34, -6), "2분할 1장": (0, 16), "Full 1 + 2분할 1": (-6, 15),
       "Full 3장": (0, 16), "Full 4장": (0, -22), "Full 2 + 2분할 1": (46, -6),
       "Full 2 + 7분할 1": (0, -22), "7분할 1장": (0, -22)}
SLO = "video_4.0x"


def pareto(pts):
    """Keep points not dominated on (low cost, high SAR)."""
    out = []
    for c, s, lab in sorted(pts):
        if not out or s > out[-1][1] + 1e-9:
            out.append((c, s, lab))
    return out


scores = json.load(open("results/scoresB.json"))
fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.9), facecolor=SURF)
fig.subplots_adjust(top=0.80, bottom=0.145, left=0.07, right=0.985, wspace=0.16)

for ax, wl, title in zip(axes, ["evenly", "skewed"],
                         ["균등 분포", "큰 요청 편중"]):
    pts = []
    for f, d in scores.items():
        if wl not in f or d["policy"] == "C1f_4full_fcfs":
            continue
        lab = NAME[d["policy"]]
        pts.append((d["gpu_seconds"] / 3600, d["scores"][SLO]["sar"], lab))

    front = pareto(pts)
    fx = [p[0] for p in front] + [max(p[0] for p in pts)]
    fy = [p[1] for p in front] + [front[-1][1]]
    ax.step(fx, fy, where="post", color=INK, lw=2.4, ls=(0, (1.6, 1.7)), zorder=3)

    fl = {p[2] for p in front}
    for c, s, lab in pts:
        on = lab in fl
        col = ORANGE if lab in SLICED else BLUE
        ax.scatter([c], [s], s=190 if on else 110, color=col,
                   edgecolor=SURF, linewidth=2, zorder=6 if on else 4,
                   alpha=1.0 if on else 0.45)
        ax.annotate(lab, xy=(c, s), xytext=DXY.get(lab, (0, -20)),
                    textcoords="offset points", ha="center", fontsize=10,
                    color=INK if on else MUTED,
                    fontweight="bold" if on else "normal")

    ax.set_title(title, fontsize=12.5, color=INK, pad=10, loc="left")
    ax.set_xlabel("비용  (GPU·시간)", fontsize=11.5, color=INK2, labelpad=7)
    ax.set_facecolor(SURF)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    for s_ in ("left", "bottom"):
        ax.spines[s_].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=10, length=0)
    ax.set_ylim(0.55, 1.06)

    print(f"\n[{wl}]  프론티어")
    for c, s, lab in front:
        kind = "분할" if lab in SLICED else "Full"
        print(f"   {c:.3f} GPU·h   SAR {s:.3f}   {lab}  ({kind})")
    dom = [(c, s, lab) for c, s, lab in pts if lab not in fl]
    for c, s, lab in sorted(dom, key=lambda x: -x[0])[:2]:
        cheaper = min((cc for cc, ss, _ in front if ss >= s), default=None)
        if cheaper:
            print(f"   [지배됨] {lab}: {c:.3f} GPU·h → 같은 SAR을 "
                  f"{cheaper:.3f} 로 달성 가능 ({(1-cheaper/c)*100:.0f}% 절감)")

axes[0].set_ylabel(f"SLO 달성률 (SAR)", fontsize=11.5, color=INK2)

from matplotlib.lines import Line2D
handles = [Line2D([], [], marker="o", ls="", ms=11, color=BLUE, mec=SURF, mew=2),
           Line2D([], [], marker="o", ls="", ms=11, color=ORANGE, mec=SURF, mew=2),
           Line2D([], [], color=INK, lw=2.4, ls=(0, (1.6, 1.7)))]
fig.legend(handles, ["Full GPU만 사용", "일부 GPU를 분할", "solver가 고르는 경계"],
           loc="upper center", bbox_to_anchor=(0.5, 0.905), ncol=3,
           frameon=False, fontsize=11, labelcolor=INK2, columnspacing=2.4)
fig.suptitle("같은 SLO를 가장 싸게 — 경계는 분할과 Full이 번갈아 만든다",
             fontsize=14.5, color=INK, y=0.965, fontweight="bold",
             x=0.07, ha="left")
fig.text(0.07, 0.032, f"실측 · 요청 150개 · 마감 {SLO.replace('video_','')} · "
         "비용 = 점유한 물리 GPU 수 × 벽시계 시간",
         fontsize=9, color=MUTED)

fig.savefig("results/fig_pareto.png", dpi=170, facecolor=SURF)
print("\nwrote results/fig_pareto.png")
