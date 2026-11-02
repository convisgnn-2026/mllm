#!/usr/bin/env python
"""
Two figures from the load sweep (same 2-GPU budget, small-request-heavy mix):

  fig_tail.png       p95 response time vs load -- where slicing actually pays
  fig_sar_vs_tail.png  the same runs scored two ways, side by side, because the
                       two metrics disagree and that disagreement is the finding

Slicing raises the worker count from 2 to 4 on the same hardware, so a large
request blocks fewer small ones behind it.  That shows up in the tail long
before it shows up in a deadline-hit rate, which is why the SAR panel looks flat
while the p95 panel does not.
"""

import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != "/" and not _os.path.isdir(_os.path.join(_d, "results")):
    _d = _os.path.dirname(_d)
_sys.path[:0] = [_os.path.join(_d, "solver"), _os.path.join(_d, "bench")]
_os.chdir(_d)            # results/ and slo_configs/ are addressed from the repo root

import glob
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

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8983"
SURF, GRID = "#fcfcfb", "#e6e5e0"

RATES = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
CAP_2F = 1.12                                   # measured capacity, req/s
CFG = [("2F", "Full 2장", BLUE),
       ("1F+1x2", "Full 1 + 2분할 1", AQUA),
       ("0F+2x2", "2장 모두 2분할", ORANGE)]


def lat(tag, r):
    f = glob.glob(f"results/small/{tag}_r{r:.1f}.json")
    if not f:
        return None
    q = sorted(x["end_s"] - x["arrival_s"] for x in json.load(open(f[0]))["requests"])
    return {"p50": q[len(q) // 2], "p95": q[int(len(q) * .95) - 1],
            "mean": sum(q) / len(q)}


def sar_of(tag, r, slo="video_3.0x"):
    s = json.load(open("results/scores_small.json"))
    for f, d in s.items():
        if d["policy"] == tag and abs(d["workload"]["rate"] - r) < 1e-6:
            return d["scores"][slo]["sar"]
    return None


def style(ax, ylab):
    ax.set_facecolor(SURF)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=10, length=0)
    ax.set_xlabel("도착률  (req/s)", fontsize=11.5, color=INK2, labelpad=7)
    ax.set_ylabel(ylab, fontsize=11.5, color=INK2)
    ax.axvline(CAP_2F, color=MUTED, lw=1.1, ls=(0, (4, 3)), zorder=2)


D = {t: {r: lat(t, r) for r in RATES} for t, *_ in CFG}

# ======================================================= figure 1: tail only
fig, ax = plt.subplots(figsize=(9.8, 6.0), facecolor=SURF)
fig.subplots_adjust(top=0.815, bottom=0.15, left=0.095, right=0.975)

p95 = {t: [D[t][r]["p95"] for r in RATES] for t, *_ in CFG}

# Standalone figure keeps only the two extremes -- the middle configuration
# sits between the curves and would collide with every gap annotation.
ax.fill_between(RATES, p95["0F+2x2"], p95["2F"], color=ORANGE, alpha=0.15, zorder=1)
for t, lab, c in [CFG[0], CFG[2]]:
    ax.plot(RATES, p95[t], color=c, lw=2.6, marker="o", ms=7,
            mec=SURF, mew=1.8, zorder=5)

for r in (1.2, 1.4, 1.6):
    a, b = D["2F"][r]["p95"], D["0F+2x2"][r]["p95"]
    ax.annotate(f"−{(1-b/a)*100:.0f}%", xy=(r, (a + b) / 2),
                xytext=(-10, 0), textcoords="offset points",
                ha="right", va="center", fontsize=13.5, color=ORANGE,
                fontweight="bold")

ax.annotate("Full 2장", xy=(1.6, p95["2F"][RATES.index(1.6)]),
            xytext=(-4, 14), textcoords="offset points", ha="right",
            fontsize=12.5, color=BLUE, fontweight="bold")
ax.annotate("2장 모두 2분할", xy=(1.6, p95["0F+2x2"][RATES.index(1.6)]),
            xytext=(4, -24), textcoords="offset points", ha="left",
            fontsize=12.5, color=ORANGE, fontweight="bold")
ax.annotate("Full 2장의 처리 한계", xy=(CAP_2F, 4), xytext=(7, 0),
            textcoords="offset points", fontsize=10, color=MUTED, va="bottom")

style(ax, "p95 응답시간 (초)")
ax.set_xlim(0.75, 2.08)
ax.set_ylim(0, max(p95["2F"]) * 1.12)
ax.set_title("같은 GPU 2장인데, 쪼개면 꼬리 지연이 최대 41% 짧다",
             fontsize=15, color=INK, pad=28, loc="left", fontweight="bold")
ax.text(0, 1.035, "부하가 오를수록 격차가 벌어진다 · 작은 요청 위주 워크로드",
        transform=ax.transAxes, fontsize=10.5, color=INK2)
fig.text(0.095, 0.032, "실측 · 요청 150개 · 감마 도착(CV 1.0) · "
         "워커 수 2개 → 4개, 물리 GPU는 2장으로 동일",
         fontsize=9, color=MUTED)
fig.savefig("results/fig_tail.png", dpi=170, facecolor=SURF)
print("wrote results/fig_tail.png")

# ============================================ figure 2: SAR beside the tail
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.4, 5.9), facecolor=SURF)
fig.subplots_adjust(top=0.775, bottom=0.15, left=0.07, right=0.98, wspace=0.22)

for t, lab, c in CFG:
    a1.plot(RATES, [sar_of(t, r) for r in RATES], color=c, lw=2.4,
            marker="o", ms=6.5, mec=SURF, mew=1.6, zorder=5)
    a2.plot(RATES, p95[t], color=c, lw=2.4, marker="o", ms=6.5,
            mec=SURF, mew=1.6, zorder=5)
a2.fill_between(RATES, p95["0F+2x2"], p95["2F"], color=ORANGE, alpha=0.13, zorder=1)

style(a1, "SLO 달성률 (SAR)")
style(a2, "p95 응답시간 (초)")
a1.set_ylim(0.5, 1.03)
a2.set_ylim(0, max(p95["2F"]) * 1.12)
for ax in (a1, a2):
    ax.set_xlim(0.75, 2.08)
# annotate the widest gap on each panel rather than asserting a size in prose
gaps = [(r, sar_of("0F+2x2", r) - sar_of("2F", r)) for r in RATES]
rg, dg = max(gaps, key=lambda x: x[1])
a1.annotate(f"+{dg*100:.0f}%p", xy=(rg, (sar_of("2F", rg) + sar_of("0F+2x2", rg)) / 2),
            xytext=(-10, 0), textcoords="offset points", ha="right", va="center",
            fontsize=13, color=ORANGE, fontweight="bold")
a1.fill_between(RATES, [sar_of("2F", r) for r in RATES],
                [sar_of("0F+2x2", r) for r in RATES], color=ORANGE,
                alpha=0.13, zorder=1)
rt = max(RATES, key=lambda r: 1 - D["0F+2x2"][r]["p95"] / D["2F"][r]["p95"])
a2.annotate(f"−{(1-D['0F+2x2'][rt]['p95']/D['2F'][rt]['p95'])*100:.0f}%",
            xy=(rt, (D["2F"][rt]["p95"] + D["0F+2x2"][rt]["p95"]) / 2),
            xytext=(-10, 0), textcoords="offset points", ha="right", va="center",
            fontsize=13, color=ORANGE, fontweight="bold")

a1.set_title(f"마감 달성률 — 고부하에서 +{dg*100:.0f}%p", fontsize=12.5,
             color=INK, pad=10, loc="left")
a2.set_title("꼬리 지연 — 최대 41% 단축", fontsize=12.5,
             color=INK, pad=10, loc="left")

from matplotlib.lines import Line2D
h = [Line2D([], [], color=c, lw=2.4, marker="o", ms=6.5, mec=SURF, mew=1.6)
     for *_, c in CFG]
fig.legend(h, [lab for _, lab, _ in CFG], loc="upper center",
           bbox_to_anchor=(0.5, 0.885), ncol=3, frameon=False,
           fontsize=11, labelcolor=INK2, columnspacing=2.6)
fig.suptitle("고부하에서는 분할이 두 지표 모두 앞선다 — 같은 GPU 2장",
             fontsize=14.5, color=INK, y=0.962, fontweight="bold",
             x=0.07, ha="left")
fig.text(0.07, 0.032, "실측 · 요청 150개 · 마감 3.0x 기준 · 점선 = Full 2장의 처리 한계 "
         "· 마감이 빡빡하면(1.0x) 반대로 Full이 앞선다",
         fontsize=9, color=MUTED)
fig.savefig("results/fig_sar_vs_tail.png", dpi=170, facecolor=SURF)
print("wrote results/fig_sar_vs_tail.png")

print(f"\n{'rate':>5} {'부하':>6} | {'2F p95':>8} {'분할 p95':>9} {'개선':>7} | "
      f"{'2F SAR':>7} {'분할 SAR':>9}")
for r in RATES:
    a, b = D["2F"][r]["p95"], D["0F+2x2"][r]["p95"]
    print(f"{r:>5} {r/CAP_2F*100:>5.0f}% | {a:>8.1f} {b:>9.1f} "
          f"{(1-b/a)*100:>6.0f}% | {sar_of('2F', r):>7.2f} {sar_of('0F+2x2', r):>9.2f}")
