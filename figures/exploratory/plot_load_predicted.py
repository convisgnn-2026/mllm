#!/usr/bin/env python
"""
Predicted load-vs-SAR curve for `2 full GPUs` against `2 GPUs each split in two`.

Anchored on measured quantities, not invented:
  - per-template slowdown on a 3g.135gb slice (70 SM): 1.64x - 1.80x
  - resulting capacity: 1.12 req/s un-sliced -> 1.27 req/s sliced (+13%)
  - mean service time: 2.18 s un-sliced, 3.85 s on a slice

Model: below capacity the queue is stable and SAR is high; above capacity the
backlog grows linearly, so request k waits ~ k(1/mu - 1/lambda) and meets its
deadline only while that stays under the slack D - s.  Slicing buys capacity but
costs per-request latency, which is why the curves cross rather than one
dominating -- that crossing is the whole point of the figure.
"""

import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != "/" and not _os.path.isdir(_os.path.join(_d, "results")):
    _d = _os.path.dirname(_d)
_sys.path[:0] = [_os.path.join(_d, "solver"), _os.path.join(_d, "bench")]
_os.chdir(_d)            # results/ and slo_configs/ are addressed from the repo root

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

MU_FULL, MU_SPLIT = 1.12, 1.27          # measured capacity, req/s
S_FULL, S_SPLIT = 2.18, 3.85            # mean service time on one worker, s
N, D = 150, 8.0                         # requests per run, absolute deadline


def sar(lam, mu, s):
    lam = np.asarray(lam, float)
    out = np.full_like(lam, 0.97)                       # stable-queue plateau
    over = lam > mu
    wait_per_req = 1.0 / mu - 1.0 / np.maximum(lam, 1e-9)
    k_ok = (D - s) / np.maximum(wait_per_req, 1e-9)     # how many clear the deadline
    out[over] = np.clip(k_ok[over] / N, 0.0, 0.97)
    return out


fig, ax = plt.subplots(figsize=(9.6, 5.8), facecolor=SURF)
fig.subplots_adjust(top=0.845, bottom=0.155, left=0.095, right=0.975)
ax.set_facecolor(SURF)

lam = np.linspace(0.6, 2.1, 400)
ax.plot(lam, sar(lam, MU_FULL, S_FULL), color=BLUE, lw=2.4, zorder=5)
ax.plot(lam, sar(lam, MU_SPLIT, S_SPLIT), color=ORANGE, lw=2.4, zorder=5)

ax.axvspan(MU_FULL, MU_SPLIT, color=ORANGE, alpha=0.10, zorder=1)
for x, c in ((MU_FULL, BLUE), (MU_SPLIT, ORANGE)):
    ax.axvline(x, color=c, lw=1.1, ls=(0, (4, 3)), zorder=3)

ax.annotate("1.12 ~ 1.27 req/s\n통짜만 무너지는 구간",
            xy=(0.635, 0.52), ha="left", va="center", fontsize=12,
            color=ORANGE, fontweight="bold", linespacing=1.6)
ax.annotate("통짜 2장", xy=(1.55, sar([1.55], MU_FULL, S_FULL)[0]),
            xytext=(0, -20), textcoords="offset points",
            fontsize=11, color=BLUE, fontweight="bold", ha="center")
ax.annotate("2장 모두 2분할", xy=(1.75, sar([1.75], MU_SPLIT, S_SPLIT)[0]),
            xytext=(0, 16), textcoords="offset points",
            fontsize=11, color=ORANGE, fontweight="bold", ha="center")

ax.set_xlim(0.6, 2.1)
ax.set_ylim(0, 1.05)
ax.set_xticks([0.6, 0.8, 1.0, 1.12, 1.27, 1.5, 1.8, 2.1])
ax.set_xticklabels(["0.6", "0.8", "1.0", "1.12", "1.27", "1.5", "1.8", "2.1"])
ax.set_xlabel("도착률  (req/s)", fontsize=11.5, color=INK2, labelpad=8)
ax.set_ylabel("SLO 달성률 (SAR)", fontsize=11.5, color=INK2)
ax.grid(True, color=GRID, lw=0.8, zorder=0)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(GRID)
ax.tick_params(colors=INK2, labelsize=10, length=0)

ax.set_title("부하가 올라가면 통짜가 먼저 무너진다", fontsize=15,
             color=INK, pad=26, loc="left", fontweight="bold")
ax.text(0, 1.035, "GPU 2장으로 동일 · 작은 요청 위주 워크로드 · 예상 곡선",
        transform=ax.transAxes, fontsize=10.5, color=INK2)
fig.text(0.095, 0.035,
         "측정값 기반: 3g.135gb 슬라이스에서 요청당 1.64~1.80배 느려지고, "
         "용량은 1.12 → 1.27 req/s (+13%)",
         fontsize=9, color=MUTED)

fig.savefig("results/fig_load_predicted.png", dpi=170, facecolor=SURF)
print("wrote results/fig_load_predicted.png")
for x in (0.8, 1.0, 1.2, 1.4, 1.6, 1.8):
    a, b = sar([x], MU_FULL, S_FULL)[0], sar([x], MU_SPLIT, S_SPLIT)[0]
    print(f"  rate {x}: 통짜 {a:.2f}  분할 {b:.2f}   차이 {b-a:+.2f}")
