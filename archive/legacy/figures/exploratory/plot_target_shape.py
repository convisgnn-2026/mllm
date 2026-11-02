#!/usr/bin/env python
"""
What the figure has to look like to carry a solver paper -- and an honest check
of whether our configuration space is actually rich enough to need one.

A two-line crossing only says "there exists a setting where slicing wins", which
a reviewer answers with "then buy 13% more GPU".  The claim that needs a solver
is stronger: across the operating space the best configuration keeps changing,
and every fixed configuration is badly wrong somewhere.

So: (left) the envelope over a family of configurations at one deadline,
(right) the map of which configuration wins where.  The headline number is the
regret of the best fixed choice against the envelope.  If the map has two
regions, the "solver" is a threshold rule and the framing is too thin -- that is
a finding, not a failure, so the script prints the region count either way.

Service times come from measurement: full GPU per-template latencies, and a
slice slowdown of (148 / SM) x 0.83, the factor observed at both 70 SM
(predicted 2.11x, measured 1.76x) and 18 SM (predicted 8.22x, measured ~6.7x).
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

import heapq
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8983"
SURF, GRID = "#fcfcfb", "#e6e5e0"

# measured full-GPU latency (20 steps + VAE), seconds
FULL = {"480x272x33": 0.819, "832x480x33": 2.590,
        "832x480x49": 4.014, "1280x720x49": 12.145}
# short_seqlen_high mix
PROB = {"480x272x33": 0.502, "832x480x33": 0.288,
        "832x480x49": 0.187, "1280x720x49": 0.022}
TPLS = list(FULL)
SLOW_K = 0.83                     # measured: slice is 0.83 x the proportional penalty
N = 400


def svc(tpl, sm):
    return FULL[tpl] * (1.0 if sm == 148 else (148.0 / sm) * SLOW_K)


# (label, worker SM list on a 2-GPU budget, colour)
CFGS = [
    ("통짜 2장",              [148, 148], BLUE),
    ("2장 모두 2분할",        [70] * 4, ORANGE),
    ("1장 통짜 + 1장 2분할",  [148, 70, 70], AQUA),
    ("2장 모두 7분할",        [18] * 14, "#eda100"),
    ("2장 모두 3g+1g×4",      ([70] + [18] * 4) * 2, "#4a3aa7"),
]


def simulate(sms, lam, D, seed=0):
    """Event-driven: SRPT queue order, request goes to the fastest free worker."""
    rng = np.random.default_rng(seed)
    gaps = rng.exponential(1.0 / lam, N)
    arr = np.cumsum(gaps) - gaps[0]
    p = np.array([PROB[t] for t in TPLS], float); p /= p.sum()
    idx = rng.choice(len(TPLS), size=N, p=p)

    free = sorted(range(len(sms)), key=lambda w: -sms[w])
    busy = []                                   # heap of (finish, worker, arrival)
    pend, hits, nxt, t = [], 0, 0, 0.0

    while len(pend) or nxt < N or busy:
        # advance to the next thing that can happen
        cands = []
        if nxt < N:
            cands.append(arr[nxt])
        if busy:
            cands.append(busy[0][0])
        if pend and free:
            cands.append(t)
        if not cands:
            break
        t = max(t, min(cands))

        while busy and busy[0][0] <= t + 1e-12:
            fin, w, a = heapq.heappop(busy)
            free.append(w); free.sort(key=lambda w: -sms[w])
            hits += (fin - a) <= D
        while nxt < N and arr[nxt] <= t + 1e-12:
            pend.append((float(arr[nxt]), TPLS[idx[nxt]])); nxt += 1
        while free and pend:
            pend.sort(key=lambda r: FULL[r[1]])          # shortest first
            a, tpl = pend.pop(0)
            w = free.pop(0)                              # fastest free worker
            heapq.heappush(busy, (t + svc(tpl, sms[w]), w, a))

    return hits / N


fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.8, 6.2), facecolor=SURF)
fig.subplots_adjust(top=0.775, bottom=0.145, left=0.065, right=0.985, wspace=0.235)

# ---------------------------------------------------------------- left
D_FIX = 12.0
lam = np.linspace(0.5, 2.4, 26)
curves = []
for lab, sms, c in CFGS:
    y = np.array([simulate(sms, l, D_FIX) for l in lam])
    curves.append(y)
    axL.plot(lam, y, color=c, lw=2.2, zorder=4)

env = np.max(np.vstack(curves), axis=0)
best = max(range(len(curves)), key=lambda i: curves[i].mean())
axL.plot(lam, env, color=INK, lw=3.2, ls=(0, (1.6, 1.7)), zorder=6)
axL.fill_between(lam, curves[best], env, color=INK, alpha=0.10, zorder=2)
regret = float((env - curves[best]).mean())

axL.annotate(f"solver가 얻는 몫\n최고 고정 구성 대비 평균 +{regret:.3f}",
             xy=(1.25, 0.60), ha="left", va="center", fontsize=11.5,
             color=INK, fontweight="bold", linespacing=1.6)
axL.set_xlim(0.5, 2.4); axL.set_ylim(0, 1.06)
axL.set_xlabel("도착률  (req/s)", fontsize=11.5, color=INK2, labelpad=7)
axL.set_ylabel("SLO 달성률 (SAR)", fontsize=11.5, color=INK2)
axL.set_title(f"마감 {D_FIX:.0f}초 고정 — 어떤 구성도 전 구간 최선이 아니다",
              fontsize=12.5, color=INK, pad=10, loc="left")

# ---------------------------------------------------------------- right
lam2 = np.linspace(0.5, 2.4, 22)
Dg = np.linspace(4, 40, 20)
win = np.zeros((len(Dg), len(lam2)), int)
for j, l in enumerate(lam2):
    for i, d in enumerate(Dg):
        win[i, j] = int(np.argmax([simulate(sms, l, d) for _, sms, _ in CFGS]))

LAM, DD = np.meshgrid(lam2, Dg)
axR.pcolormesh(LAM, DD, win, cmap=ListedColormap([c for *_, c in CFGS]),
               vmin=-0.5, vmax=len(CFGS) - 0.5, alpha=0.30, shading="auto")
share = {i: float((win == i).mean()) for i in range(len(CFGS))}
for i, (lab, sms, c) in enumerate(CFGS):
    if share[i] < 0.04:
        continue
    m = win == i
    axR.text(LAM[m].mean(), DD[m].mean(), f"{lab}\n{share[i]*100:.0f}%",
             ha="center", va="center", fontsize=11.5, color=c,
             fontweight="bold", linespacing=1.5)
axR.set_xlim(0.5, 2.4); axR.set_ylim(4, 40)
axR.set_xlabel("도착률  (req/s)", fontsize=11.5, color=INK2, labelpad=7)
axR.set_ylabel("마감 (초)", fontsize=11.5, color=INK2)
nreg = sum(1 for v in share.values() if v >= 0.04)
axR.set_title(f"운영점에 따라 이기는 구성이 갈린다 — 영역 {nreg}개",
              fontsize=12.5, color=INK, pad=10, loc="left")

for ax in (axL, axR):
    ax.set_facecolor(SURF)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=10, length=0)

handles = [Line2D([], [], color=c, lw=2.2) for *_, c in CFGS] + \
          [Line2D([], [], color=INK, lw=3.2, ls=(0, (1.6, 1.7)))]
labels = [lab for lab, *_ in CFGS] + ["가능한 최선 (solver)"]
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.875),
           ncol=6, frameon=False, fontsize=10.5, labelcolor=INK2,
           columnspacing=1.8, handletextpad=0.6)
fig.suptitle("연구적으로 의미 있으려면 — 목표 형태 (GPU 2장 고정, 분할 방식만 다름)",
             fontsize=14.5, color=INK, y=0.962, fontweight="bold", x=0.065, ha="left")

fig.savefig("results/fig_target_shape.png", dpi=170, facecolor=SURF)
print("wrote results/fig_target_shape.png")
print(f"\n최고 고정 구성 = {CFGS[best][0]} ,  envelope 대비 평균 regret {regret:.3f}")
print(f"영역 개수(>=4%): {nreg}")
for i, (lab, sms, _) in enumerate(CFGS):
    print(f"  {lab:>22}  워커 {len(sms):>2}개  영역 {share[i]*100:>5.1f}%")
