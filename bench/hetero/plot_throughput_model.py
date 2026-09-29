#!/usr/bin/env python3
"""그림 2: 다중 요청 처리량 모델 v1 — 예측 대 실측 (복제 구성 A/B, 닫힌 투입 24칸).
  왼쪽: 산점(예측 vs 실측 req/s), 구성별 색, 워크로드별 마커.  오른쪽: 혼합비 × 처리량 곡선(n=200), 실선 실측 · 점선 예측."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import throughput_model as T

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
COLOR = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "D": "#eda100"}
LABEL = {"A": "A  replicate + RR", "B": "B  replicate + weighted 2:1", "C": "C  PD 4090→3090", "D": "D  PD 3090→4090"}
MARK = {"prefill": "o", "p75": "v", "mixed": "s", "p25": "^", "decode": "D"}
MIXL = {"prefill": "100% prefill", "p75": "75%", "mixed": "50%", "p25": "25%", "decode": "0% (decode)"}
params = json.load(open(os.path.join(T.OUT, "throughput_params.json"))); T.CHUNK = params["chunk"]; BM = params["beta_mix"]


def style(ax, title):
    ax.set_facecolor(SURF); ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=8)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8.5); ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)


def main():
    pts = []
    def pred(cfg, mix, n):
        return (T.predict_rep(mix, n, *T.CFG[cfg], BM) if cfg in T.CFG else T.predict_pd_calibrated(mix, n, *T.CFG_PD[cfg], BM))["req_per_s"]
    for cfg in COLOR:
        for mix in T.MIXES:
            for n in T.NS:
                m = T.measured(cfg, mix, n)
                if m:
                    pts.append((cfg, mix, n, pred(cfg, mix, n), m["req_per_s"]))
    mape_ab = sum(abs(p - m) / m for c, _, _, p, m in pts if c in "AB") / sum(1 for c, *_ in pts if c in "AB")
    mape_cd = sum(abs(p - m) / m for c, _, _, p, m in pts if c in "CD") / sum(1 for c, *_ in pts if c in "CD")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8)); fig.patch.set_facecolor(SURF)
    style(ax1, f"closed-batch throughput, all configs  (A/B MAPE {mape_ab:.0%},  C/D MAPE {mape_cd:.0%})")
    for cfg, mix, n, p, m in pts:
        ax1.plot(m, p, MARK[mix], color=COLOR[cfg], ms=7 + (n // 100) * 1.5, mec=SURF, mew=1)
    lim = max(max(p, m) for *_, p, m in pts) * 1.08; ax1.plot([0, lim], [0, lim], "--", color=INK2, lw=1)
    ax1.text(0.05, lim * 0.93, "PD cells use the thrash calibration\nratio = min(1, (overflow·feed / 1.58)^-0.71)", fontsize=8, color=INK2, va="top")
    ax1.fill_between([0, lim], [0, lim * 0.85], [0, lim * 1.15], color=GRID, alpha=0.5, lw=0)
    ax1.text(lim * 0.97, lim * 0.83, "±15%", color=INK2, fontsize=8, ha="right")
    ax1.set_xlim(0, lim); ax1.set_ylim(0, lim); ax1.set_xlabel("measured (req/s)", color=INK2, fontsize=9); ax1.set_ylabel("predicted (req/s)", color=INK2, fontsize=9)
    h = [Line2D([], [], color=COLOR[c], marker="o", ls="", label=LABEL[c]) for c in COLOR] + \
        [Line2D([], [], color=INK2, marker=MARK[m], ls="", label=MIXL[m]) for m in MARK]
    ax1.legend(handles=h, frameon=False, fontsize=7.5, loc="lower right", ncol=2)
    style(ax2, "mix sweep at n=200: measured (solid) vs model (dashed)")
    pred_fn = pred
    xs = [T.MIXES[m] for m in T.MIXES]
    for cfg in COLOR:
        meas = [T.measured(cfg, m, 200)["req_per_s"] if T.measured(cfg, m, 200) else None for m in T.MIXES]
        pred = [pred_fn(cfg, m, 200) for m in T.MIXES]
        xm = [x for x, v in zip(xs, meas) if v]; ym = [v for v in meas if v]
        ax2.plot(xm, ym, "-o", color=COLOR[cfg], lw=2, ms=6, mec=SURF, label=f"{LABEL[cfg]} measured")
        ax2.plot(xs, pred, "--s", color=COLOR[cfg], lw=1.5, ms=5, mec=SURF, alpha=0.8, label=f"{cfg} model")
    ax2.set_xticks([0, .25, .5, .75, 1]); ax2.set_xticklabels(["0%\n(decode)", "25%", "50%", "75%", "100%\n(prefill)"])
    ax2.set_xlabel("share of prefill-heavy requests", color=INK2, fontsize=9); ax2.set_ylabel("throughput (req/s)", color=INK2, fontsize=9)
    ax2.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=2)
    fig.suptitle(f"Multi-request model v1 = step model + KV-capacity admission + mixed steps (β_mix={BM}, chunk={T.CHUNK}) + PD thrash calibration",
                 x=0.01, ha="left", fontsize=10, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); p = os.path.join(T.OUT, "fig2_throughput_model.png"); fig.savefig(p, dpi=160); plt.close(fig); print(p)


if __name__ == "__main__":
    main()
