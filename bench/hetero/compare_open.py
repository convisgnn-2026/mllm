#!/usr/bin/env python3
"""열린 도착 검증: 고정된 예측(results/hetero/open/predictions.json) 대 실측(results/hetero/open/<cfg>/<mix>_r<rate>/summary.json).
지표: 달성 처리량, TTFT p50, TPOT p50. 표 + 그림(fig3_open_validation.png: 3 패널, 예측 대 실측)."""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "hetero")
OPEN = os.path.join(ROOT, "open"); PRED = json.load(open(os.path.join(OPEN, "predictions.json")))
COLOR = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "D": "#eda100"}
LABEL = {"A": "A  replicate + RR", "B": "B  replicate + weighted 2:1", "C": "C  PD 4090→3090", "D": "D  PD 3090→4090"}
MARK = {"prefill": "o", "mixed": "s", "decode": "D"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(OPEN, "*", "*_r*", "summary.json"))):
        cfg = f.split(os.sep)[-3]; cell = f.split(os.sep)[-2]; mix, r = cell.split("_r")
        m = json.load(open(f)); p = PRED.get(f"{cfg}:{mix}:{r}")
        if not p:
            continue
        rows.append(dict(cfg=cfg, mix=mix, rate=float(r), m=m, p=p))
    if not rows:
        print("no measured cells yet"); return
    lines = ["# 열린 도착 검증 — 예측(고정) 대 실측", "", "| cfg | mix | λ | 처리량 예/실 | TTFT p50 예/실 ms | TPOT p50 예/실 ms | 오류 |", "|---|---|---|---|---|---|---|"]
    E = {"thr": [], "ttft": [], "tpot": []}
    for r in rows:
        m, p = r["m"], r["p"]
        e = dict(thr=(p["req_per_s"] - m["req_per_s"]) / m["req_per_s"], ttft=(p["ttft_ms"] - m["ttft_ms"]["p50"]) / m["ttft_ms"]["p50"], tpot=(p["tpot_ms"] - m["tpot_ms"]["p50"]) / m["tpot_ms"]["p50"])
        for k in E: E[k].append(e[k])
        lines.append(f"| {r['cfg']} | {r['mix']} | {r['rate']} | {p['req_per_s']:.2f}/{m['req_per_s']:.2f} | {p['ttft_ms']:.0f}/{m['ttft_ms']['p50']:.0f} | {p['tpot_ms']:.1f}/{m['tpot_ms']['p50']:.1f} | {m['errors']} |")
    lines += ["", "MAPE: " + ", ".join(f"{k} {np.mean(np.abs(v)):.0%} (편향 {np.mean(v):+.0%})" for k, v in E.items()), ""]
    for cfg in COLOR:
        sub = [r for r in rows if r["cfg"] == cfg]
        if sub:
            lines.append(f"- {cfg}: 처리량 MAPE {np.mean([abs((r['p']['req_per_s']-r['m']['req_per_s'])/r['m']['req_per_s']) for r in sub]):.0%}, "
                         f"TTFT {np.mean([abs((r['p']['ttft_ms']-r['m']['ttft_ms']['p50'])/r['m']['ttft_ms']['p50']) for r in sub]):.0%}, "
                         f"TPOT {np.mean([abs((r['p']['tpot_ms']-r['m']['tpot_ms']['p50'])/r['m']['tpot_ms']['p50']) for r in sub]):.0%} (n={len(sub)})")
    open(os.path.join(OPEN, "README.md"), "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3)); fig.patch.set_facecolor(SURF)
    for ax, (key, title, fp, fm) in zip(axes, [("thr", "achieved throughput (req/s)", lambda p: p["req_per_s"], lambda m: m["req_per_s"]),
                                              ("ttft", "TTFT p50 (ms, log)", lambda p: p["ttft_ms"], lambda m: m["ttft_ms"]["p50"]),
                                              ("tpot", "TPOT p50 (ms)", lambda p: p["tpot_ms"], lambda m: m["tpot_ms"]["p50"])]):
        ax.set_facecolor(SURF); ax.set_title(title + f"   MAPE {np.mean(np.abs(E[key])):.0%}", loc="left", fontsize=10, color=INK, pad=8)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=8.5); ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
        xs = [fm(r["m"]) for r in rows]; ys = [fp(r["p"]) for r in rows]
        for r, x, y in zip(rows, xs, ys):
            ax.plot(x, y, MARK[r["mix"]], color=COLOR[r["cfg"]], ms=7, mec=SURF)
        lo, hi = min(xs + ys) * 0.8, max(xs + ys) * 1.15; ax.plot([lo, hi], [lo, hi], "--", color=INK2, lw=1)
        if key == "ttft":
            ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_xlabel("measured", color=INK2, fontsize=9); ax.set_ylabel("predicted", color=INK2, fontsize=9)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=COLOR[c], marker="o", ls="", label=LABEL[c]) for c in COLOR] + [Line2D([], [], color=INK2, marker=MARK[m], ls="", label=m) for m in MARK]
    fig.legend(handles=h, frameon=False, fontsize=8, loc="lower center", ncol=7, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Open-arrival validation (Poisson λ, n=150): model parameters frozen from closed-batch calibration", x=0.01, ha="left", fontsize=10, color=INK2)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95)); p = os.path.join(OPEN, "fig3_open_validation.png"); fig.savefig(p, dpi=160); plt.close(fig); print(p)


if __name__ == "__main__":
    main()
