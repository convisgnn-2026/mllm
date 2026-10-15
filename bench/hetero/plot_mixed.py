#!/usr/bin/env python3
"""혼합 워크로드(Azure code+conv 트레이스) 결과: 구성 A~D × 도착률. 표 + 그림(fig_mixed_azure.png).
지표: 달성 처리량, TTFT p50/p90, TPOT p50/p90 (요청별 CSV 에서 재계산, TPOT 는 completion_tokens 기준)."""
import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "hetero", "open")
COLOR = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "D": "#eda100"}
LABEL = {"A": "A  replicate + round-robin (baseline)", "B": "B  replicate + weighted 2:1", "C": "C  PD 4090→3090", "D": "D  PD 3090→4090"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


def cell(cfg, rate):
    d = os.path.join(ROOT, cfg, f"mixed_azure_r{rate}")
    if not os.path.exists(os.path.join(d, "summary.json")):
        return None
    s = json.load(open(os.path.join(d, "summary.json")))
    R = [r for r in csv.DictReader(open(os.path.join(d, "requests.csv"))) if r["err"] == "" and r["first"] not in ("nan", "")]
    tt = np.array([float(r["first"]) - float(r["launch"]) for r in R])
    tp = np.array([(float(r["end"]) - float(r["first"])) / max(int(r.get("usage_completion") or r["out_tokens"]) - 1, 1) for r in R if int(r.get("usage_completion") or r["out_tokens"]) > 1])
    q = lambda v, p: float(np.percentile(v, p))
    return dict(thr=s["req_per_s"], ttft50=q(tt, 50), ttft90=q(tt, 90), ttft99=q(tt, 99), tpot50=q(tp, 50) * 1e3, tpot90=q(tp, 90) * 1e3, tpot99=q(tp, 99) * 1e3, n=len(R), err=s["errors"])


def main():
    rates = ["2.5", "3.0", "3.5", "4.0", "4.5"]; D = {(c, r): cell(c, r) for c in COLOR for r in rates}
    L = ["# 혼합 워크로드 (Azure LLM trace code+conv, 300 요청, 실제 도착 간격을 목표 도착률로 배율) — 4090 + 3090", "",
         "| 도착률 rps | 구성 | 달성 rps | TTFT p50 / p90 / p99 s | TPOT p50 / p90 / p99 ms |", "|---|---|---|---|---|"]
    for r in rates:
        for c in COLOR:
            x = D[(c, r)]
            if x:
                L.append(f"| {r} | {c} | {x['thr']:.2f} | {x['ttft50']:.2f} / {x['ttft90']:.2f} / {x['ttft99']:.1f} | {x['tpot50']:.0f} / {x['tpot90']:.0f} / {x['tpot99']:.0f} |")
    L += ["", "## 도착률별 최선 (지표별)", "", "| 도착률 | 처리량 | TTFT p90 | TPOT p90 |", "|---|---|---|---|"]
    for r in rates:
        have = {c: D[(c, r)] for c in COLOR if D[(c, r)]}
        if have:
            L.append(f"| {r} | {max(have, key=lambda c: have[c]['thr'])} | {min(have, key=lambda c: have[c]['ttft90'])} | {min(have, key=lambda c: have[c]['tpot90'])} |")
    open(os.path.join(ROOT, "mixed_azure.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4)); fig.patch.set_facecolor(SURF); xs = [float(r) for r in rates]
    for ax, (key, title, log) in zip(axes, [("thr", "achieved throughput (req/s)", False), ("ttft90", "TTFT p90 (s, log scale)", True), ("tpot90", "TPOT p90 (ms)", False)]):
        ax.set_facecolor(SURF); ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=8)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=8.5); ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
        for c in COLOR:
            ys = [D[(c, r)][key] if D[(c, r)] else np.nan for r in rates]
            ax.plot(xs, ys, "-o", color=COLOR[c], lw=2, ms=6, mec=SURF, mew=1.5, label=LABEL[c]); ax.text(xs[-1] + 0.05, ys[-1], c, color=INK2, fontsize=8.5, va="center")
        if log: ax.set_yscale("log")
        ax.set_xlabel("arrival rate (req/s)", color=INK2, fontsize=9); ax.set_xticks(xs)
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, fontsize=9, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Mixed production workload (Azure LLM trace: code + conversation) on RTX 4090 + RTX 3090, Llama-3.1-8B, vLLM 0.28", x=0.01, ha="left", fontsize=10, color=INK2)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95)); p = os.path.join(ROOT, "fig_mixed_azure.png"); fig.savefig(p, dpi=160); plt.close(fig); print(p)


if __name__ == "__main__":
    main()
