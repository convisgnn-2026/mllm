#!/usr/bin/env python3
"""닫힌 실험 결과 그림. results/hetero/ab/<CONFIG>/<cell>/summary.json → results/hetero/ab/figs/*.png

  fig1  처리량(req/s)  워크로드 3패널 × n 3그룹 × 구성 4막대, A 대비 배수 직접 라벨
  fig2  TPOT p50(ms)   같은 배치 (SLO 축)
  fig3  혼합비 스윕     x = 프리필 중심 요청 비율, y = A 대비 배수, 구성별 선, n 별 패널 (p25/p75 가 있으면)
팔레트: dataviz 참조 팔레트의 범주형 슬롯 1~4 (검증된 순서), 텍스트는 텍스트 토큰.
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "hetero", "ab")
FIG = os.path.join(ROOT, "figs"); os.makedirs(FIG, exist_ok=True)
CONFIGS = ["A", "B", "C", "D"]
ALL = ["A", "B", "C", "D", "C_v1runner_cross"]
LABEL = {"A": "A  replicate + RR", "B": "B  replicate + weighted 2:1", "C": "C  PD 4090→3090", "D": "D  PD 3090→4090",
         "C_v1runner_cross": "C  PD 4090→3090 (legacy runner, cross-layers on)"}
COLOR = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "D": "#eda100", "C_v1runner_cross": "#1baf7a"}   # categorical slots 1-4
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
WORKLOADS = [("prefill", "Prefill-heavy  2048/128"), ("mixed", "Mixed  50/50"), ("decode", "Decode-heavy  128/1024")]
MIX_ORDER = [("decode", 0.0), ("p25", 0.25), ("mixed", 0.5), ("p75", 0.75), ("prefill", 1.0)]
NS = [50, 100, 200]


def load():
    d = {}
    for c in ALL:
        for f in glob.glob(os.path.join(ROOT, c, "*_n*", "summary.json")):
            cell = os.path.basename(os.path.dirname(f)); name, n = cell.split("_n")
            d[(c, name, int(n))] = json.load(open(f))
    return d


def style(ax, title):
    ax.set_facecolor(SURF); ax.set_title(title, loc="left", fontsize=11, color=INK, pad=8)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9); ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)


def grouped(d, key, ylabel, fname, fmt, ratio_to_a):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=False); fig.patch.set_facecolor(SURF)
    w = 0.2
    for ax, (wl, title) in zip(axes, WORKLOADS):
        style(ax, title)
        for i, c in enumerate(CONFIGS):
            xs, ys = [], []
            for j, n in enumerate(NS):
                s = d.get((c, wl, n))
                if s is None:
                    continue
                v = key(s); xs.append(j + (i - 1.5) * (w + 0.02)); ys.append(v)
                base = d.get(("A", wl, n))
                lab = fmt(v) if not ratio_to_a or base is None else f"{v / key(base):.2f}×"
                ax.text(xs[-1], v, lab, ha="center", va="bottom", fontsize=7.5, color=INK2)
            ax.bar(xs, ys, width=w, color=COLOR[c], label=LABEL[c], linewidth=0)
        ax.set_xticks(range(len(NS))); ax.set_xticklabels([f"n={n}" for n in NS])
        if wl == "decode":   # C n=200 capped
            ax.text(2 + 0.5 * (w + 0.02), 0.02, "C: >600 s\ncapped", ha="center", va="bottom", fontsize=7, color=INK2, transform=ax.get_xaxis_transform())
    axes[0].set_ylabel(ylabel, color=INK2, fontsize=9)
    for ax in axes:
        ax.set_ylim(0, ax.get_ylim()[1] * 1.08)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=9, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(os.path.join(FIG, fname), dpi=160); plt.close(fig)


def mix_sweep(d):
    have = {k[1] for k in d}
    if not ({"p25", "p75"} & have):
        return
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True); fig.patch.set_facecolor(SURF)
    for ax, n in zip(axes, NS):
        style(ax, f"n = {n}")
        for c in CONFIGS:   # A 는 0/50/100% 만 있음 → 절대값으로 그린다
            xs, ys = [], []
            for name, frac in MIX_ORDER:
                s = d.get((c, name, n))
                if s:
                    xs.append(frac); ys.append(s["req_per_s"])
            ax.plot(xs, ys, "-o", color=COLOR[c], lw=2, ms=6, label=LABEL[c], mec=SURF, mew=1.5)
            if xs:
                ax.text(xs[-1] + 0.02, ys[-1], c, color=INK2, fontsize=8, va="center")
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_xticklabels(["0%\n(decode)", "25%", "50%", "75%", "100%\n(prefill)"])
        ax.set_xlabel("share of prefill-heavy requests", color=INK2, fontsize=9)
    axes[0].set_ylabel("throughput (req/s)", color=INK2, fontsize=9)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=9, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(os.path.join(FIG, "fig3_mix_sweep.png"), dpi=160); plt.close(fig)


def cross_compare(d):
    if not any(k[0] == "C_v1runner_cross" for k in d):
        return
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2)); fig.patch.set_facecolor(SURF)
    w = 0.2; pairs = [("C_v1runner_cross", "C")]
    for ax, (wl, title) in zip(axes, WORKLOADS):
        style(ax, title)
        for pi, (off, on) in enumerate(pairs):
            for oi, c in enumerate((off, on)):
                xs, ys = [], []
                for j, n in enumerate(NS):
                    s = d.get((c, wl, n))
                    if s is None:
                        continue
                    xs.append(j + (pi * 2 + oi - 0.5) * (w + 0.02)); ys.append(s["req_per_s"])
                    ax.text(xs[-1], ys[-1], f"{ys[-1]:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
                ax.bar(xs, ys, width=w, color=COLOR[c], hatch="//" if oi == 0 else None, edgecolor=SURF, linewidth=0.8,
                       alpha=0.55 if oi == 0 else 1.0, label=LABEL[c])
        ax.set_xticks(range(len(NS))); ax.set_xticklabels([f"n={n}" for n in NS])
        ax.set_ylim(0, ax.get_ylim()[1] * 1.08)
    axes[0].set_ylabel("throughput (req/s)", color=INK2, fontsize=9)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(os.path.join(FIG, "fig4_cross_layers.png"), dpi=160); plt.close(fig)


def main():
    d = load()
    grouped(d, lambda s: s["req_per_s"], "throughput (req/s)", "fig1_throughput.png", lambda v: f"{v:.2f}", ratio_to_a=True)
    grouped(d, lambda s: s["tpot_ms"]["p50"], "TPOT p50 (ms)", "fig2_tpot.png", lambda v: f"{v:.0f}", ratio_to_a=False)
    mix_sweep(d)
    cross_compare(d)
    print("figs ->", FIG, sorted(os.listdir(FIG)))


if __name__ == "__main__":
    main()
