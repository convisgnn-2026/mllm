#!/usr/bin/env python3
"""기존 requests.csv 를 TPOT SLO 로 재채점해 goodput 을 낸다 (새 실험 없음).

  goodput(T) = #{요청: TPOT ≤ T ms} / makespan      [req/s]
  SAR(T)     = 만족 비율

  python bench/hetero/score_goodput.py [--slo 50 75 100]
출력: 표(stdout) + results/hetero/ab/goodput_<T>.md + figs/fig5_goodput_mix_T<T>.png (혼합비 × goodput, 구성별 선)
TTFT 는 닫힌 투입에서 큐 대기가 지배하므로 여기서는 걸지 않는다(열린 도착 실험에서 추가).
"""
import argparse
import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "hetero", "ab")
FIG = os.path.join(ROOT, "figs")
CONFIGS = ["A", "B", "C", "D"]
LABEL = {"A": "A  replicate + RR", "B": "B  replicate + weighted 2:1", "C": "C  PD 4090→3090", "D": "D  PD 3090→4090"}
COLOR = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "D": "#eda100"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
MIX_ORDER = [("decode", 0.0), ("p25", 0.25), ("mixed", 0.5), ("p75", 0.75), ("prefill", 1.0)]
NS = [50, 100, 200]


def load():
    d = {}
    for c in CONFIGS:
        for f in glob.glob(os.path.join(ROOT, c, "*_n*", "summary.json")):
            cell = os.path.dirname(f); name, n = os.path.basename(cell).split("_n")
            s = json.load(open(f))
            rows = list(csv.DictReader(open(os.path.join(cell, "requests.csv"))))
            tpot = [(float(r["end"]) - float(r["first"])) / max(int(r.get("usage_completion") or r["out_tokens"]) - 1, 1) * 1e3
                    for r in rows if r["err"] == "" and int(r["out_tokens"]) > 0 and r["first"] not in ("nan", "")]
            d[(c, name, int(n))] = dict(makespan=s["makespan_s"], tpot=tpot, n=len(rows))
    return d


def score(d, T):
    out = {}
    for k, v in d.items():
        ok = sum(1 for t in v["tpot"] if t <= T)
        out[k] = dict(goodput=ok / v["makespan"], sar=ok / v["n"], thr=len(v["tpot"]) / v["makespan"])
    return out


def table(sc, T):
    lines = [f"# Goodput @ TPOT ≤ {T} ms  (req/s; 괄호 = SAR)", "",
             "| n | 프리필 비율 | A | B | C | D |", "|---|---|---|---|---|---|"]
    for n in NS:
        for name, frac in MIX_ORDER:
            cells = []
            for c in CONFIGS:
                r = sc.get((c, name, n))
                cells.append(f"{r['goodput']:.2f} ({r['sar']:.0%})" if r else "–")
            lines.append(f"| {n} | {frac:.0%} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def fig(sc, T):
    figh, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True); figh.patch.set_facecolor(SURF)
    for ax, n in zip(axes, NS):
        ax.set_facecolor(SURF); ax.set_title(f"n = {n}   goodput @ TPOT ≤ {T} ms", loc="left", fontsize=11, color=INK, pad=8)
        for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
        for s_ in ("left", "bottom"): ax.spines[s_].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=9); ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
        for c in CONFIGS:
            xs, ys = [], []
            for name, frac in MIX_ORDER:
                r = sc.get((c, name, n))
                if r:
                    xs.append(frac); ys.append(r["goodput"])
            ax.plot(xs, ys, "-o", color=COLOR[c], lw=2, ms=6, label=LABEL[c], mec=SURF, mew=1.5)
            if xs:
                ax.text(xs[-1] + 0.02, ys[-1], c, color=INK2, fontsize=8, va="center")
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_xticklabels(["0%\n(decode)", "25%", "50%", "75%", "100%\n(prefill)"])
        ax.set_xlabel("share of prefill-heavy requests", color=INK2, fontsize=9)
    axes[0].set_ylabel("goodput (req/s meeting SLO)", color=INK2, fontsize=9)
    h, l = axes[0].get_legend_handles_labels()
    figh.legend(h, l, frameon=False, fontsize=9, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
    figh.tight_layout(rect=(0, 0.06, 1, 1)); p = os.path.join(FIG, f"fig5_goodput_mix_T{T}.png"); figh.savefig(p, dpi=160); plt.close(figh)
    return p


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--slo", type=int, nargs="+", default=[50, 75, 100]); a = ap.parse_args()
    d = load()
    for T in a.slo:
        sc = score(d, T); txt = table(sc, T)
        open(os.path.join(ROOT, f"goodput_T{T}.md"), "w").write(txt + "\n")
        print(txt, "\n"); print("fig ->", fig(sc, T), "\n")


if __name__ == "__main__":
    main()
