#!/usr/bin/env python3
"""동료 가설 검증: (1) TTFT 는 도착 시점에 앞에 쌓인 프리필 토큰 수에 바운드되는가  (2) 꼬리(p90/p99)와 상한 예측.
열린 도착 셀의 requests.csv (launch=도착, first=첫 토큰, in_len) 로 요청마다
   Q_i = Σ_{j: launch_j < launch_i, first_j > launch_i} in_len_j        (내 앞에서 아직 첫 토큰을 못 받은 프리필 토큰 수)
를 구하고 TTFT_i 와 회귀:  TTFT = a + b·Q  (b = 대기 프리필 토큰당 지연, 청크 스텝 시간/2048 과 비교).
상한:  TTFT_bound = T_pre(in_i) + (Q_i / CHUNK) · T_chunk   → 실측 ≤ 상한 비율(coverage).
"""
import csv
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_llm as M

OPEN = os.path.join(M.ROOT, "open"); BETA = json.load(open(os.path.join(M.OUT, "beta.json"))); CHUNK = 2048
COLOR = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "D": "#eda100"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


def t_pre(S, k):
    b = BETA[k]["prefill"]; return b["b1"] * M.t_pre_raw(S, k) + b["b0"]


def load_cell(f):
    rows = [r for r in csv.DictReader(open(f)) if r["err"] == "" and r["first"] not in ("nan", "")]
    arr = np.array([float(r["launch"]) for r in rows]); first = np.array([float(r["first"]) for r in rows])
    S = np.array([int(r["in_len"]) for r in rows]); ttft = first - arr
    Q = np.array([S[(arr < arr[i]) & (first > arr[i])].sum() for i in range(len(rows))])
    return arr, first, S, ttft, Q


def main():
    cells = sorted(glob.glob(os.path.join(OPEN, "*", "*_r*", "requests.csv")))
    lines = ["# HOL 가설 검증: TTFT = a + b·Q (Q = 도착 시 앞에 대기 중인 프리필 토큰)", "",
             "| cfg | cell | n | TTFT p50/p90/p99 ms | a ms | b µs/token | R² | Q p50/p90 tokens | 상한 coverage p50/p90/p99 |", "|---|---|---|---|---|---|---|---|---|"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6)); fig.patch.set_facecolor(SURF); allpts = []
    for f in cells:
        cfg = f.split(os.sep)[-3]; cell = f.split(os.sep)[-2]; mix = cell.split("_r")[0]
        arr, first, S, ttft, Q = load_cell(f)
        if len(ttft) < 20:
            continue
        b, a = np.polyfit(Q, ttft, 1); pred = a + b * Q; r2 = 1 - ((ttft - pred) ** 2).sum() / ((ttft - ttft.mean()) ** 2).sum()
        # 상한: 자기 프리필 + 앞선 프리필 토큰을 청크 예산으로 처리하는 시간 (4090 기준 청크 스텝 ≈ t_pre(2048)); 복제는 카드 미상 → 느린 3090 기준이 상한
        card = "3090" if cfg in ("A", "B", "C") else "4090"
        bound = np.array([t_pre(int(s), card) + (q / CHUNK) * t_pre(CHUNK, card) for s, q in zip(S, Q)])
        cov = [(ttft <= bound).mean()]
        p = lambda x, q: np.percentile(x, q)
        lines.append(f"| {cfg} | {cell} | {len(ttft)} | {p(ttft,50)*1e3:.0f}/{p(ttft,90)*1e3:.0f}/{p(ttft,99)*1e3:.0f} | {a*1e3:.0f} | {b*1e6:.1f} | {r2:.2f} | {p(Q,50):.0f}/{p(Q,90):.0f} | {cov[0]:.0%} |")
        allpts.append((cfg, mix, Q, ttft, bound))
    # 그림: 왼쪽 TTFT vs Q (A/B 열린 셀 전부), 오른쪽 실측 TTFT 대 상한
    ax = axes[0]; ax.set_facecolor(SURF); ax.set_title("TTFT vs prefill tokens queued ahead at arrival  (open-arrival cells)", loc="left", fontsize=10, color=INK)
    for cfg, mix, Q, ttft, bound in allpts:
        ax.plot(Q, ttft * 1e3, ".", color=COLOR[cfg], ms=4, alpha=0.5)
    ax.set_xlabel("Q = prefill tokens ahead (waiting for first token)", color=INK2, fontsize=9); ax.set_ylabel("TTFT (ms)", color=INK2, fontsize=9)
    ax = axes[1]; ax.set_facecolor(SURF); ax.set_title("measured TTFT vs HOL upper bound  (points below the line are covered)", loc="left", fontsize=10, color=INK)
    for cfg, mix, Q, ttft, bound in allpts:
        ax.plot(bound * 1e3, ttft * 1e3, ".", color=COLOR[cfg], ms=4, alpha=0.5)
    hi = max(max(b.max(), t.max()) for *_, t, b in allpts) * 1e3 * 1.05; ax.plot([0, hi], [0, hi], "--", color=INK2, lw=1)
    ax.set_xlabel("bound = T_pre(mine) + Q/2048 · T_chunk  (ms)", color=INK2, fontsize=9); ax.set_ylabel("measured TTFT (ms)", color=INK2, fontsize=9)
    for ax in axes:
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(colors=INK2, labelsize=8.5)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=COLOR[c], marker="o", ls="", label=c) for c in COLOR if any(p[0] == c for p in allpts)], frameon=False, fontsize=9, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.06, 1, 1)); p = os.path.join(OPEN, "fig4_hol_ttft.png"); fig.savefig(p, dpi=160); plt.close(fig)
    open(os.path.join(OPEN, "hol.md"), "w").write("\n".join(lines) + "\n"); print("\n".join(lines)); print(p)


if __name__ == "__main__":
    main()
