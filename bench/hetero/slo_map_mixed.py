#!/usr/bin/env python3
"""Splitwise 식으로 본 PD 의 자리: SLO 가 빡빡해질 때 구성별 '지원 가능한 최대 도착률'(SAR ≥ 90%) 과 최선 구성.
축: TPOT SLO (무경합 4090 디코드 17 ms 의 배율 → 절대값) × TTFT SLO (종류별 기준값 conv 1 s / code 0.5 s 의 배율).
데이터: 혼합 Azure 트레이스 1차+2차 실행 로그(구성×도착률별 SAR 은 두 실행 평균).
출력: results/hetero/open/slo_map.md, fig_slo_map.png
"""
import csv, json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_slo_mixed import sources, ROOT, RATES, CFG, COLOR, LABEL, INK, INK2, GRID, SURF

TTFT_BASE = {"conv": 1.0, "code": 0.5}
TPOT_UNC = 0.017                                              # 4090 B=1 디코드 스텝 (무경합)
TPOT_MULT = [1.25, 1.5, 2, 3, 4, 6, 9]                         # Splitwise: p50 1.25×, p90 1.5×
TTFT_SCALE = [0.5, 1, 2, 3, 5, 8]
SAR_MIN = 0.9


def load_all(src):
    D = {}
    for c in CFG:
        for r in RATES:
            runs = []
            for tag in ("mixed_azure_v1_r", "mixed_azure_r"):
                d = os.path.join(ROOT, c, f"{tag}{r}")
                if not os.path.exists(os.path.join(d, "requests.csv")): continue
                reqs = []
                for x in csv.DictReader(open(os.path.join(d, "requests.csv"))):
                    if x["err"] or x["first"] in ("nan", ""): continue
                    g = int(x.get("usage_completion") or x["out_tokens"])
                    reqs.append((src[int(x["idx"])], float(x["first"]) - float(x["launch"]), (float(x["end"]) - float(x["first"])) / max(g - 1, 1)))
                runs.append(reqs)
            D[(c, r)] = runs
    return D


def sar(runs, ttft_scale, tpot_abs):
    v = []
    for reqs in runs:
        ok = sum((tt <= TTFT_BASE[k] * ttft_scale) and (tp <= tpot_abs) for k, tt, tp in reqs); v.append(ok / len(reqs))
    return float(np.mean(v))


def max_rate(runs_by_rate, ttft_scale, tpot_abs):
    """SAR ≥ 90% 를 만족하는 최대 도착률 (측정 격자 위; 없으면 0)."""
    best = 0.0
    for r in RATES:
        if sar(runs_by_rate[r], ttft_scale, tpot_abs) >= SAR_MIN: best = float(r)
    return best


def main():
    src = sources(); D = load_all(src)
    L = ["# SLO 지도 — 구성별 지원 가능한 최대 도착률 (SAR ≥ 90%, 혼합 Azure 트레이스, 1차+2차 평균)", "",
         f"TPOT SLO = 무경합 4090 디코드 {TPOT_UNC*1e3:.0f} ms × 배율.  TTFT SLO = (conv 1 s / code 0.5 s) × 배율.  격자 {RATES} rps 밖은 0 또는 4.5+.", ""]
    M = {}                                                     # (ttft_scale, tpot_mult) -> {cfg: max_rate}
    for ts in TTFT_SCALE:
        L += [f"## TTFT × {ts}", "", "| TPOT SLO | " + " | ".join(CFG) + " | 최선 |", "|---|" + "---|" * (len(CFG) + 1)]
        for tm in TPOT_MULT:
            ta = TPOT_UNC * tm; row = {c: max_rate({r: D[(c, r)] for r in RATES}, ts, ta) for c in CFG}; M[(ts, tm)] = row
            mx = max(row.values()); best = "–" if mx == 0 else "/".join(c for c in CFG if row[c] == mx)
            L.append(f"| {tm}× = {ta*1e3:.0f} ms | " + " | ".join(f"{row[c]:.1f}" if row[c] else "–" for c in CFG) + f" | {best} |")
        L.append("")
    open(os.path.join(ROOT, "slo_map.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))

    # 그림: (1) TTFT×2 에서 TPOT SLO 별 최대 도착률 곡선  (2) 최선 구성 지도  (3) 간섭: TPOT p90 vs 도착률 (B vs D)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), gridspec_kw=dict(width_ratios=[1.1, 1.2, 1])); fig.patch.set_facecolor(SURF)
    for ax in axes:
        ax.set_facecolor(SURF); ax.tick_params(colors=INK2, labelsize=8.5)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
    ax = axes[0]; ts = 2; ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_title(f"max arrival rate with SAR ≥ 90%   (TTFT SLO ×{ts})", loc="left", fontsize=10.5, color=INK, pad=8)
    xs = [TPOT_UNC * m * 1e3 for m in TPOT_MULT]
    for c in CFG:
        ys = [M[(ts, m)][c] for m in TPOT_MULT]; ax.plot(xs, ys, "-o", color=COLOR[c], lw=2, ms=6, mec=SURF, mew=1.5, label=LABEL[c])
    ax.set_xscale("log"); ax.set_xticks(xs); ax.set_xticklabels([f"{x:.0f}" for x in xs]); ax.minorticks_off()
    ax.set_xlabel("TPOT SLO (ms)  ← tight        loose →", color=INK2, fontsize=9); ax.set_ylabel("max arrival rate (req/s)", color=INK2, fontsize=9); ax.set_ylim(0, 4.9)
    ax.text(xs[0], 4.6, "0 = no rate on grid meets SLO", color=INK2, fontsize=7.5)
    ax = axes[1]; ax.set_title("best configuration (highest supported rate)", loc="left", fontsize=10.5, color=INK, pad=8)
    idx = {c: i for i, c in enumerate(CFG)}; grid = np.full((len(TTFT_SCALE), len(TPOT_MULT)), np.nan); txt = {}
    for i, ts_ in enumerate(TTFT_SCALE):
        for j, tm in enumerate(TPOT_MULT):
            row = M[(ts_, tm)]; mx = max(row.values())
            if mx == 0: txt[(i, j)] = "–"; continue
            win = [c for c in CFG if row[c] == mx]
            pref = [c for c in win if c in "CD"] or win     # 동률이면 PD 를 표시하지 않고 사전순 첫 구성 (B 우선)
            b = sorted(win)[0]; grid[i, j] = idx[b]; txt[(i, j)] = "".join(win) + f"\n{mx:.1f}"
    cmap = ListedColormap([COLOR[c] for c in CFG]); ax.imshow(grid, cmap=cmap, vmin=-0.5, vmax=3.5, aspect="auto", alpha=0.85, origin="lower")
    for (i, j), s in txt.items(): ax.text(j, i, s, ha="center", va="center", fontsize=7.5, color="white" if s != "–" else INK2, fontweight="bold")
    ax.set_xticks(range(len(TPOT_MULT))); ax.set_xticklabels([f"{TPOT_UNC*m*1e3:.0f}" for m in TPOT_MULT]); ax.set_yticks(range(len(TTFT_SCALE))); ax.set_yticklabels([f"×{s}" for s in TTFT_SCALE])
    ax.set_xlabel("TPOT SLO (ms)", color=INK2, fontsize=9); ax.set_ylabel("TTFT SLO scale (conv 1 s / code 0.5 s)", color=INK2, fontsize=9)
    ax = axes[2]; ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_title("interference: TPOT p90 vs load (Splitwise Insight)", loc="left", fontsize=10.5, color=INK, pad=8)
    for c in CFG:
        ys = [np.mean([np.percentile([tp for _, _, tp in reqs], 90) * 1e3 for reqs in D[(c, r)]]) for r in RATES]
        ax.plot([float(r) for r in RATES], ys, "-o", color=COLOR[c], lw=2, ms=5.5, mec=SURF, mew=1.5); ax.text(4.55, ys[-1], c, color=INK2, fontsize=8.5, va="center")
    ax.axhline(TPOT_UNC * 1e3, color=INK2, lw=0.8, ls=":"); ax.text(2.5, TPOT_UNC * 1e3 + 3, "uncontended 4090 decode 17 ms", color=INK2, fontsize=7.5)
    ax.set_ylim(0, 140); ax.set_xlabel("arrival rate (req/s)", color=INK2, fontsize=9); ax.set_ylabel("TPOT p90 (ms)", color=INK2, fontsize=9)
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, fontsize=9, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Where PD wins on this box (4090 + 3090, Llama-3.1-8B, mixed Azure trace): tight TPOT SLO and loose TTFT SLO", x=0.01, ha="left", fontsize=10, color=INK2)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95)); p = os.path.join(ROOT, "fig_slo_map.png"); fig.savefig(p, dpi=160); print(p)


if __name__ == "__main__":
    main()
