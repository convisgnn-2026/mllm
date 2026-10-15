#!/usr/bin/env python3
"""PD 가 이기는 곳: 워크로드 4종(프리필 중심 / 혼합 50% / 디코드 중심 / Azure 혼합 트레이스) × TPOT SLO 빡빡함 →
구성별 '지원 가능한 최대 도착률'(SAR ≥ 90%, TTFT SLO ×2: 긴 입력 2 s / 짧은 입력 1 s). Splitwise 의 조건(TBT ≤ 1.5× 무경합)이 우리 박스에서 켜지는 지점.
출력 results/hetero/open/fig_pd_where.png, pd_where.md"""
import csv, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_slo_mixed import sources, ROOT, CFG, COLOR, LABEL, INK, INK2, GRID, SURF

TPOT_UNC = 0.017; MULT = [1.25, 1.5, 2, 3, 4, 6, 9]; TTFT_SCALE = 2; SAR_MIN = 0.9
WL = {"prefill-heavy (2048/128)": ("prefill", ["1.1", "2.2", "3.3"]), "mixed 50/50": ("mixed", ["0.8", "1.6", "2.4"]),
      "decode-heavy (128/1024)": ("decode", ["0.7", "1.4", "2.2"]), "Azure trace (code+conv)": ("mixed_azure", ["2.5", "3.0", "3.5", "4.0", "4.5"])}


def load(c, name, r, src):
    fs = [os.path.join(ROOT, c, f"{name}_r{r}", "requests.csv")]
    if name == "mixed_azure": fs.append(os.path.join(ROOT, c, f"mixed_azure_v1_r{r}", "requests.csv"))
    runs = []
    for f in fs:
        if not os.path.exists(f): continue
        out = []
        for x in csv.DictReader(open(f)):
            if x["err"] or x["first"] in ("nan", ""): continue
            g = int(x.get("usage_completion") or x["out_tokens"]); il = int(x["in_len"])
            cls = src[int(x["idx"])] if name == "mixed_azure" else ("code" if il > 1000 else "conv")
            out.append((cls, float(x["first"]) - float(x["launch"]), (float(x["end"]) - float(x["first"])) / max(g - 1, 1)))
        runs.append(out)
    return runs


def sar(runs, ta):
    base = {"conv": 1.0, "code": 0.5}
    return float(np.mean([np.mean([(tt <= base[k] * TTFT_SCALE) and (tp <= ta) for k, tt, tp in reqs]) for reqs in runs])) if runs else np.nan


def main():
    src = sources(); L = [f"# PD 가 이기는 곳 — SAR ≥ 90% 최대 도착률 (TTFT SLO ×{TTFT_SCALE}: code/긴 입력 1 s, conv/짧은 입력 2 s)", ""]
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.3), sharey=False); fig.patch.set_facecolor(SURF)
    for ax, (title, (name, rates)) in zip(axes, WL.items()):
        D = {(c, r): load(c, name, r, src) for c in CFG for r in rates}
        L += [f"## {title}  (도착률 격자 {rates})", "", "| TPOT SLO | " + " | ".join(CFG) + " | 최선 |", "|---|" + "---|" * 5]
        ax.set_facecolor(SURF); ax.tick_params(colors=INK2, labelsize=8.5); ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
        ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=8)
        xs = [TPOT_UNC * m * 1e3 for m in MULT]; Y = {c: [] for c in CFG}
        for m in MULT:
            ta = TPOT_UNC * m; row = {}
            for c in CFG:
                ok = [float(r) for r in rates if D[(c, r)] and sar(D[(c, r)], ta) >= SAR_MIN]; row[c] = max(ok) if ok else 0.0; Y[c].append(row[c])
            mx = max(row.values()); L.append(f"| {m}× = {ta*1e3:.0f} ms | " + " | ".join(f"{row[c]:.1f}" if row[c] else "–" for c in CFG) + f" | {'–' if mx == 0 else '/'.join(c for c in CFG if row[c] == mx)} |")
        for k, c in enumerate(CFG):
            ax.plot(xs, np.array(Y[c]) + (k - 1.5) * 0.02, "-o", color=COLOR[c], lw=2, ms=6, mec=SURF, mew=1.5, label=LABEL[c])
        ax.set_xscale("log"); ax.set_xticks(xs); ax.set_xticklabels([f"{x:.0f}" for x in xs]); ax.minorticks_off()
        ax.axvline(TPOT_UNC * 1.5e3, color=INK2, lw=0.8, ls="--"); ax.text(TPOT_UNC * 1.5e3 * 1.03, 0.15, "Splitwise TBT SLO\n(1.5× uncontended)", color=INK2, fontsize=7)
        ax.set_xlabel("TPOT SLO (ms)   tight → loose", color=INK2, fontsize=9); ax.set_ylim(0, max(float(r) for r in rates) + 0.4)
        ax.set_yticks([0] + [float(r) for r in rates]); L.append("")
    axes[0].set_ylabel("max arrival rate with SAR ≥ 90% (req/s)\n0 = none on grid", color=INK2, fontsize=9)
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, fontsize=9, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Where PD wins on RTX 4090 + 3090 (Llama-3.1-8B): supported load vs TPOT SLO tightness, per workload", x=0.01, ha="left", fontsize=10, color=INK2)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95)); p = os.path.join(ROOT, "fig_pd_where.png"); fig.savefig(p, dpi=160)
    open(os.path.join(ROOT, "pd_where.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L)); print(p)


if __name__ == "__main__":
    main()
