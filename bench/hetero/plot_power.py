#!/usr/bin/env python3
"""혼합 워크로드 구성별 전력 그림: 합계 평균 전력 / 카드별 평균 전력 / 요청당 에너지 (도착률 축). power.json 사용."""
import json, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "hetero", "open")
RATES = ["2.5", "3.0", "3.5", "4.0", "4.5"]; CFG = "ABCD"
COLOR = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "D": "#eda100"}
LABEL = {"A": "A  replicate + RR", "B": "B  replicate + 2:1", "C": "C  PD 4090→3090", "D": "D  PD 3090→4090"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
P = {(c, r): json.load(open(os.path.join(ROOT, c, f"mixed_azure_r{r}", "power.json"))) for c in CFG for r in RATES}
idle = json.load(open(os.path.join(ROOT, "..", "power_idle.json")))
xs = [float(r) for r in RATES]
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4)); fig.patch.set_facecolor(SURF)
for ax in axes:
    ax.set_facecolor(SURF); ax.tick_params(colors=INK2, labelsize=8.5); ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
    ax.set_xlabel("arrival rate (req/s)", color=INK2, fontsize=9); ax.set_xticks(xs)
ax = axes[0]; ax.set_title("total mean power, both cards (W)", loc="left", fontsize=10.5, color=INK, pad=8)
for c in CFG:
    ys = [P[(c, r)]["mean_total_w"] for r in RATES]; ax.plot(xs, ys, "-o", color=COLOR[c], lw=2, ms=6, mec=SURF, mew=1.5, label=LABEL[c]); ax.text(xs[-1] + 0.05, ys[-1], c, color=INK2, fontsize=8.5, va="center")
ax.axhline(450 + 370, color=INK2, lw=0.8, ls="--"); ax.text(2.5, 450 + 370 - 8, "rated 450 + 370 W", color=INK2, fontsize=7.5, va="top")
ax.axhline(idle["4090_loaded_idle_w"] + idle["3090_loaded_idle_w"], color=INK2, lw=0.8, ls=":"); ax.text(2.5, idle["4090_loaded_idle_w"] + idle["3090_loaded_idle_w"] + 8, "idle, model loaded", color=INK2, fontsize=7.5)
ax.set_ylim(0, 860)
ax = axes[1]; ax.set_title("per-card mean power (W)   solid 4090 · dashed 3090", loc="left", fontsize=10.5, color=INK, pad=8)
for c in CFG:
    ax.plot(xs, [P[(c, r)]["gpu"]["0"]["mean_w"] for r in RATES], "-o", color=COLOR[c], lw=2, ms=5, mec=SURF, mew=1.5)
    ax.plot(xs, [P[(c, r)]["gpu"]["1"]["mean_w"] for r in RATES], "--s", color=COLOR[c], lw=1.6, ms=4.5, mec=SURF, mew=1.2)
ax.axhline(370, color=INK2, lw=0.8, ls="--"); ax.text(2.5, 372, "3090 power limit 370 W", color=INK2, fontsize=7.5, va="bottom"); ax.set_ylim(100, 400)
ax = axes[2]; ax.set_title("energy per request (Wh)", loc="left", fontsize=10.5, color=INK, pad=8)
for c in CFG:
    ys = [P[(c, r)]["wh_per_req"] for r in RATES]; ax.plot(xs, ys, "-o", color=COLOR[c], lw=2, ms=6, mec=SURF, mew=1.5); ax.text(xs[-1] + 0.05, ys[-1], c, color=INK2, fontsize=8.5, va="center")
ax.set_ylim(0.04, 0.09)
h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, fontsize=9, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
fig.suptitle("GPU power on mixed Azure trace (nvidia-smi 1 s samples), RTX 4090 + RTX 3090, Llama-3.1-8B — C/D saturated at ≥4.0 / 4.5 req/s", x=0.01, ha="left", fontsize=10, color=INK2)
fig.tight_layout(rect=(0, 0.07, 1, 0.95)); p = os.path.join(ROOT, "fig_mixed_power.png"); fig.savefig(p, dpi=160); print(p)
