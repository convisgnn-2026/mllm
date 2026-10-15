#!/usr/bin/env python3
"""LLM-Pilot Fig.2 식 처리량 그림: x = 부하 수준(LLM rps × DiT rps), 묶음 막대 = 네 배치, y = 달성 처리량.
(a) LLM 출력 tok/s, (b) DiT 이미지/s. 점선 = 혼자 돌렸을 때 달성 처리량(간섭 없음 기준). ★ = 두 SLO(LLM TTFT≤1 s & TPOT≤150 ms, DiT E2E≤5 s, SAR≥90%) 통과,
빨간 ★ = 통과 중 전력(두 카드 합) 최소. → results/hetero/mixed_modal/fig_mixed_modal_tps.png"""
import json, os, glob, re
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Noto Sans CJK KR", "DejaVu Sans"]
R = "results/hetero/mixed_modal"; cells = {(c["llm"], c["dit"], c["lr"], c["dr"]): c for c in json.load(open(f"{R}/cells.json"))}
def summ(llm, dit, lr, dr, kind):
    f = lambda v: "0" if v == 0 else f"{v}"
    p = f"{R}/{llm}_{dit}/code_L{f(lr)}_D{f(dr)}/{kind}/summary.json"
    return json.load(open(p)) if os.path.exists(p) else None
PL = [("C", "4090", "① PD 분할 + DiT 4090", "#1f77b4"), ("C", "3090", "② PD 분할 + DiT 3090", "#d62728"), ("C", "split", "③ PD 분할 + DiT 단계 분리", "#2ca02c"), ("S4", "3090", "④ 4090 단일 + DiT 3090", "#9467bd")]
LOADS = [(1.0, 0.1), (1.0, 0.3), (2.0, 0.1), (2.0, 0.3), (3.0, 0.1), (3.0, 0.3)]
SLO_TTFT, SLO_TPOT, SLO_DIT = 1.0, 150, 5
fig, axes = plt.subplots(1, 2, figsize=(14, 5)); W = 0.19
for ax, kind in zip(axes, ["llm", "dit"]):
    for gi, (lr, dr) in enumerate(LOADS):
        passing = []
        for pi, (llm, dit, name, col) in enumerate(PL):
            c = cells[(llm, dit, lr, dr)]; s = summ(llm, dit, lr, dr, kind)
            y = s["out_tok_per_s"] if kind == "llm" else s["req_per_s"]
            x = gi + (pi - 1.5) * W
            ok = c["L"][f"sar_{SLO_TTFT}_{SLO_TPOT}"] >= 90 and c["D"][f"sar_{SLO_DIT}"] >= 90
            ax.bar(x, y, W, color=col, label=name if gi == 0 else None, alpha=1.0 if ok else 0.45, hatch=None if ok else "//")
            if ok: passing.append((c["W"].get("4090", 0) + c["W"].get("3090", 0), x, y))
        for k, (w, x, y) in enumerate(sorted(passing)):
            ax.text(x, y * 1.02, "★", ha="center", va="bottom", fontsize=11, color="red" if k == 0 else "goldenrod")
        if kind == "llm": off = max(summ("C", "none", lr, 0.0, "llm")["out_tok_per_s"], summ("S4", "none", lr, 0.0, "llm")["out_tok_per_s"])
        else: off = summ("none", "4090", 0.0, dr, "dit")["req_per_s"]
        ax.plot([gi - 2 * W, gi + 2 * W], [off, off], ls="--", c="k", lw=0.9)
    ax.set_xticks(range(len(LOADS))); ax.set_xticklabels([f"LLM {lr}\nDiT {dr}" for lr, dr in LOADS], fontsize=9)
    ax.set_xlabel("제시한 도착률 (req/s)"); ax.grid(axis="y", alpha=0.3)
    ax.set_ylabel("LLM 출력 처리량 (tok/s)" if kind == "llm" else "DiT 처리량 (images/s)")
    ax.set_title("(a) LLM 처리량" if kind == "llm" else "(b) DiT 처리량", fontsize=11)
axes[0].legend(fontsize=8.5, loc="upper left")
fig.suptitle("4090+3090 두 장, LLM(3B code)+DiT(SD3.5-m 512²) 동시 배치 — 막대: 달성 처리량, 점선: 단독 실행 달성값(간섭 없음), "
             "빗금: SLO 미달(LLM TTFT≤1 s·TPOT≤150 ms, DiT E2E≤5 s, SAR≥90%), ★ SLO 통과, 빨간 ★ 통과 중 전력 최소", fontsize=9.5)
plt.tight_layout(rect=(0, 0, 1, 0.94)); plt.savefig(f"{R}/fig_mixed_modal_tps.png", dpi=150); print("ok")
