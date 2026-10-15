#!/usr/bin/env python3
"""혼재 그림 v2: 같은 두 카드(4090+3090)에 LLM 과 DiT 를 같이 올린 네 배치만 막대. 혼자 돌린 값은 점선(기준).
입력 results/hetero/mixed_modal/cells.json → fig_mixed_modal_v2.png"""
import json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Noto Sans CJK KR", "DejaVu Sans"]
R = "results/hetero/mixed_modal"; cells = json.load(open(f"{R}/cells.json"))
def find(llm, dit, lr, dr):
    for c in cells:
        if c["llm"] == llm and c["dit"] == dit and abs(c["lr"] - lr) < 1e-9 and abs(c["dr"] - dr) < 1e-9: return c
PL = [("C", "4090", "①\nDiT: 4090"), ("C", "3090", "②\nDiT: 3090"), ("C", "split", "③\nDiT 단계 분리"), ("S4", "3090", "④\nDiT: 3090")]
COL = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
fig, axes = plt.subplots(2, 3, figsize=(14, 8.2))
for row, (lr, dr) in enumerate([(2.0, 0.3), (3.0, 0.3)]):
    cs = [find(l, d, lr, dr) for l, d, _ in PL]
    tt = [c["L"]["ttft_p90"] * 1e3 for c in cs]; tp = [c["L"]["tpot_p90"] * 1e3 for c in cs]; de = [c["D"]["e2e_p90"] for c in cs]
    refC = find("C", "none", lr, 0); refS = find("S4", "none", lr, 0); ref4 = find("none", "4090", 0, dr); ref3 = find("none", "3090", 0, dr)
    panels = [(tt, "LLM TTFT p90 (ms)", [("LLM 단독 실행 (PD 분할)", refC["L"]["ttft_p90"] * 1e3, "#555"), ("LLM 단독 실행 (4090 단일)", refS["L"]["ttft_p90"] * 1e3, "#9467bd")]),
              (tp, "LLM TPOT p90 (ms)", [("LLM 단독 실행 (PD 분할)", refC["L"]["tpot_p90"] * 1e3, "#555"), ("LLM 단독 실행 (4090 단일)", refS["L"]["tpot_p90"] * 1e3, "#9467bd")]),
              (de, "DiT E2E p90 (s)", [("DiT 단독 실행 (4090)", ref4["D"]["e2e_p90"], "#555"), ("DiT 단독 실행 (3090)", ref3["D"]["e2e_p90"], "#d62728")])]
    for ax, (v, title, refs) in zip(axes[row], panels):
        ax.bar(range(4), v, color=COL, width=0.6)
        for i, y in enumerate(v): ax.text(i, y, f"{y:.0f}" if y > 20 else f"{y:.1f}", ha="center", va="bottom", fontsize=9)
        for name, y, col in refs:
            ax.axhline(y, ls="--", c=col, lw=1.2, label=f"{name}: {y:.0f}" if y > 20 else f"{name}: {y:.1f}")
        ax.legend(fontsize=7.5, loc="upper right", frameon=True)
        ax.set_xticks(range(4)); ax.set_xticklabels([p[2] for p in PL], fontsize=9)
        ax.set_title(f"{title}   [LLM {lr} rps + DiT {dr} rps 동시 배치]", fontsize=10)
        ax.set_ylim(0, max(v + [r[1] for r in refs]) * 1.3)
fig.suptitle("4090 + 3090 두 장에 LLM(3B, code)과 DiT(SD3.5-m, 512²)를 동시 배치한 네 구성 — 막대: 동시 배치 측정값, 점선: 단독 실행 기준값", fontsize=11)
fig.text(0.5, 0.012, "① LLM PD 분할(프리필 4090 → 디코드 3090) + DiT 4090      ② LLM PD 분할 + DiT 3090      ③ LLM PD 분할 + DiT 단계 분리(Denoise 4090, 인코더·VAE 3090)      ④ LLM 4090 단일 + DiT 3090",
         ha="center", fontsize=9.5)
plt.tight_layout(rect=(0, 0.035, 1, 0.96)); plt.savefig(f"{R}/fig_mixed_modal_v2.png", dpi=150); print("ok")
