#!/usr/bin/env python3
"""혼합 워크로드(Azure 트레이스) 로그를 종류별 SLO 로 재채점 → SAR·goodput 표 + 그림.
SLO 기준값(근거: Abbas 2022 응답 지연 허용, Rayner 2016 읽기 속도):
  conv: TTFT ≤ 1.0 s, TPOT ≤ 150 ms     code: TTFT ≤ 0.5 s, TPOT ≤ 150 ms
민감도: TPOT 50 ms (LLM-Pilot 실험값).  SLO Scale 은 TTFT·TPOT 에 같이 곱한다.
  python bench/hetero/score_slo_mixed.py
"""
import csv, json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loadgen import load_trace

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", "..", "results", "hetero", "open")
TRACE = os.path.join(HERE, "traces", "azure_mixed.csv"); RATES = ["2.5", "3.0", "3.5", "4.0", "4.5"]; CFG = "ABCD"
COLOR = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "D": "#eda100"}
LABEL = {"A": "A  replicate + RR", "B": "B  replicate + 2:1", "C": "C  PD 4090→3090", "D": "D  PD 3090→4090"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
BASE = {"conv": (1.0, 0.150), "code": (0.5, 0.150)}          # (TTFT s, TPOT s)


def sources(n=300, seed=0):
    rows = list(csv.DictReader(open(TRACE))); lens, _ = load_trace(TRACE, n, seed)
    # load_trace 와 같은 offset 을 다시 찾는다: 길이 열이 일치하는 첫 연속 구간
    key = [(max(8, min(int(r["ContextTokens"]), 8000)), max(1, int(r["GeneratedTokens"]))) for r in rows]
    for off in range(len(rows) - n + 1):
        if key[off:off + n] == lens:
            return [rows[off + i]["Source"] for i in range(n)]
    raise RuntimeError("trace offset not found")


def load(c, r, src):
    d = os.path.join(ROOT, c, f"mixed_azure_r{r}"); s = json.load(open(os.path.join(d, "summary.json")))
    R = list(csv.DictReader(open(os.path.join(d, "requests.csv")))); out = []
    for x in R:
        if x["err"] or x["first"] in ("nan", ""): continue
        g = int(x.get("usage_completion") or x["out_tokens"]); i = int(x["idx"])
        out.append((src[i], float(x["first"]) - float(x["launch"]), (float(x["end"]) - float(x["first"])) / max(g - 1, 1)))
    return out, s["makespan_s"]


def sar(reqs, scale, tpot_base=None):
    ok = 0
    for cls, tt, tp in reqs:
        T1, T2 = BASE[cls]; T2 = tpot_base if tpot_base else T2
        ok += (tt <= T1 * scale) and (tp <= T2 * scale)
    return ok / len(reqs), ok


def main():
    src = sources(); D = {(c, r): load(c, r, src) for c in CFG for r in RATES}
    nconv = sum(1 for s in src if s == "conv"); L = [f"# 혼합 워크로드 SLO 재채점 — conv {nconv} / code {300-nconv} 건", "",
        "기준값: conv TTFT ≤ 1.0 s · TPOT ≤ 150 ms, code TTFT ≤ 0.5 s · TPOT ≤ 150 ms (Abbas 2022, Rayner 2016). SLO Scale 은 둘에 같이 곱함.", ""]
    settings = [("TPOT 150 ms", None, [1.0, 1.5, 2.0, 3.0]), ("TPOT 50 ms (LLM-Pilot 실험값, 민감도)", 0.05, [1.0, 2.0])]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9), sharey=True); fig.patch.set_facecolor(SURF); ax_i = 0
    for name, tb, scales in settings:
        for sc in scales:
            L += [f"## {name}, SLO Scale {sc}×  — SAR % (goodput req/s)", "", "| rps | A | B | C | D | 최선 |", "|---|---|---|---|---|---|"]
            for r in RATES:
                cells, best = [], None
                for c in CFG:
                    reqs, mk = D[(c, r)]; s_, ok = sar(reqs, sc, tb); cells.append(f"{s_*100:.0f} ({ok/mk:.2f})")
                    if best is None or s_ > best[1]: best = (c, s_)
                L.append(f"| {r} | " + " | ".join(cells) + f" | {best[0]} |")
            L.append("")
            if (tb is None and sc in (1.0, 2.0)) or (tb and sc in (1.0, 2.0)):
                ax = axes[ax_i]; ax_i += 1; ax.set_facecolor(SURF)
                ax.set_title(f"{'TPOT 150 ms' if tb is None else 'TPOT 50 ms'}, scale {sc}×", loc="left", fontsize=10, color=INK, pad=6)
                for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
                for s_ in ("left", "bottom"): ax.spines[s_].set_color(GRID)
                ax.tick_params(colors=INK2, labelsize=8.5); ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
                ax.axhline(90, color=INK2, lw=0.8, ls="--"); ax.text(4.55, 90, "90%", color=INK2, fontsize=7.5, va="bottom", ha="right")
                for c in CFG:
                    ys = [sar(D[(c, r)][0], sc, tb)[0] * 100 for r in RATES]
                    ax.plot([float(r) for r in RATES], ys, "-o", color=COLOR[c], lw=2, ms=5.5, mec=SURF, mew=1.5, label=LABEL[c])
                ax.set_xlabel("arrival rate (req/s)", color=INK2, fontsize=9); ax.set_ylim(0, 102)
    axes[0].set_ylabel("SLO attainment ratio (%)", color=INK2, fontsize=9)
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, fontsize=9, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("SAR on mixed Azure trace — per-class SLO (conv TTFT 1 s / code TTFT 0.5 s), 4090 + 3090", x=0.01, ha="left", fontsize=10, color=INK2)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94)); p = os.path.join(ROOT, "fig_mixed_sar.png"); fig.savefig(p, dpi=160)
    open(os.path.join(ROOT, "mixed_slo.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L)); print(p)


if __name__ == "__main__":
    main()
