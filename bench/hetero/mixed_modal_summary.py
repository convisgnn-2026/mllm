#!/usr/bin/env python3
"""혼재(LLM+DiT) 결과 채점. 칸별 LLM(TTFT/TPOT p50/p90, SAR) + DiT(E2E p50/p90, SAR) + 전력.
SLO 는 절대값 스윕: LLM TTFT {0.5,1,2} s × TPOT {50,100,150} ms, DiT E2E {3,5,10} s. 요청별 만족 → SAR(%).
출력 results/hetero/mixed_modal/summary.md, fig_mixed_modal.png
"""
import csv, glob, json, os, re
import numpy as np
R = "results/hetero/mixed_modal"
TTFT = [0.5, 1.0, 2.0]; TPOT = [0.05, 0.10, 0.15]; DE2E = [3.0, 5.0, 10.0]
PLACE = ["none", "4090", "3090", "split"]; LLMS = ["none", "S4", "C"]

def rd(p):
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []
def f(x):
    try: return float(x)
    except: return np.nan
def power(cell):
    rows = rd(os.path.join(cell, "gpu.csv")); w = {}
    for r in rows:
        k = list(r.keys())
        try: idx = r[k[1]].strip(); pw = float(r[k[4]].replace(" W", ""))
        except: continue
        w.setdefault(idx, []).append(pw)
    return {("4090" if i == "0" else "3090"): float(np.mean(v)) for i, v in w.items()}
def llm_metrics(d):
    rows = [r for r in rd(os.path.join(d, "requests.csv")) if not r.get("err")]
    if not rows: return None
    tt = np.array([f(r["ttft"]) for r in rows]); tp = np.array([f(r["tpot"]) for r in rows])
    m = dict(n=len(rows), ttft_p50=np.nanpercentile(tt, 50), ttft_p90=np.nanpercentile(tt, 90), tpot_p50=np.nanpercentile(tp, 50), tpot_p90=np.nanpercentile(tp, 90))
    for a in TTFT:
        for b in TPOT: m[f"sar_{a}_{int(b*1000)}"] = 100 * np.mean((tt <= a) & (np.nan_to_num(tp, nan=0) <= b))
    return m
def dit_metrics(d):
    rows = [r for r in rd(os.path.join(d, "requests.csv")) if not r.get("err")]
    if not rows: return None
    e = np.array([f(r["e2e"]) for r in rows]); q = np.array([f(r["queue"]) for r in rows]); dn = np.array([f(r["denoise"]) for r in rows])
    m = dict(n=len(rows), e2e_p50=np.percentile(e, 50), e2e_p90=np.percentile(e, 90), queue_p50=np.percentile(q, 50), denoise_mean=np.mean(dn))
    for a in DE2E: m[f"sar_{int(a)}"] = 100 * np.mean(e <= a)
    return m

cells = []
for cell in sorted(glob.glob(f"{R}/*_*/code_L*_D*")):
    cfg = os.path.basename(os.path.dirname(cell)); mm = re.match(r"code_L([\d.]+)_D([\d.]+)", os.path.basename(cell))
    llm, dit = cfg.split("_"); lr, dr = float(mm.group(1)), float(mm.group(2))
    cells.append(dict(llm=llm, dit=dit, lr=lr, dr=dr, L=llm_metrics(os.path.join(cell, "llm")), D=dit_metrics(os.path.join(cell, "dit")), W=power(cell)))
json.dump(cells, open(f"{R}/cells.json", "w"), indent=1, default=float)

L = []
L.append("# 혼재(LLM 3B + DiT SD3.5-medium) 정적 배치 결과 — code 트레이스, 512² 28스텝\n")
L.append("구성 = LLM 구성_DiT 위치. none = 그 워크로드 없음. split = Denoise 4090 / 인코더·VAE 3090. LLM 메모리 비율 0.5 고정.\n")
L.append("## 1. 원시 지표 (LLM: TTFT/TPOT p50/p90, DiT: E2E p50/p90, 대기 p50, Denoise 평균; 전력 W 4090/3090)\n")
L.append("| 구성 | LLM rps | DiT rps | TTFT p50/p90 ms | TPOT p50/p90 ms | DiT E2E p50/p90 s | DiT 대기 p50 s | Denoise s | W 4090 | W 3090 |")
L.append("|---|---|---|---|---|---|---|---|---|---|")
for c in cells:
    l, d, w = c["L"], c["D"], c["W"]
    ls = f"{l['ttft_p50']*1e3:.0f}/{l['ttft_p90']*1e3:.0f} | {l['tpot_p50']*1e3:.1f}/{l['tpot_p90']*1e3:.1f}" if l else "– | –"
    ds = f"{d['e2e_p50']:.2f}/{d['e2e_p90']:.2f} | {d['queue_p50']:.2f} | {d['denoise_mean']:.2f}" if d else "– | – | –"
    L.append(f"| {c['llm']}_{c['dit']} | {c['lr']} | {c['dr']} | {ls} | {ds} | {w.get('4090',0):.0f} | {w.get('3090',0):.0f} |")

L.append("\n## 2. 단독 대비 간섭 (혼재 칸 / 같은 도착률의 단독 칸)\n")
L.append("| 구성 | LLM rps | DiT rps | TTFT p90 ×배 | TPOT p90 ×배 | DiT E2E p90 ×배 | Denoise ×배 |")
L.append("|---|---|---|---|---|---|---|")
def find(llm, dit, lr, dr):
    for c in cells:
        if c["llm"] == llm and c["dit"] == dit and abs(c["lr"] - lr) < 1e-9 and abs(c["dr"] - dr) < 1e-9: return c
for c in cells:
    if c["llm"] == "none" or c["dit"] == "none": continue
    b_l = find(c["llm"], "none", c["lr"], 0.0); b_d = find("none", c["dit"], 0.0, c["dr"])
    if not (b_l and b_d and b_l["L"] and b_d["D"]): continue
    L.append(f"| {c['llm']}_{c['dit']} | {c['lr']} | {c['dr']} | {c['L']['ttft_p90']/b_l['L']['ttft_p90']:.2f} | {c['L']['tpot_p90']/b_l['L']['tpot_p90']:.2f} | "
             f"{c['D']['e2e_p90']/b_d['D']['e2e_p90']:.2f} | {c['D']['denoise_mean']/b_d['D']['denoise_mean']:.2f} |")

L.append("\n## 3. SAR 스윕 (%) — LLM: TTFT×TPOT 아홉 조합, DiT: E2E 세 값\n")
hdr = " | ".join(f"L{a}s/{int(b*1000)}" for a in TTFT for b in TPOT) + " | " + " | ".join(f"D{int(a)}s" for a in DE2E)
L.append(f"| 구성 | LLM rps | DiT rps | {hdr} |"); L.append("|---|---|---|" + "---|" * (len(TTFT) * len(TPOT) + len(DE2E)))
for c in cells:
    l, d = c["L"], c["D"]
    ls = " | ".join(f"{l[f'sar_{a}_{int(b*1000)}']:.0f}" if l else "–" for a in TTFT for b in TPOT)
    ds = " | ".join(f"{d[f'sar_{int(a)}']:.0f}" if d else "–" for a in DE2E)
    L.append(f"| {c['llm']}_{c['dit']} | {c['lr']} | {c['dr']} | {ls} | {ds} |")
open(f"{R}/summary.md", "w").write("\n".join(L) + "\n")

# 그림: (LLM 3.0 rps, DiT 0.3 rps) 와 (LLM 2.0, DiT 0.3) 에서 단독 vs 혼재 네 배치 — TTFT p90, TPOT p90, DiT E2E p90
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; plt.rcParams["font.family"]=["Noto Sans CJK KR","DejaVu Sans"]
fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
for row, (lr, dr) in enumerate([(2.0, 0.3), (3.0, 0.3)]):
    names = ["단독", "C+DiT 4090", "C+DiT 3090", "C+분리", "S4+DiT 3090"]
    cs = [None, find("C", "4090", lr, dr), find("C", "3090", lr, dr), find("C", "split", lr, dr), find("S4", "3090", lr, dr)]
    bC = find("C", "none", lr, 0.0); bS = find("S4", "none", lr, 0.0); b4 = find("none", "4090", 0.0, dr); b3 = find("none", "3090", 0.0, dr)
    tt = [bC["L"]["ttft_p90"] * 1e3] + [c["L"]["ttft_p90"] * 1e3 for c in cs[1:]]
    tp = [bC["L"]["tpot_p90"] * 1e3] + [c["L"]["tpot_p90"] * 1e3 for c in cs[1:]]
    de = [b4["D"]["e2e_p90"]] + [c["D"]["e2e_p90"] for c in cs[1:]]
    for ax, v, t, ref in zip(axes[row], [tt, tp, de], ["LLM TTFT p90 (ms)", "LLM TPOT p90 (ms)", "DiT E2E p90 (s)"], [None, None, b3["D"]["e2e_p90"]]):
        cols = ["#888", "#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
        ax.bar(range(5), v, color=cols); ax.set_xticks(range(5)); ax.set_xticklabels(names, rotation=20, fontsize=8)
        ax.set_title(f"{t}  [LLM {lr} rps, DiT {dr} rps]", fontsize=9)
        for i, y in enumerate(v): ax.text(i, y, f"{y:.0f}" if y > 20 else f"{y:.1f}", ha="center", va="bottom", fontsize=8)
        if ref: ax.axhline(ref, ls="--", c="#d62728", lw=0.8); ax.text(4.4, ref, "3090 단독", fontsize=7, color="#d62728", va="bottom", ha="right")
axes[0][0].text(0, 1.0, "단독 = C(LLM) / 4090(DiT) 각자 혼자", transform=axes[0][0].transAxes, fontsize=7, va="bottom")
plt.tight_layout(); plt.savefig(f"{R}/fig_mixed_modal.png", dpi=150)
print(open(f"{R}/summary.md").read())
