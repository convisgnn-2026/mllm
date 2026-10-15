#!/usr/bin/env python3
"""1단계 지도 채점: 구성 × 워크로드 × 도착률 → SAR(Scale 1/2/3), 통과 최대 도착률, 전력.
SLO = 3090 무경합 실측 기준 × (TTFT 2, TPOT 1.25) × Scale, 요청별 만족, SAR ≥ 90%.
  python bench/hetero/map_score.py [configs...]     → results/hetero/open/map.md (+ 표 stdout)
"""
import csv, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from splitwise_score import ref

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "hetero", "open")
TPOT_MULT = float(os.environ.get("TPOT_MULT", "1.25")); TAG = os.environ.get("MAP_TAG", "")
WL = ["code", "conv", "decode"]; RATES = ["0.5", "1.0", "1.5", "2.0", "2.5", "3.0"]; SCALES = [1, 2, 3]; SAR_MIN = 0.9
COST_H = {"S3": 1499 / (4 * 365 * 24), "S4": 1599 / (4 * 365 * 24)}; COST_H.update({c: COST_H["S3"] + COST_H["S4"] for c in "ABCD"})   # 상각 $/h, 4년


def cell(c, w, r):
    d = os.path.join(ROOT, c, f"{w}_r{r}")
    if not os.path.exists(os.path.join(d, "summary.json")): return None
    s = json.load(open(os.path.join(d, "summary.json"))); tt, tp = [], []
    if s.get("capped"): return dict(thr=0.0, sar={sc: 0.0 for sc in SCALES}, ttft50=float("inf"), tpot50=float("inf"), capped=True)
    for x in csv.DictReader(open(os.path.join(d, "requests.csv"))):
        if x["err"] or x["first"] in ("nan", ""): continue
        g = int(x.get("usage_completion") or x["out_tokens"]); rt, rb, _ = ref(int(x["in_len"]), g)
        tt.append((float(x["first"]) - float(x["launch"])) / rt); tp.append(((float(x["end"]) - float(x["first"])) / max(g - 1, 1)) / rb)
    tt, tp = np.array(tt), np.array(tp)
    out = dict(thr=s["req_per_s"], sar={sc: float(((tt <= 2 * sc) & (tp <= TPOT_MULT * sc)).mean()) for sc in SCALES}, ttft50=float(np.median(tt)), tpot50=float(np.median(tp)))
    pw = os.path.join(d, "power.json")
    if os.path.exists(pw): out["w"] = json.load(open(pw))["mean_total_w"]
    return out


def main():
    cfgs = sys.argv[1:] or ["S3", "S4", "A", "B", "C", "D"]; D = {(c, w, r): cell(c, w, r) for c in cfgs for w in WL for r in RATES}
    L = [f"# 1단계 실측 지도 — SAR % (Scale 1 / 2 / 3), 달성 rps, 평균 전력 W   [TTFT 기준 하한 {os.environ.get('TTFT_FLOOR','0')} s, TPOT 배율 {TPOT_MULT}]", ""]
    for w in WL:
        L += [f"## {w}", "", "| 도착률 | " + " | ".join(cfgs) + " |", "|---|" + "---|" * len(cfgs)]
        for r in RATES:
            cells = []
            for c in cfgs:
                x = D[(c, w, r)]
                cells.append("–" if not x else f"{x['sar'][1]*100:.0f}/{x['sar'][2]*100:.0f}/{x['sar'][3]*100:.0f}  ({x['thr']:.2f} rps" + (f", {x['w']:.0f} W)" if "w" in x else ")"))
            L.append(f"| {r} | " + " | ".join(cells) + " |")
        L += ["", "**통과 최대 도착률 (SAR ≥ 90%)**", "", "| Scale | " + " | ".join(cfgs) + " | 가장 싼 통과 구성 |", "|---|" + "---|" * (len(cfgs) + 1)]
        for sc in SCALES:
            mx = {}
            for c in cfgs:
                ok = [float(r) for r in RATES if D[(c, w, r)] and D[(c, w, r)]["sar"][sc] >= SAR_MIN]; mx[c] = max(ok) if ok else 0
            cheapest = []
            for r in RATES:
                cand = [c for c in cfgs if mx[c] >= float(r)]
                cheapest.append(min(cand, key=lambda c: COST_H[c]) if cand else "–")
            L.append(f"| ×{sc} | " + " | ".join(f"{mx[c]:.1f}" if mx[c] else "–" for c in cfgs) + " | " + " ".join(f"{r}→{x}" for r, x in zip(RATES, cheapest)) + " |")
        L.append("")
    open(os.path.join(ROOT, f"map{TAG}.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
