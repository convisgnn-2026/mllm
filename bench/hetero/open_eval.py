#!/usr/bin/env python3
"""열린 도착 평가 v1.1 — 클래스 × 카드 단위로 예측 대 실측 (p50/p90/p99), 동시 상주 KV 로 PD 실현가능성 표시.

모델 v1.1 = throughput_model 의 스텝 시뮬 + TTFT 상수 c0 (A/B 열린 셀에서 적합한 49 ms; PD 는 전송·프록시 왕복 c0_pd 추가).
  주의: c0 는 열린 도착 A/B 셀을 본 뒤 넣은 값이므로 A/B 의 TTFT 는 '적합', C/D 와 고부하 격자가 '검증'.
동시 상주 KV (디코더 또는 카드):  R = Σ_class λ_c · 체류시간_c · 평균문맥_c ,  체류시간 = S_out · T_dec
"""
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import throughput_model as T
import model_llm as M

OPEN = os.path.join(M.ROOT, "open")
P = json.load(open(os.path.join(T.OUT, "throughput_params.json"))); T.CHUNK = P["chunk"]; BM = P["beta_mix"]
C0 = 0.049            # TTFT 상수 (프록시·API·스텝 정렬), A/B 열린 셀 적합
C0_PD = 0.110         # PD 추가분 (프록시가 프리필 응답을 받고 디코더 요청을 새로 여는 왕복), C 프리필 λ=1.1 한 셀로 적합
SEQ = {"A": ["4090", "3090"], "B": ["4090", "4090", "3090"]}
WARM = 8              # 워밍업 요청 수 — 프록시의 RR 순환을 먼저 돌려놓음 (B 는 주기 3 이라 2칸 밀림)
PD = {"C": ("4090", "3090"), "D": ("3090", "4090")}
def cls_of(S):
    return "decode128" if S < 1000 else "prefill2048"


def measured(cfg, cell):
    f = os.path.join(OPEN, cfg, cell, "requests.csv")
    if not os.path.exists(f) or not os.path.exists(os.path.join(OPEN, cfg, cell, "summary.json")):
        return None
    out = []
    for r in csv.DictReader(open(f)):
        if r["err"] != "" or r["first"] in ("nan", ""):
            continue
        i = int(r["idx"]); card = SEQ[cfg][(i + WARM) % len(SEQ[cfg])] if cfg in SEQ else PD[cfg][1]
        gen = int(r.get("usage_completion") or r["out_tokens"])
        out.append(dict(idx=i, S=int(r["in_len"]), card=card, ttft=float(r["first"]) - float(r["launch"]),
                        tpot=(float(r["end"]) - float(r["first"])) / max(gen - 1, 1), end=float(r["end"])))
    return out


def predicted(cfg, mix, n, rate, seed=0):
    rs = T.reqs_for(mix, n, seed); arr = T.arrivals_for(n, rate, seed); out = []
    if cfg in SEQ:
        seq = SEQ[cfg]
        for k in ("4090", "3090"):
            idx = [j for j in range(n) if seq[(j + WARM) % len(seq)] == k]
            if not idx:
                continue
            mk, tt, tp = T.simulate_card_arrivals([rs[j] for j in idx], k, BM, [arr[j] for j in idx], prefilled=False)
            order = sorted(range(len(idx)), key=lambda q: arr[idx[q]])      # simulate 는 도착 순으로 돌지만 목록은 요청 순 → 길이만 맞춰 매핑
            for q, j in enumerate(idx):
                out.append(dict(idx=j, S=rs[j][0], card=k, ttft=(tt[q] if q < len(tt) else np.nan) + C0, tpot=tp[q] if q < len(tp) else np.nan, end=mk))
        mk = max(o["end"] for o in out)
    else:
        Pc, Dc = PD[cfg]; p = T.predict_open_pd(mix, n, rate, Pc, Dc, BM, seed)
        for j in range(n):
            out.append(dict(idx=j, S=rs[j][0], card=Dc, ttft=p["ttft"][j] + C0 + C0_PD + T.t_dec(1, rs[j][0], Dc), tpot=p["tpot"][j] if j < len(p["tpot"]) else np.nan, end=p["makespan"]))
        mk = p["makespan"]
    return out, n / mk


def resident_kv(cfg, mix, rate):
    """동시 상주 KV (토큰) 와 용량. 복제는 카드별(배정 비율), PD 는 디코더 한 장."""
    fr = T.MIXES[mix]; cls = [(2048, 128, fr), (128, 1024, 1 - fr)]
    def R(k, lam):
        tot = 0.0
        for s_in, s_out, f in cls:
            res = s_out * T.t_dec(8, 8 * (s_in + s_out / 2), k)             # 체류시간 ≈ 출력 길이 × 스텝 시간(대표 배치 8)
            tot += lam * f * res * (s_in + s_out / 2)
        return tot
    if cfg in SEQ:
        w = SEQ[cfg]; f4 = w.count("4090") / len(w)
        return max(R("4090", rate * f4) / T.KV_CAP["4090"], R("3090", rate * (1 - f4)) / T.KV_CAP["3090"])
    return R(PD[cfg][1], rate) / T.KV_CAP_PD[PD[cfg][1]]


def pct(v, q):
    v = np.array([x for x in v if x == x]); return float(np.percentile(v, q)) if len(v) else float("nan")


def main():
    cells = sorted({(f.split(os.sep)[-3], f.split(os.sep)[-2]) for f in glob.glob(os.path.join(OPEN, "*", "*_r*", "summary.json"))})
    L1 = ["# 열린 도착: 셀 요약 (달성 처리량, 동시 상주 KV/용량)", "", "| cfg | cell | λ | 처리량 예/실 | 상주KV/용량 | 실측 상태 |", "|---|---|---|---|---|---|"]
    agg = {}
    for cfg, cell in cells:
        mix, rate = cell.split("_r"); rate = float(rate); m = measured(cfg, cell)
        if not m:
            continue
        s = json.load(open(os.path.join(OPEN, cfg, cell, "summary.json"))); p, thr = predicted(cfg, mix, 150, rate)
        rk = resident_kv(cfg, mix, rate); ok = s["req_per_s"] >= 0.8 * thr        # 포화 = 달성 처리량이 예측의 80% 미만
        L1.append(f"| {cfg} | {cell} | {rate} | {thr:.2f}/{s['req_per_s']:.2f} | {rk:.2f} | {'정상' if ok else '**포화**'} |")
        for src, rows in (("m", m), ("p", p)):
            for r in rows:
                agg.setdefault((cfg, cls_of(r["S"]), r["card"]), {"m": {"ttft": [], "tpot": []}, "p": {"ttft": [], "tpot": []}, "collapse": []})
                a = agg[(cfg, cls_of(r["S"]), r["card"])]
                if ok:                                                    # 포화 셀은 분포 비교에서 제외(별도 표시)
                    a[src]["ttft"].append(r["ttft"]); a[src]["tpot"].append(r["tpot"])
    L2 = ["", "# 클래스 × 카드별 분포 (포화 셀 제외): 실측 / 예측, ms", "", "| cfg | 클래스 | 카드 | n | TTFT p50 | TTFT p90 | TTFT p99 | TPOT p50 | TPOT p90 | TPOT p99 |", "|---|---|---|---|---|---|---|---|---|---|"]
    errs = {"ttft50": [], "ttft90": [], "ttft99": [], "tpot50": [], "tpot90": [], "tpot99": []}
    for (cfg, cl, card), a in sorted(agg.items()):
        if len(a["m"]["ttft"]) < 20:
            continue
        cellsx = []
        for key in ("ttft", "tpot"):
            for q in (50, 90, 99):
                mv, pv = pct(a["m"][key], q) * 1e3, pct(a["p"][key], q) * 1e3
                cellsx.append(f"{mv:.0f} / {pv:.0f}"); errs[f"{key}{q}"].append((pv - mv) / mv)
        L2.append(f"| {cfg} | {cl} | {card} | {len(a['m']['ttft'])} | " + " | ".join(cellsx) + " |")
    L2 += ["", "오차 (예측−실측)/실측, 그룹 평균 |절대| (편향): " + ", ".join(f"{k} {np.mean(np.abs(v)):.0%} ({np.mean(v):+.0%})" for k, v in errs.items() if v)]
    txt = "\n".join(L1 + L2); open(os.path.join(OPEN, "eval_v1_1.md"), "w").write(txt + "\n"); print(txt)


if __name__ == "__main__":
    main()
