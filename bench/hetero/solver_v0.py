#!/usr/bin/env python3
"""solver v0 — vLLM 실측으로 보정한 해석 모델로, 후보 구성의 닫힌 처리량·TPOT 을 예측하고 최선을 고른다.

후보(레버) : 복제 + 가중 RR (w4090:w3090)      PD (P 카드 → D 카드)
카드 모델  : results/hetero/model/beta*.json (bench/hetero/model_llm.py 의 보정 계수)
용량       : KV 용량 = (mem·util − 가중치) / (KV/token), 요청당 문맥 = S_in + S_out → 동시 실행 상한 cap
닫힌 예측  : 카드가 받은 요청들을 cap 크기 웨이브로 나눠, 웨이브마다  Σ T_pre(청크) + S_out · T_dec(B, K)
PD         : 프리필 카드 직렬 Σ T_pre,  디코드 카드 웨이브,  링크 = n·(a + bytes/BW) 직렬  → max(...) + 파이프라인 채움
검증       : results/hetero/ab/{A,B,C,D}/<cell>/summary.json 과 비교 (승자 일치, MAPE)
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_llm as M

ROOT = M.ROOT; AB = os.path.join(ROOT, "ab"); OUT = os.path.join(ROOT, "solver_v0"); os.makedirs(OUT, exist_ok=True)
BETA = json.load(open(os.path.join(ROOT, "model", "beta.json")))
BETA_D = json.load(open(os.path.join(ROOT, "model", "beta_decode_multi.json")))
MEM = {"4090": 24.0, "3090": 24.0}; UTIL = 0.85
LINK = dict(a=0.005, bw=2.4e9)                      # 4090↔3090 NIXL/UCX 튜닝 실측: 고정 5 ms + 2.4 GB/s
CLASSES = {"prefill": (2048, 128), "decode": (128, 1024)}
MIXES = {"prefill": 1.0, "p75": 0.75, "mixed": 0.5, "p25": 0.25, "decode": 0.0}   # 프리필 중심 비율
NS = [50, 100, 200]
CONFIGS = {"A": ("rep", (1, 1)), "B": ("rep", (2, 1)), "C": ("pd", ("4090", "3090")), "D": ("pd", ("3090", "4090"))}


def t_pre(S, k):
    b = BETA[k]["prefill"]; return b["b1"] * M.t_pre_raw(S, k) + b["b0"]


def t_dec(B, K, k):
    b = BETA_D[k]; c = M.CARDS[k]
    return b["b0"] + b["bc"] * M.f_decode(B, K / max(B, 1)) / c["C"] + b["bw"] * M.BYTES_W / c["BW"] + b["bkv"] * K * M.KV_TOK / c["BW"]


def cap_tokens(k):
    return (MEM[k] * 1024 ** 3 * UTIL - M.BYTES_W) // M.KV_TOK


def reqs(mix, n):
    """(S_in, S_out) 목록. 혼합비대로."""
    np_ = round(n * MIXES[mix]); return [CLASSES["prefill"]] * np_ + [CLASSES["decode"]] * (n - np_)


def card_closed(rs, k, prefill_here=True, decode_here=True):
    """한 카드가 요청 rs 를 닫힌 투입으로 처리하는 시간과 평균 TPOT. 웨이브 모델."""
    if not rs:
        return 0.0, float("nan")
    cap = cap_tokens(k); t = 0.0; tpots = []
    order = sorted(rs, key=lambda r: -(r[0] + r[1]))     # 긴 것부터 (보수적)
    i = 0
    while i < len(order):
        wave, toks = [], 0
        while i < len(order) and toks + order[i][0] + order[i][1] <= cap and len(wave) < 256:
            wave.append(order[i]); toks += order[i][0] + order[i][1]; i += 1
        if not wave:                                       # 한 요청도 안 들어가면 강제
            wave.append(order[i]); i += 1
        B = len(wave)
        if prefill_here:
            t += sum(t_pre(s_in, k) for s_in, _ in wave)  # 프리필 청크는 디코드 스텝을 막음 → 가산
        if decode_here:
            s_out = max(s for _, s in wave); K0 = sum(s_in for s_in, _ in wave)
            # 디코드 스텝: 문맥이 자라며 K 증가, 짧은 요청은 먼저 끝나 B 감소 → 스텝별로 적분
            for step in range(1, s_out):
                alive = [r for r in wave if r[1] > step]; Bt = len(alive)
                if Bt == 0:
                    break
                Kt = sum(s_in + step for s_in, _ in alive)
                dt = t_dec(Bt, Kt, k); t += dt; tpots.append(dt)
    return t, float(np.mean(tpots)) if tpots else float("nan")


def predict(cfg, mix, n):
    kind, arg = CONFIGS[cfg]; rs = reqs(mix, n)
    if kind == "rep":
        w4, w3 = arg; n4 = round(n * w4 / (w4 + w3))
        # 클래스가 섞이면 RR 이 번갈아 보내므로 각 카드가 같은 비율로 받음
        r4 = [r for j, r in enumerate(rs) if (j * (w4 + w3)) % (w4 + w3) < w4 * 1] if False else rs[:n4]
        r3 = rs[n4:]
        # 혼합비 유지: 정렬로 클래스 섞기
        rs_sorted = sorted(rs, key=lambda r: r[0]); r4 = rs_sorted[0::2] if w4 == w3 else rs_sorted[: n4]; r3 = [r for r in rs_sorted if r not in r4] if w4 == w3 else rs_sorted[n4:]
        # 위 트릭은 클래스 편중을 만들 수 있어 단순히 비율대로 각 클래스에서 배분
        pf = [r for r in rs if r == CLASSES["prefill"]]; dc = [r for r in rs if r == CLASSES["decode"]]
        f4 = w4 / (w4 + w3); r4 = pf[: round(len(pf) * f4)] + dc[: round(len(dc) * f4)]; r3 = pf[round(len(pf) * f4):] + dc[round(len(dc) * f4):]
        t4, tp4 = card_closed(r4, "4090"); t3, tp3 = card_closed(r3, "3090")
        mk = max(t4, t3); tpot = np.nanmean([tp4] * len(r4) + [tp3] * len(r3)) if rs else float("nan")
        return dict(makespan=mk, req_per_s=n / mk, tpot_ms=tpot * 1e3, detail=f"4090 {t4:.0f}s/{len(r4)} · 3090 {t3:.0f}s/{len(r3)}")
    P, D = arg
    tP = sum(t_pre(s_in, P) for s_in, _ in rs)                              # 프리필 카드 직렬
    tX = sum(LINK["a"] + s_in * M.KV_TOK / LINK["bw"] for s_in, _ in rs)    # 링크 직렬
    tD, tpD = card_closed(rs, D, prefill_here=False)                          # 디코드 카드 웨이브
    mk = max(tP, tX, tD) + min(tP, tD) / max(len(rs), 1)                     # 파이프라인 채움 ≈ 요청 하나 분
    return dict(makespan=mk, req_per_s=len(rs) / mk, tpot_ms=tpD * 1e3, detail=f"P {tP:.0f}s · link {tX:.0f}s · D {tD:.0f}s")


def measured(cfg, mix, n):
    f = os.path.join(AB, cfg, f"{mix}_n{n}", "summary.json")
    return json.load(open(f)) if os.path.exists(f) else None


def main():
    lines = ["# solver v0 — 닫힌 예측 대 실측 (req/s), 승자 비교", "",
             "| mix | n | " + " | ".join(f"{c} 예측/실측" for c in CONFIGS) + " | 예측 승자 | 실측 승자 |", "|---|---|" + "---|" * (len(CONFIGS) + 2)]
    errs, agree, tot = [], 0, 0
    for mix in MIXES:
        for n in NS:
            cells, pw, mw = [], {}, {}
            for c in CONFIGS:
                p = predict(c, mix, n); m = measured(c, mix, n)
                pw[c] = p["req_per_s"]
                if m:
                    mw[c] = m["req_per_s"]; errs.append((p["req_per_s"] - m["req_per_s"]) / m["req_per_s"])
                    cells.append(f"{p['req_per_s']:.2f}/{m['req_per_s']:.2f}")
                else:
                    cells.append(f"{p['req_per_s']:.2f}/–")
            if mw:
                bp = max(mw, key=lambda c: pw[c]); bm = max(mw, key=mw.get); tot += 1; agree += (bp == bm)
            else:
                bp = bm = "–"
            lines.append(f"| {mix} | {n} | " + " | ".join(cells) + f" | {bp} | {bm} |")
    errs = np.array(errs)
    lines += ["", f"MAPE {np.mean(np.abs(errs)):.1%} (n={len(errs)}), 편향 {np.mean(errs):+.1%}, 승자 일치 {agree}/{tot}", ""]
    lines += ["## 구성별 오차", ""]
    for c in CONFIGS:
        e = [(predict(c, mix, n)["req_per_s"] - measured(c, mix, n)["req_per_s"]) / measured(c, mix, n)["req_per_s"]
             for mix in MIXES for n in NS if measured(c, mix, n)]
        lines.append(f"- {c}: MAPE {np.mean(np.abs(e)):.1%}, 편향 {np.mean(e):+.1%}")
    open(os.path.join(OUT, "README.md"), "w").write("\n".join(lines) + "\n"); print("\n".join(lines))


if __name__ == "__main__":
    main()
