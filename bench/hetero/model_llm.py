#!/usr/bin/env python3
"""단일 쿼리(no PD, colo) 해석 모델 — LLM-Pilot §IV-B 를 vLLM 실측(4090·3090 프로파일)에 맞춰 보정.

  T_pre(S)      = max( F_pre(S)/C_k , M_w/BW_k )                       프리필 (연산 vs 가중치 읽기)
  T_dec(B, l)   = F_dec(B, l)/C_k + ( M_w + KV(B, l) )/BW_k             디코드 스텝 (LLM-Pilot 식 (12), α=0)
  T_real        = β1 · T_pred + β0                                      카드·단계별 선형 보정 (LLM-Pilot §IV-C)
  E2E(S_in,S_out) = T_pre(S_in) + Σ_t T_dec(1, S_in + t)                단일 쿼리

vLLM 과 FlexLLMGen 의 차이는 β 로 흡수되는 것(고정 오버헤드, 커널 효율)과 흡수 안 되는 것(연속 배칭의 혼합 스텝,
chunked prefill, 페이지 KV, 선점)으로 나뉜다. 이 파일은 전자까지. 후자는 hetsim 의 스텝 시뮬레이션 몫.
"""
import glob
import json
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "hetero")
PROF = os.path.join(ROOT, "profile_llm")
OUT = os.path.join(ROOT, "model")

# Llama-3.1-8B (bf16)
L, H, FFN, KVH, DH = 32, 4096, 14336, 8, 128
PARAMS = 8.03e9                       # 전체 (임베딩·LM head 포함; 매 토큰 LM head 도 곱해지므로 포함)
BYTES_W = PARAMS * 2                  # 16.06 GB
KV_TOK = 2 * L * KVH * DH * 2         # 131072 B/token (K,V × 32 layers × 8 heads × 128 × bf16)

# 카드 사양 (공개값): bf16 dense TFLOPS, HBM GB/s
CARDS = {"4090": dict(C=165e12, BW=1008e9), "3090": dict(C=71e12, BW=936e9)}


def f_prefill(S):        # FLOPs: 선형층 2·P·S + 어텐션 QKᵀ, AV (2·2·S²·H per layer, causal 이지만 커널은 대략 절반 → ×0.5)
    return 2 * PARAMS * S + 0.5 * 4 * L * H * S * S


def f_decode(B, l):      # 토큰당 2·P + 어텐션 4·L·H·l
    return B * (2 * PARAMS + 4 * L * H * l)


def t_pre_raw(S, k):
    c = CARDS[k]; return max(f_prefill(S) / c["C"], BYTES_W / c["BW"])


def t_dec_raw(B, l, k):
    c = CARDS[k]; return f_decode(B, l) / c["C"] + (BYTES_W + B * l * KV_TOK) / c["BW"]


def load(k):
    d = {}
    for f in glob.glob(os.path.join(PROF, k, "*.json")):
        d[os.path.basename(f)[:-5]] = json.load(open(f))["avg_latency"]
    return d


def points(k):
    """(단계, 설명, 예측 raw s, 실측 s). 디코드 스텝 실측 = (E2E − T_pre(S_in)) / 255, 평균 문맥 l = S_in + 128."""
    d = load(k); pre, dec = [], []
    for S in (128, 256, 512, 1024, 2048, 4096):
        pre.append((f"prefill S={S}", t_pre_raw(S, k), d[f"prefill_in{S}"]))
    for B in (1, 2, 4, 8, 16, 32, 64):
        dec.append((f"decode B={B} S=128", t_dec_raw(B, 256, k), (d[f"decode_b{B}_in128"] - d["prefill_in128"]) / 255))
    for S in (512, 1024, 2048, 4096):
        dec.append((f"decode B=8 S={S}", t_dec_raw(8, S + 128, k), (d[f"decode_b8_in{S}"] - d[f"prefill_in{S}"]) / 255))
    return pre, dec


def fit(rows):
    x = np.array([r[1] for r in rows]); y = np.array([r[2] for r in rows])
    b1, b0 = np.polyfit(x, y, 1); pred = b1 * x + b0
    return b1, b0, float(np.mean(np.abs(pred - y) / y)), float(np.max(np.abs(pred - y) / y)), pred


def main():
    os.makedirs(OUT, exist_ok=True)
    beta, lines = {}, ["# 단일 쿼리 해석 모델 — 보정 결과 (Llama-3.1-8B bf16, vLLM 0.28.0)", "",
                       "모델: LLM-Pilot §IV-B (F/M/D) + §IV-C 선형 보정. 사양값 4090 165 TFLOPS·1008 GB/s, 3090 71 TFLOPS·936 GB/s.",
                       f"가중치 {BYTES_W/1e9:.2f} GB, KV {KV_TOK/1024:.0f} KB/token.", ""]
    for k in CARDS:
        pre, dec = points(k); beta[k] = {}
        for name, rows in (("prefill", pre), ("decode", dec)):
            b1, b0, mape, mx, pred = fit(rows); beta[k][name] = dict(b1=b1, b0=b0, mape=mape, max=mx)
            lines += [f"## {k} {name}: T_real = {b1:.3f}·T_pred + {b0*1e3:.1f} ms   (MAPE {mape:.1%}, max {mx:.1%}, n={len(rows)})", "",
                      "| 점 | 예측 raw ms | 보정 ms | 실측 ms | 오차 |", "|---|---|---|---|---|"]
            for (lab, x, y), p in zip(rows, pred):
                lines.append(f"| {lab} | {x*1e3:.1f} | {p*1e3:.1f} | {y*1e3:.1f} | {(p-y)/y:+.1%} |")
            lines.append("")
    # 단일 쿼리 E2E 예측 vs 실측 (프로파일의 B=1 셀: S_in=128, S_out=256)
    lines += ["## 단일 쿼리 E2E (S_in=128, S_out=256): 예측 = T_pre + Σ_t T_dec(1, 128+t)", "",
              "| 카드 | 예측 s | 실측 s | 오차 |", "|---|---|---|---|"]
    for k in CARDS:
        bp, bd = beta[k]["prefill"], beta[k]["decode"]
        e2e = bp["b1"] * t_pre_raw(128, k) + bp["b0"] + sum(bd["b1"] * t_dec_raw(1, 128 + t, k) + bd["b0"] for t in range(1, 256))
        meas = load(k)["decode_b1_in128"]
        lines.append(f"| {k} | {e2e:.3f} | {meas:.3f} | {(e2e-meas)/meas:+.1%} |")
    # 워크로드 2종 단일 쿼리 예측 (실측 없음 — 열린 도착 실험의 첫 검증 대상)
    lines += ["", "## 단일 쿼리 예측 (검증 대상): TTFT = T_pre, TPOT = 평균 T_dec, E2E", "",
              "| 카드 | 워크로드 | TTFT ms | TPOT ms | E2E s |", "|---|---|---|---|---|"]
    for k in CARDS:
        bp, bd = beta[k]["prefill"], beta[k]["decode"]
        for (si, so, nm) in ((2048, 128, "프리필 중심 2048/128"), (128, 1024, "디코드 중심 128/1024")):
            ttft = bp["b1"] * t_pre_raw(si, k) + bp["b0"]
            steps = [bd["b1"] * t_dec_raw(1, si + t, k) + bd["b0"] for t in range(1, so)]
            lines.append(f"| {k} | {nm} | {ttft*1e3:.0f} | {np.mean(steps)*1e3:.1f} | {ttft + sum(steps):.2f} |")
    json.dump(beta, open(os.path.join(OUT, "beta.json"), "w"), indent=1)
    open(os.path.join(OUT, "README.md"), "w").write("\n".join(lines) + "\n"); print("\n".join(lines))


if __name__ == "__main__":
    main()


# ---- 자원별 계수 (vLLM 보정 2안) -------------------------------------------------------------
# T_dec = β0 + βc·F/C + βw·M_w/BW + βkv·KV/BW   : 가중치 읽기와 페이지드 KV 읽기의 실효 대역폭이 다르다는 가정.
def fit_multi(k):
    from scipy.optimize import nnls
    d = load(k); c = CARDS[k]; X, y, lab = [], [], []
    def row(B, l): return [1.0, f_decode(B, l) / c["C"], BYTES_W / c["BW"], B * l * KV_TOK / c["BW"]]
    import re
    for key in sorted(d):                      # decode_b{B}_in{S}.json 전부 (추가 프로파일 점 포함)
        m = re.match(r"decode_b(\d+)_in(\d+)$", key)
        if not m or f"prefill_in{m.group(2)}" not in d:
            continue
        B, S = int(m.group(1)), int(m.group(2))
        X.append(row(B, S + 128)); y.append((d[key] - d[f"prefill_in{S}"]) / 255); lab.append(f"B={B} S={S}")
    X, y = np.array(X), np.array(y); beta, _ = nnls(X, y); pred = X @ beta
    err = (pred - y) / y
    return beta, lab, pred, y, float(np.mean(np.abs(err))), float(np.max(np.abs(err)))


def main_multi():
    lines = ["", "# 디코드 2안: 자원별 계수  T = β0 + βc·(F/C) + βw·(M_w/BW) + βkv·(KV/BW)   (NNLS, n=11)", ""]
    out = {}
    for k in CARDS:
        beta, lab, pred, y, mape, mx = fit_multi(k); out[k] = dict(b0=beta[0], bc=beta[1], bw=beta[2], bkv=beta[3], mape=mape, max=mx)
        lines += [f"## {k}: β0={beta[0]*1e3:.1f} ms, βc={beta[1]:.2f}, βw={beta[2]:.2f}, βkv={beta[3]:.2f}   (MAPE {mape:.1%}, max {mx:.1%})", "",
                  "| 점 | 예측 ms | 실측 ms | 오차 |", "|---|---|---|---|"]
        lines += [f"| {l} | {p*1e3:.1f} | {m*1e3:.1f} | {(p-m)/m:+.1%} |" for l, p, m in zip(lab, pred, y)]
        lines.append("")
    lines += ["해석: βw ≈ 1 이면 가중치 읽기는 HBM 피크로, βkv > 1 이면 KV 읽기(페이지드 어텐션)는 그보다 느린 실효 대역폭.",
              "βc 는 디코드 GEMM 의 연산 효율(작은 배치라 피크 대비 낮음). β0 는 스텝 고정 오버헤드(스케줄·샘플러·런치)."]
    lines += ["", "## 2안 계수로 단일 쿼리 예측 (프리필은 1안 β 사용)", "",
              "| 카드 | 워크로드 | TTFT ms | TPOT ms | E2E s | 실측 E2E (있으면) |", "|---|---|---|---|---|---|"]
    bp = json.load(open(os.path.join(OUT, "beta.json")))
    for k in CARDS:
        c = CARDS[k]; b = out[k]; d = load(k)
        def tdec(B, l): return b["b0"] + b["bc"] * f_decode(B, l) / c["C"] + b["bw"] * BYTES_W / c["BW"] + b["bkv"] * B * l * KV_TOK / c["BW"]
        for (si, so, nm, meas) in ((128, 256, "프로파일 B=1 128/256", d.get("decode_b1_in128")),
                                   (2048, 128, "프리필 중심 2048/128", None), (128, 1024, "디코드 중심 128/1024", None)):
            ttft = bp[k]["prefill"]["b1"] * t_pre_raw(si, k) + bp[k]["prefill"]["b0"]
            steps = [tdec(1, si + t) for t in range(1, so)]
            e2e = ttft + sum(steps)
            lines.append(f"| {k} | {nm} | {ttft*1e3:.0f} | {np.mean(steps)*1e3:.1f} | {e2e:.2f} | " + (f"{meas:.2f} ({(e2e-meas)/meas:+.1%})" if meas else "–") + " |")
    json.dump(out, open(os.path.join(OUT, "beta_decode_multi.json"), "w"), indent=1)
    open(os.path.join(OUT, "README.md"), "a").write("\n".join(lines) + "\n"); print("\n".join(lines))


if __name__ == "__main__":
    main_multi()
