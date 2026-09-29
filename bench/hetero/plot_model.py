#!/usr/bin/env python3
"""그림 1: 단일 쿼리 모델 보정 — 예측 대 실측 (프리필·디코드 × 4090·3090), 1안(LLM-Pilot 단일 β) 대 2안(자원별 β).
용량 초과 점(B·(S+256) > KV 용량)은 회색 ×로 표시하고 보정에서 제외. 출력 results/hetero/model/fig1_model_calibration.png"""
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import nnls

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_llm as M

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
C1, C2, CX = "#2a78d6", "#eb6834", "#9a9a94"
UTIL = 0.90   # 프로파일 실행 설정


def cap_tokens(k):
    return (24 * 1024 ** 3 * UTIL - M.BYTES_W) / M.KV_TOK


def decode_points(k):
    d = M.load(k); pts = []
    for key in sorted(d):
        m = re.match(r"decode_b(\d+)_in(\d+)$", key)
        if not m or f"prefill_in{m.group(2)}" not in d:
            continue
        B, S = int(m.group(1)), int(m.group(2))
        meas = (d[key] - d[f"prefill_in{S}"]) / 255
        over = B * (S + 256) > cap_tokens(k)
        pts.append(dict(B=B, S=S, l=S + 128, meas=meas, over=over))
    return pts


def fit_decode(k, pts):
    c = M.CARDS[k]; use = [p for p in pts if not p["over"]]
    X = np.array([[1.0, M.f_decode(p["B"], p["l"]) / c["C"], M.BYTES_W / c["BW"], p["B"] * p["l"] * M.KV_TOK / c["BW"]] for p in use])
    y = np.array([p["meas"] for p in use]); beta, _ = nnls(X, y)
    raw = np.array([M.t_dec_raw(p["B"], p["l"], k) for p in use]); b1, b0 = np.polyfit(raw, y, 1)
    def pred1(p): return b1 * M.t_dec_raw(p["B"], p["l"], k) + b0
    def pred2(p): return float(np.array([1.0, M.f_decode(p["B"], p["l"]) / c["C"], M.BYTES_W / c["BW"], p["B"] * p["l"] * M.KV_TOK / c["BW"]]) @ beta)
    mape = lambda f: np.mean([abs(f(p) - p["meas"]) / p["meas"] for p in use])
    return pred1, pred2, mape(pred1), mape(pred2), beta, len(use)


def fit_prefill(k):
    d = M.load(k); S = [128, 256, 512, 1024, 2048, 4096]
    raw = np.array([M.t_pre_raw(s, k) for s in S]); y = np.array([d[f"prefill_in{s}"] for s in S]); b1, b0 = np.polyfit(raw, y, 1)
    pred = b1 * raw + b0; return S, y, pred, np.mean(np.abs(pred - y) / y)


def style(ax, title):
    ax.set_facecolor(SURF); ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=8)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8.5); ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.6)); fig.patch.set_facecolor(SURF); out = {}
    for row, k in enumerate(("4090", "3090")):
        S, y, pred, mp = fit_prefill(k); ax = axes[row][0]; style(ax, f"{k} prefill  —  β1·max(F/C, W/BW) + β0  (MAPE {mp:.1%})")
        ax.plot(S, y * 1e3, "o", color=INK, ms=6, label="measured"); ax.plot(S, pred * 1e3, "-", color=C1, lw=2, label="LLM-Pilot single β")
        ax.set_xscale("log", base=2); ax.set_xlabel("prompt tokens", color=INK2, fontsize=9); ax.set_ylabel("ms", color=INK2, fontsize=9)
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")
        pts = decode_points(k); p1, p2, m1, m2, beta, n = fit_decode(k, pts); out[k] = dict(beta0_ms=beta[0] * 1e3, bc=beta[1], bw=beta[2], bkv=beta[3], mape1=m1, mape2=m2, n=n)
        ax = axes[row][1]; style(ax, f"{k} decode step  —  MAPE single β {m1:.1%} / per-resource β {m2:.1%}  (n={n})")
        use = [p for p in pts if not p["over"]]; over = [p for p in pts if p["over"]]
        xs = [p["meas"] * 1e3 for p in use]
        ax.plot(xs, [p1(p) * 1e3 for p in use], "o", color=C1, ms=7, mec=SURF, label="LLM-Pilot single β (F/C + M/BW)")
        ax.plot(xs, [p2(p) * 1e3 for p in use], "s", color=C2, ms=7, mec=SURF, label="per-resource β (β0 + βc·F/C + βw·W/BW + βkv·KV/BW)")
        if over:
            ax.plot([p["meas"] * 1e3 for p in over], [p2(p) * 1e3 for p in over], "x", color=CX, ms=8, mew=2, label="KV over capacity (excluded)")
        lim = max(xs + [p2(p) * 1e3 for p in use]) * 1.08; ax.plot([0, lim], [0, lim], "--", color=INK2, lw=1)
        for p in use:
            if p["S"] >= 2048 or p["B"] >= 32:
                ax.annotate(f"B{p['B']}·S{p['S']}", (p["meas"] * 1e3, p2(p) * 1e3), textcoords="offset points", xytext=(6, -10), fontsize=7.5, color=INK2)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_xlabel("measured step (ms)", color=INK2, fontsize=9); ax.set_ylabel("predicted step (ms)", color=INK2, fontsize=9)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Single-query step model calibration on vLLM 0.28 (Llama-3.1-8B bf16): prefill fits LLM-Pilot form; decode needs a separate KV-read coefficient",
                 x=0.01, ha="left", fontsize=10, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); p = os.path.join(M.OUT, "fig1_model_calibration.png"); fig.savefig(p, dpi=160); plt.close(fig)
    json.dump(out, open(os.path.join(M.OUT, "beta_decode_multi_v2.json"), "w"), indent=1); print(p); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
