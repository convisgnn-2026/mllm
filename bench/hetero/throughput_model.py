#!/usr/bin/env python3
"""다중 요청 처리량 모델 (high-level). 스텝 모델(model_llm) 위에 두 가지만 얹는다:
  ① 동시 실행 상한: KV 용량(토큰) — 요청은 FCFS 로, 도달 길이(S_in+S_out) 블록이 들어갈 때 admit
  ② 혼합 스텝: 스텝마다 디코드 B 개 + 프리필 청크 P 토큰(예산 CHUNK).  T_step = T_dec(B, K) + β_mix · β1 · F_pre(P)/C
스텝 단위 결정론 시뮬(도착 없음 = 닫힌 투입). 복제 구성(A/B) 18+6칸으로 β_mix 를 맞추고 MAPE 를 본다.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_llm as M

ROOT = M.ROOT; AB = os.path.join(ROOT, "ab"); OUT = os.path.join(ROOT, "model")
BETA = json.load(open(os.path.join(OUT, "beta.json")))
BETA_D = json.load(open(os.path.join(OUT, "beta_decode_multi_v2.json")))
KV_CAP = {"4090": 30608, "3090": 31072}          # 복제 구성 vLLM 로그 "GPU KV cache size" (util 0.85)
KV_CAP_PD = {"4090": 37088, "3090": 37552}       # PD 구성 로그값 (커넥터 구성 시 KV 가 조금 더 큼)
CHUNK = 8192                                      # max_num_batched_tokens (스텝당 프리필 토큰 예산)
MAX_SEQS = 256
CLASSES = {"prefill": (2048, 128), "decode": (128, 1024)}
MIXES = {"prefill": 1.0, "p75": 0.75, "mixed": 0.5, "p25": 0.25, "decode": 0.0}
NS = [50, 100, 200]


def t_pre_marginal(P, k):                          # 혼합 스텝에 얹히는 프리필 청크의 연산 시간 (고정비 제외)
    return BETA[k]["prefill"]["b1"] * M.f_prefill(P) / M.CARDS[k]["C"]


def t_dec(B, K, k):
    b = BETA_D[k]; return b["beta0_ms"] / 1e3 + b["bkv"] * K * M.KV_TOK / M.CARDS[k]["BW"]


def simulate_card(reqs, k, beta_mix, chunk=None):
    chunk = CHUNK if chunk is None else chunk
    """reqs: [(S_in, S_out)] FCFS. 반환 makespan, TTFT 목록, TPOT 목록."""
    cap = KV_CAP[k]; waiting = [(i, 0) for i in range(len(reqs))]; running = []   # waiting: (idx, g_saved); running: [idx, pf, g, g_saved]
    used = 0; t = 0.0; ttft = [None] * len(reqs); tpot = [[] for _ in reqs]; first = [None] * len(reqs); preempted = 0
    while waiting or running:
        # vLLM V1 식: 현재 문맥(프리필 완료분 + 생성분)만큼만 블록 할당. admit 은 프롬프트가 들어가면 함.
        tgt = lambda r: reqs[r[0]][0] + r[3]                                   # 프리필(재계산 포함) 목표 길이
        used = sum(min(r[1], tgt(r)) + (r[2] - r[3] if r[1] >= tgt(r) else 0) for r in running)
        while waiting and len(running) < MAX_SEQS and used + reqs[waiting[0][0]][0] + waiting[0][1] <= cap:
            i, gs = waiting.pop(0); used += reqs[i][0] + gs; running.append([i, 0, gs, gs])
        # 디코드가 자라 용량을 넘으면 가장 최근에 admit 된 요청을 선점 (recompute: 생성분은 기억, 블록 반납)
        while used + sum(1 for r in running if r[1] >= tgt(r)) > cap and len(running) > 1:
            v = running.pop(); used -= min(v[1], tgt(v)) + (v[2] - v[3] if v[1] >= tgt(v) else 0)
            waiting.insert(0, (v[0], v[2])); preempted += 1
        # schedule step: decodes first, then prefill chunks within budget
        B, K, P = 0, 0, 0; budget = chunk
        for r in running:
            i, pf, g, gs = r
            if pf >= reqs[i][0] + gs:              # decode phase
                B += 1; K += reqs[i][0] + g
        for r in running:
            i, pf, g, gs = r
            if pf < reqs[i][0] + gs and budget > 0:
                take = min(reqs[i][0] + gs - pf, budget); r[1] += take; budget -= take; P += take
        dt = (t_dec(B, K, k) if B > 0 else BETA[k]["prefill"]["b0"]) + (beta_mix * t_pre_marginal(P, k) if P > 0 else 0.0)
        t += dt
        done = []
        for r in running:
            i, pf, g, gs = r
            if pf >= reqs[i][0] + gs:
                if first[i] is None:
                    ttft[i] = t; first[i] = t                 # 프리필 완료 = 첫 토큰
                else:
                    tpot[i].append(dt)
                r[2] += 1
                if r[2] >= reqs[i][1]:
                    done.append(r)
        for r in done:
            running.remove(r)
    simulate_card.preempted = preempted
    return t, [x for x in ttft if x], [np.mean(x) for x in tpot if x]


def reqs_for(mix, n, seed=0):
    np_ = round(n * MIXES[mix]); rs = [CLASSES["prefill"]] * np_ + [CLASSES["decode"]] * (n - np_)
    rng = np.random.default_rng(seed); rng.shuffle(rs); return [tuple(r) for r in rs]


def split(rs, w4, w3):
    """가중 RR: 요청 순서대로 w4:w3 비율로 카드에 배정."""
    a, b, seq = [], [], (["4090"] * w4 + ["3090"] * w3)
    for j, r in enumerate(rs):
        (a if seq[j % len(seq)] == "4090" else b).append(r)
    return a, b


def predict_rep(mix, n, w4, w3, beta_mix):
    r4, r3 = split(reqs_for(mix, n), w4, w3)
    t4, tt4, tp4 = simulate_card(r4, "4090", beta_mix); t3, tt3, tp3 = simulate_card(r3, "3090", beta_mix)
    return dict(makespan=max(t4, t3), req_per_s=n / max(t4, t3), tpot_ms=1e3 * np.mean(tp4 + tp3), ttft_ms=1e3 * np.median(tt4 + tt3), t4=t4, t3=t3)


def measured(cfg, mix, n):
    f = os.path.join(AB, cfg, f"{mix}_n{n}", "summary.json"); return json.load(open(f)) if os.path.exists(f) else None


CFG = {"A": (1, 1), "B": (2, 1)}


def evaluate(beta_mix, verbose=False):
    errs, rows = [], []
    for cfg, (w4, w3) in CFG.items():
        for mix in MIXES:
            for n in NS:
                m = measured(cfg, mix, n)
                if not m:
                    continue
                p = predict_rep(mix, n, w4, w3, beta_mix); e = (p["req_per_s"] - m["req_per_s"]) / m["req_per_s"]; errs.append(e)
                rows.append(f"| {cfg} | {mix} | {n} | {p['req_per_s']:.2f} | {m['req_per_s']:.2f} | {e:+.0%} | {p['tpot_ms']:.0f} | {m['tpot_ms']['p50']:.0f} | {p['t4']:.0f}/{p['t3']:.0f} |")
    return np.mean(np.abs(errs)), np.mean(errs), rows


def main():
    global CHUNK
    lines = ["# 다중 요청 처리량 모델 (닫힌 투입, 복제 구성 A/B) — β_mix 적합", "",
             f"KV 용량 4090 {KV_CAP['4090']} / 3090 {KV_CAP['3090']} 토큰, 스텝 프리필 예산 {CHUNK}, FCFS admit(도달 길이 예약).", ""]
    res = []
    for ch in (2048, 4096, 8192):
        CHUNK = ch
        for bm in (0.0, 0.5, 1.0, 1.5, 2.0):
            mape, bias, _ = evaluate(bm); res.append((ch, bm, mape, bias))
    lines += ["| chunk | β_mix | MAPE | 편향 |", "|---|---|---|---|"] + [f"| {ch} | {bm:.1f} | {mape:.1%} | {bias:+.1%} |" for ch, bm, mape, bias in res]
    ch, best, _, _ = min(res, key=lambda r: r[2]); CHUNK = ch
    mape, bias, rows = evaluate(best, True)
    lines += ["", f"## 최적 chunk = {ch}, β_mix = {best:.1f}: MAPE {mape:.1%}, 편향 {bias:+.1%}", "",
              "| 구성 | mix | n | 예측 req/s | 실측 | 오차 | 예측 TPOT | 실측 TPOT p50 | 카드별 makespan 4090/3090 s |", "|---|---|---|---|---|---|---|---|---|"] + rows
    open(os.path.join(OUT, "throughput_model.md"), "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
    json.dump({"beta_mix": best, "kv_cap": KV_CAP, "chunk": ch}, open(os.path.join(OUT, "throughput_params.json"), "w"), indent=1)


if __name__ == "__main__":
    main()


# ---- PD 구성 (C: 4090P→3090D, D: 반대) ------------------------------------------------------
LINK = dict(a=0.005, bw=2.4e9)      # 4090↔3090 NIXL/UCX 튜닝 실측: 고정 5 ms + 2.4 GB/s, 전송 직렬


def simulate_card_arrivals(reqs, k, beta_mix, arrivals, prefilled, chunk=None):
    """simulate_card 와 같되, 요청마다 도착 시각이 있고(prefilled=True 면 프리필 완료 상태로 도착) 선점 시 로컬 재프리필."""
    chunk = CHUNK if chunk is None else chunk
    cap = KV_CAP_PD[k]; order = sorted(range(len(reqs)), key=lambda i: arrivals[i])
    waiting = [(i, 0, True) for i in order]      # (idx, g_saved, fresh)  fresh=True 이면 프리필 완료 상태로 admit
    running = []; t = 0.0; ttft = [None] * len(reqs); tpot = [[] for _ in reqs]; first = [None] * len(reqs); preempted = 0
    while waiting or running:
        if not running and waiting and arrivals[waiting[0][0]] > t:
            t = arrivals[waiting[0][0]]                                  # 놀다가 도착 기다림
        tgt = lambda r: reqs[r[0]][0] + r[3]
        used = sum(min(r[1], tgt(r)) + (r[2] - r[3] if r[1] >= tgt(r) else 0) for r in running)
        while waiting and arrivals[waiting[0][0]] <= t and len(running) < MAX_SEQS and used + reqs[waiting[0][0]][0] + waiting[0][1] <= cap:
            i, gs, fresh = waiting.pop(0); used += reqs[i][0] + gs
            running.append([i, reqs[i][0] if (fresh and prefilled) else 0, gs, gs])
        while used + sum(1 for r in running if r[1] >= tgt(r)) > cap and len(running) > 1:
            v = running.pop(); used -= min(v[1], tgt(v)) + (v[2] - v[3] if v[1] >= tgt(v) else 0)
            waiting.insert(0, (v[0], v[2], False)); preempted += 1
        B, K, P = 0, 0, 0; budget = chunk
        for r in running:
            if r[1] >= tgt(r):
                B += 1; K += reqs[r[0]][0] + r[2]
        for r in running:
            if r[1] < tgt(r) and budget > 0:
                take = min(tgt(r) - r[1], budget); r[1] += take; budget -= take; P += take
        if B == 0 and P == 0:                                            # 아무것도 못 함 → 다음 도착까지
            t = arrivals[waiting[0][0]] if waiting else t; continue
        dt = (t_dec(B, K, k) if B > 0 else BETA[k]["prefill"]["b0"]) + (beta_mix * t_pre_marginal(P, k) if P > 0 else 0.0)
        t += dt; done = []
        for r in running:
            i = r[0]
            if r[1] >= tgt(r):
                if first[i] is None:
                    ttft[i] = t - arrivals[i]; first[i] = t
                else:
                    tpot[i].append(dt)
                r[2] += 1
                if r[2] >= reqs[i][1]:
                    done.append(r)
        for r in done:
            running.remove(r)
    simulate_card_arrivals.preempted = preempted
    return t, [x for x in ttft if x is not None], [np.mean(x) for x in tpot if x]


def simulate_prefill_card(reqs, k, chunk=None):
    """프리필 전용 카드: 스텝마다 예산 chunk 토큰의 프리필. 완료 시각 목록 반환 (KV 보유 한도는 링크 직렬화가 지배하므로 무시)."""
    chunk = CHUNK if chunk is None else chunk
    b = BETA[k]["prefill"]; t = 0.0; done_t = [None] * len(reqs); pf = [0] * len(reqs); i0 = 0
    while i0 < len(reqs):
        budget = chunk; P = 0; j = i0
        while j < len(reqs) and budget > 0:
            take = min(reqs[j][0] - pf[j], budget); pf[j] += take; budget -= take; P += take
            if pf[j] >= reqs[j][0]:
                j += 1
            else:
                break
        t += b["b0"] + b["b1"] * M.f_prefill(P) / M.CARDS[k]["C"]
        for q in range(i0, j):
            done_t[q] = t
        i0 = j
    return done_t


def predict_pd(mix, n, P, D, beta_mix):
    rs = reqs_for(mix, n)
    done = simulate_prefill_card(rs, P)
    link_free = 0.0; arr = []
    for i, r in enumerate(rs):                                           # 링크 직렬: 프리필 완료 순으로 전송
        s = max(done[i], link_free); link_free = s + LINK["a"] + r[0] * M.KV_TOK / LINK["bw"]; arr.append(link_free)
    mk, tt, tp = simulate_card_arrivals(rs, D, beta_mix, arr, prefilled=True)
    ttft = [d + a - 0 for d, a in zip(done, arr)]                         # 첫 토큰 = 프리필 완료(+전송) 시각
    return dict(makespan=mk, req_per_s=n / mk, tpot_ms=1e3 * np.mean(tp), ttft_ms=1e3 * np.median(arr),
                tP=done[-1], tX=link_free, preempted=simulate_card_arrivals.preempted)


CFG_PD = {"C": ("4090", "3090"), "D": ("3090", "4090")}


def evaluate_all(beta_mix):
    rows, errs = [], {c: [] for c in list(CFG) + list(CFG_PD)}; agree = tot = 0
    for mix in MIXES:
        for n in NS:
            pred, meas = {}, {}
            for c, (w4, w3) in CFG.items():
                pred[c] = predict_rep(mix, n, w4, w3, beta_mix)
            for c, (P, D) in CFG_PD.items():
                pred[c] = predict_pd_calibrated(mix, n, P, D, beta_mix)
            for c in pred:
                m = measured(c, mix, n)
                if m:
                    meas[c] = m["req_per_s"]; errs[c].append((pred[c]["req_per_s"] - m["req_per_s"]) / m["req_per_s"])
            if meas:
                bp = max(meas, key=lambda c: pred[c]["req_per_s"]); bm = max(meas, key=meas.get); tot += 1; agree += bp == bm
            cells = " | ".join(f"{pred[c]['req_per_s']:.2f}/{meas[c]:.2f}" if c in meas else f"{pred[c]['req_per_s']:.2f}/–" for c in pred)
            rows.append(f"| {mix} | {n} | {cells} | {bp if meas else '–'} | {bm if meas else '–'} |")
    return rows, errs, agree, tot


def main_pd():
    params = json.load(open(os.path.join(OUT, "throughput_params.json"))); bm = params["beta_mix"]
    global CHUNK; CHUNK = params["chunk"]
    rows, errs, agree, tot = evaluate_all(bm)
    lines = ["", "# 전 구성 예측 대 실측 (req/s, 예측/실측) — 모델 v1 + PD + 붕괴 보정(overflow×feed)", "",
             "| mix | n | A | B | C | D | 예측 승자 | 실측 승자 |", "|---|---|---|---|---|---|---|---|"] + rows
    lines += ["", "구성별 MAPE / 편향: " + ", ".join(f"{c} {np.mean(np.abs(e)):.0%} / {np.mean(e):+.0%}" for c, e in errs.items() if e),
              f"승자 일치 {agree}/{tot}", ""]
    open(os.path.join(OUT, "throughput_model.md"), "a").write("\n".join(lines) + "\n"); print("\n".join(lines))


if __name__ == "__main__":
    main_pd()


# ---- PD 붕괴 보정 (LLM-Pilot §IV-C 식 회귀, 변수 = KV overflow × 공급비) --------------------------------
# 디코더 용량을 넘는 디코드 KV 요구(overflow)와 프리필러가 디코더보다 빨리 공급하는 정도(feed)가 클수록
# 선점 순환으로 처리량이 무너진다. 28칸 적합: ratio = min(1, (ov·min(feed,3)/1.58)^-0.71)  → PD MAPE 109% → 15%.
THRASH = dict(x0=1.58, k=0.71, feed_cap=3.0)


def pd_overflow(mix, n, D):
    rs = reqs_for(mix, n); dec = [r for r in rs if r == CLASSES["decode"]]
    return sum(r[0] + r[1] for r in dec) / KV_CAP_PD[D]


def predict_pd_calibrated(mix, n, P, D, beta_mix):
    p = predict_pd(mix, n, P, D, beta_mix)
    ov = pd_overflow(mix, n, D); feed = min(p["makespan"] / max(p["tP"], 1e-9), THRASH["feed_cap"])
    x = ov * feed; ratio = min(1.0, (x / THRASH["x0"]) ** (-THRASH["k"])) if x > 0 else 1.0
    q = dict(p); q["req_per_s"] = p["req_per_s"] * ratio; q["makespan"] = n / q["req_per_s"]; q["thrash_ratio"] = ratio; q["overflow"] = ov
    return q
