#!/usr/bin/env python
"""
Time-axis experiment: does the optimal split actually move during a run, and is
a static choice good enough?

This is the test that decides whether a solver is needed at all.  The static
tables (§15.6-15.9) say "for this demand, use that split" -- an operator can
print them and pin them to the wall.  The argument for a solver only exists if
(a) the optimum moves while the service is running, and (b) reconfiguration
costs enough that "just switch every time" is also wrong.

Policies compared, all on the same demand trace:

  equal           a fixed human-plausible split (e.g. 2:1:1), never changes
  static-best     the single best fixed split for the whole trace (oracle:
                  it is chosen with full knowledge of the future)
  adaptive-naive  switch to this window's optimum every window, pay the cost
  adaptive-cost   switch only when the gain over the next H windows beats the
                  reconfiguration cost  <- what a solver would do

Reconfiguration: a GPU that changes role is unavailable for RECONF seconds
(measured: 33 s worker cold start, §14 U2).  During that time it belongs to
neither tenant, so the window is scored under the intersection of old and new.

Score: demand-weighted SLO attainment.  Each tenant contributes
rate * window * SAR(gpus, rate) successful requests; we report the fraction of
all arriving requests that met their SLO, plus the fraction of windows in which
every tenant cleared the floor.
"""
import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != "/" and not _os.path.isdir(_os.path.join(_d, "results")):
    _d = _os.path.dirname(_d)
_sys.path[:0] = [_os.path.join(_d, "solver"), _os.path.join(_d, "bench")]
_os.chdir(_d)

import argparse
import csv
import json
import math
import random

import cosolve as CS

G = 4
RECONF_S = 33.0          # measured worker cold start (§14 U2)
FLOOR = 0.90


# ---------------------------------------------------------------- tables
def dit_curve(scale, mix, rates):
    return {g: {r: CS.dit_sar(g, r, mix, scale) for r in rates} for g in range(G + 1)}


def load_curve(path, gpus=G):
    tab = {}
    for r in csv.DictReader(open(path)):
        tab.setdefault(int(r["gpus"]), {})[float(r["rate"])] = float(r["sar"])
    for g in range(gpus + 1):
        tab.setdefault(g, {})
        tab[g][0.0] = 1.0 if g >= 0 else 0.0
    return tab


def vlm_curve(enc_ms, root, gpus=G):
    raw = CS.load_vlm(enc_ms, root)
    tab = {}
    for (g, r), v in raw.items():
        tab.setdefault(g, {})[r] = v
    for g in range(gpus + 1):
        tab.setdefault(g, {})
        tab[g][0.0] = 1.0
    return tab


N_REQ = 300              # requests per simulated run in the pool sweeps (sar denominators confirm it)


def work_llm(path, gpus=G):
    """GPU-seconds of service per request, read off the saturated point of the table.

    At the highest offered rate the pool is the bottleneck, so makespan is the time it
    took to drain N_REQ requests.  gpus * makespan / N_REQ is then the GPU-seconds one
    request actually consumes -- the honest unit of "how much of the pool did you buy".
    """
    rows = list(csv.DictReader(open(path)))
    top = max((r for r in rows if int(r["gpus"]) == gpus), key=lambda r: float(r["rate"]))
    return gpus * float(top["makespan_s"]) / N_REQ


def work_dit(scale, mix):
    """Mean solo latency of the template mix on one worker of the scaled hardware."""
    import model as MD
    return scale * sum(w * MD.T_FULL[t] for t, w in mix.items())


def sar(tab, g, rate):
    """Linear interpolation in rate; 0 GPUs serves nothing; beyond the grid, hold the last value."""
    if rate <= 0:
        return 1.0
    if g <= 0:
        return 0.0
    row = tab.get(g)
    if not row:
        return 0.0
    xs = sorted(row)
    if rate <= xs[0]:
        return row[xs[0]]
    if rate >= xs[-1]:
        return row[xs[-1]]
    for a, b in zip(xs, xs[1:]):
        if a <= rate <= b:
            w = (rate - a) / (b - a)
            return row[a] * (1 - w) + row[b] * w
    return row[xs[-1]]


# ---------------------------------------------------------------- traces
def trace(shape, n, seed=11):
    """n windows of (dit, llm, vlm) demand.  Peaks are scaled so that each tenant
    alone can saturate the pool, which is what makes the split contested."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        p = i / n
        if shape == "staggered":       # each tenant peaks in a different third
            d = 0.045 * max(0.0, math.sin(math.pi * min(1.0, max(0.0, (p - 0.00) * 3))))
            l = 7.5 * max(0.0, math.sin(math.pi * min(1.0, max(0.0, (p - 0.33) * 3))))
            v = 2.0 * max(0.0, math.sin(math.pi * min(1.0, max(0.0, (p - 0.66) * 3))))
        elif shape == "correlated":    # all three rise and fall together
            s = max(0.0, math.sin(math.pi * p))
            d, l, v = 0.045 * s, 7.5 * s, 2.0 * s
        elif shape == "random":        # independent random walks
            if not out:
                d, l, v = 0.02, 3.0, 1.0
            else:
                pd, pl, pv = out[-1]
                d = min(0.05, max(0.0, pd + rng.uniform(-0.012, 0.012)))
                l = min(8.0, max(0.0, pl + rng.uniform(-2.0, 2.0)))
                v = min(2.0, max(0.0, pv + rng.uniform(-0.6, 0.6)))
        elif shape == "burst":         # quiet with two sharp spikes on different tenants
            d = 0.04 if 0.20 <= p < 0.30 else 0.008
            l = 7.0 if 0.55 <= p < 0.68 else 1.5
            v = 1.8 if 0.80 <= p < 0.88 else 0.4
        else:
            raise ValueError(shape)
        out.append((round(d, 4), round(l, 2), round(v, 2)))
    return out


# ---------------------------------------------------------------- scoring
def splits():
    return [(a, b, c) for a in range(G + 1) for b in range(G + 1 - a) for c in range(G + 1 - a - b)]


def window_score(cfg, demand, T, W, work=None):
    """(request-weighted successes, arrivals, floor?, worst SAR, mean SAR, work-ok, work-arr)

    Two aggregate weightings, because they disagree.  Counting requests makes the LLM
    tenant 99.4% of the traffic (7.5 req/s vs 0.045), so a policy that protects DiT looks
    terrible.  Counting GPU-seconds of demanded work -- what each tenant actually buys --
    puts the ratio at about 1:7 instead of 1:167.  We report both."""
    (gd, gl, gv), (rd, rl, rv) = cfg, demand
    sd, sl, sv = sar(T["dit"], gd, rd), sar(T["llm"], gl, rl), sar(T["vlm"], gv, rv)
    arr = (rd + rl + rv) * W
    ok = sum(x * W * s for x, s in ((rd, sd), (rl, sl), (rv, sv)))
    act = [(sd, rd), (sl, rl), (sv, rv)]
    floor_ok = all(x >= FLOOR for x, r in act if r > 0) or arr == 0
    live = [x for x, r in act if r > 0]
    wk = work or {"dit": 1.0, "llm": 1.0, "vlm": 1.0}
    warr = sum(r * W * c for r, c in ((rd, wk["dit"]), (rl, wk["llm"]), (rv, wk["vlm"])))
    wok = sum(r * W * c * s for r, c, s in
              ((rd, wk["dit"], sd), (rl, wk["llm"], sl), (rv, wk["vlm"], sv)))
    return ok, arr, floor_ok, (min(live) if live else 1.0), (sum(live) / len(live) if live else 1.0), wok, warr


def value(cfg, demand, T, W, work, objective):
    """The scalar the policy is trying to maximise in one window.

    This is the knob that decides the whole experiment.  `floor` is a per-tenant SLA:
    the weakest tenant's attainment.  `work` is an aggregate: the fraction of demanded
    GPU-seconds that met their SLO.  They disagree -- the aggregate is maximised by
    abandoning the expensive tenant -- so we run the comparison under both."""
    ok, arr, fl, wv, mv, wok, warr = window_score(cfg, demand, T, W, work)
    return wv if objective == "floor" else (wok / warr if warr else 1.0)


def best_split(demand, T, W, work=None, objective="floor"):
    best = None
    for cfg in splits():
        ok, arr, fl, _, _, wok, warr = window_score(cfg, demand, T, W, work)
        used = sum(cfg)
        key = ((0 if fl else 1, -ok, used) if objective == "floor"
               else (-(wok / warr if warr else 1.0), used))
        if best is None or key < best[0]:
            best = (key, cfg)
    return best[1]


def run_policy(policy, tr, T, W, horizon=3, work=None, objective="floor"):
    cur = None
    streak = [None, 0]
    tot_ok = tot_arr = tot_wok = tot_warr = 0.0
    win_ok = 0; worst = []; mean = []
    switches = 0
    for i, dem in enumerate(tr):
        if policy[0] == "fixed":
            nxt = policy[1]
        elif policy[0] == "naive":
            nxt = best_split(dem, T, W, work, objective)
        elif policy[0] == "hyst":
            # no model: switch only after the optimum has been the same for k windows
            cand = best_split(dem, T, W, work, objective)
            k = policy[1]
            streak[1] = streak[1] + 1 if cand == streak[0] else 1
            streak[0] = cand
            nxt = cand if (cur is None or streak[1] >= k) else cur
        elif policy[0] in ("cost", "cost-blind"):
            cand = best_split(dem, T, W, work, objective)
            if cur is None or cand == cur:
                nxt = cand
            else:
                # gain over the next `horizon` windows if we switch now, minus the
                # cold-start loss this window.  The solver only pays when it pays off.
                # "cost"       sees the real future (oracle forecast)
                # "cost-blind" assumes the future looks like now (persistence) -- what a
                #              router without a demand predictor would actually have
                fut = tr[i:i + horizon] if policy[0] == "cost" else [dem] * horizon
                g_new = sum(value(cand, d, T, W, work, objective) for d in fut)
                g_old = sum(value(cur, d, T, W, work, objective) for d in fut)
                inter = tuple(min(a, b) for a, b in zip(cur, cand))
                loss = (value(cand, dem, T, W, work, objective)
                        - value(inter, dem, T, W, work, objective)) * (RECONF_S / W)
                nxt = cand if (g_new - g_old) > loss else cur
        else:
            raise ValueError(policy)

        if cur is not None and nxt != cur:
            switches += 1
            inter = tuple(min(a, b) for a, b in zip(cur, nxt))
            f = min(1.0, RECONF_S / W)
            a1, n1, _, w1, m1, k1, q1 = window_score(inter, dem, T, W, work)
            a2, n2, fl, w2, m2, k2, q2 = window_score(nxt, dem, T, W, work)
            ok, arr = a1 * f + a2 * (1 - f), n2
            wok, warr = k1 * f + k2 * (1 - f), q2
            wv, mv = w1 * f + w2 * (1 - f), m1 * f + m2 * (1 - f)
            floor_ok = fl and f < 0.5
        else:
            ok, arr, floor_ok, wv, mv, wok, warr = window_score(nxt, dem, T, W, work)
        cur = nxt
        tot_ok += ok; tot_arr += arr; win_ok += floor_ok; worst.append(wv); mean.append(mv)
        tot_wok += wok; tot_warr += warr
    return dict(attain=tot_ok / tot_arr if tot_arr else 1.0,
                work_attain=tot_wok / tot_warr if tot_warr else 1.0,
                worst=sum(worst) / len(worst), mean=sum(mean) / len(mean),
                floor_windows=win_ok / len(tr), switches=switches)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=60)
    ap.add_argument("--window-s", type=float, default=300.0, help="window length in seconds")
    ap.add_argument("--dit-scale", type=float, default=10.0)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--shapes", default="staggered,correlated,random,burst")
    ap.add_argument("--objective", default="floor", choices=["floor", "work"],
                    help="what every policy maximises: per-tenant SLA floor, or aggregate work-weighted attainment")
    ap.add_argument("-o", "--out", default="results/servingsim/dynamic/summary.md")
    a = ap.parse_args()
    W = a.window_s

    rates = [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06]
    T = {"dit": dit_curve(a.dit_scale, CS.MIXES["short"], rates),
         "llm": load_curve("results/servingsim/pool_park/summary.csv"),
         "vlm": vlm_curve(25.0, "results/servingsim/pool_vlm_park")}

    # GPU-seconds of service per request, per tenant.  Counting requests is the wrong
    # unit here: one DiT request is worth ~24 LLM requests of pool time.
    work = {"dit": work_dit(a.dit_scale, CS.MIXES["short"]),
            "llm": work_llm("results/servingsim/pool_park/summary.csv"),
            "vlm": work_llm("results/servingsim/pool_vlm_park/summary.csv")}

    lines = [f"# Time-axis: does the optimum move?  ({G} GPUs, window {W:.0f}s x {a.windows}, "
             f"reconfiguration {RECONF_S:.0f}s, floor {FLOOR}, objective={a.objective})", "",
             f"per-request work (GPU-s): DiT {work['dit']:.2f} · LLM {work['llm']:.2f} · "
             f"VLM {work['vlm']:.2f}  (ratio {work['dit']/work['llm']:.0f} : 1 : "
             f"{work['vlm']/work['llm']:.1f})", "",
             "objective=floor: every policy maximises the weakest tenant's SAR (per-tenant SLA). "
             "objective=work: every policy maximises work-weighted attainment (aggregate). "
             "The point of running both is that the winner is not supposed to depend on it -- "
             "cost-awareness should help under whichever objective you declare.", ""]
    rows = []
    for shape in a.shapes.split(","):
        tr = trace(shape, a.windows)
        # how often does the per-window optimum change at all?
        opts = [best_split(d, T, W, work, a.objective) for d in tr]
        moves = sum(1 for x, y in zip(opts, opts[1:]) if x != y)
        distinct = len(set(opts))

        # oracle static: best single fixed split for this trace
        # chosen with the same lexicographic objective the adaptive policy uses,
        # and with full knowledge of the whole trace -- an oracle static baseline
        def _stat_key(c):
            sc = [window_score(c, d, T, W, work) for d in tr]
            if a.objective == "work":
                return (-sum(x[5] for x in sc), sum(c))
            return (-sum(x[2] for x in sc), -sum(x[3] for x in sc), sum(c))
        stat = min(splits(), key=_stat_key)
        stat_a = min(splits(), key=lambda c: -sum(window_score(c, d, T, W, work)[0] for d in tr))
        stat_w = min(splits(), key=lambda c: -sum(window_score(c, d, T, W, work)[5] for d in tr))
        pol = {"equal 2:1:1": ("fixed", (2, 1, 1)),
               f"static-best(하한) {stat[0]}:{stat[1]}:{stat[2]}": ("fixed", stat),
               f"static-best(요청가중) {stat_a[0]}:{stat_a[1]}:{stat_a[2]}": ("fixed", stat_a),
               f"static-best(작업가중) {stat_w[0]}:{stat_w[1]}:{stat_w[2]}": ("fixed", stat_w),
               "hysteresis k=2": ("hyst", 2),
               "hysteresis k=3": ("hyst", 3),
               "adaptive-naive": ("naive", None),
               "adaptive-cost(예측 완벽)": ("cost", None),
               "adaptive-cost(지속 가정)": ("cost-blind", None)}
        lines += [f"## {shape}", "",
                  f"per-window optimum changes **{moves}/{len(tr)-1}** times, **{distinct}** distinct splits used", "",
                  "| policy | 최약 테넌트 SAR | 테넌트 평균 SAR | 하한 통과 창 | 요청가중 만족 | 작업가중 만족 | 전환 |",
                  "|---|---|---|---|---|---|---|"]
        res = {}
        for name, p in pol.items():
            r = run_policy(p, tr, T, W, a.horizon, work, a.objective)
            res[name] = r
            lines.append(f"| {name} | {r['worst']:.3f} | {r['mean']:.3f} | {r['floor_windows']:.2f} | "
                         f"{r['attain']:.3f} | {r['work_attain']:.3f} | {r['switches']} |")
        lines.append("")
        rows.append((shape, moves, distinct, res))

    lines += ["## 한 줄", ""]
    for shape, moves, distinct, res in rows:
        ad = res["adaptive-cost(지속 가정)"]
        st = max((v for k, v in res.items() if k.startswith(("equal", "static"))), key=lambda v: v["worst"])
        lines.append(f"- **{shape}**: 최적 {moves}회 이동 · 최약 테넌트 SAR "
                     f"{st['worst']:.3f} → {ad['worst']:.3f} ({(ad['worst']-st['worst'])*100:+.1f}%p), "
                     f"하한 통과 창 {st['floor_windows']:.2f} → {ad['floor_windows']:.2f} · "
                     f"작업가중 {st['work_attain']:.3f} → {ad['work_attain']:.3f}")
    txt = "\n".join(lines)
    _os.makedirs(_os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
