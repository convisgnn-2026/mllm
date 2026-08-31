#!/usr/bin/env python
"""
Provisioning solver:  min N  s.t.  Capacity(sigma) >= lambda,  every template servable.

The cost is the number of physical GPUs held, so slicing one of them is free in
cost terms -- the question is only how to carve what you are paying for anyway.
That is what makes a partition worth solving for: two configurations of the same
N can differ by 11% in sustainable rate, which is sometimes exactly the
difference between needing N and needing N+1.

GPUs are interchangeable, so a configuration is a multiset of per-GPU layouts.
The catalogue is small (9 layouts) and the feasibility filter is cheap, so the
search is exhaustive -- the same shape as LLM-Pilot's filter-then-enumerate.

  python solver.py --rate 1.1 --mix short
  python solver.py --sweep --mix short
"""

import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != "/" and not _os.path.isdir(_os.path.join(_d, "results")):
    _d = _os.path.dirname(_d)
_sys.path[:0] = [_os.path.join(_d, "solver"), _os.path.join(_d, "bench")]
_os.chdir(_d)            # results/ and slo_configs/ are addressed from the repo root

import argparse
from itertools import combinations_with_replacement

import json

import model as M
import sim as SIM

# REALISED mixes of the workloads we actually served -- not the nominal
# probabilities.  With n=150 the sample deviates: the "short" trace drew zero
# 1280x720x49 despite a nominal 2.2%, and because that template is 14x the
# cost of the cheapest one, using the nominal mix instead understates capacity
# by 16%.  Capacity is that sensitive to the tail of the mix, which is why the
# solver has to be fed a measured mix rather than a guessed one.
MIXES = {
    "short": {"480x272x33": 0.547, "832x480x33": 0.313,
              "832x480x49": 0.140, "1280x720x49": 0.0},
    "even": {"480x272x33": 0.247, "832x480x33": 0.293,
             "832x480x49": 0.240, "1280x720x49": 0.220},
    "long": {"480x272x33": 0.173, "832x480x33": 0.180,
             "832x480x49": 0.280, "1280x720x49": 0.367},
}

# Layouts we have actually served traffic on.  The capacity model is a
# work-conserving approximation: it assumes every worker sees the same template
# mix.  With a heterogeneous pool and size-matched dispatch that is false, and
# the one heterogeneous configuration we measured (2 Full + one 7-way GPU) came
# in 11% BELOW prediction.  So an answer outside this set is an extrapolation
# and should be measured before it is believed.
VALIDATED = {("7g.269gb",), ("3g.135gb",) * 2, ("1g.34gb",) * 7}


def is_validated(cfg):
    return all(L.profiles in VALIDATED for L in cfg)


def name(cfg):
    """Readable label for a multiset of layouts."""
    parts = []
    for L in sorted(set(cfg), key=lambda L: -len(L.profiles)):
        k = cfg.count(L)
        lab = "Full" if L.profiles == ("7g.269gb",) else (
            f"{len(L.profiles)}분할" if len(set(L.profiles)) == 1
            else "+".join(f"{L.profiles.count(p)}×{p.split('.')[0]}"
                          for p in dict.fromkeys(L.profiles)))
        parts.append(f"{k}×{lab}" if k > 1 else lab)
    return " + ".join(parts)


def enumerate_configs(max_gpus, catalog=None):
    cat = catalog or M.layout_catalog()
    for n in range(1, max_gpus + 1):
        for cfg in combinations_with_replacement(cat, n):
            yield list(cfg)


def solve(rate, mix, max_gpus=8, catalog=None, all_optima=False):
    """Cheapest configuration that sustains `rate`; None if none does."""
    tpls = list(mix)
    best_n, winners = None, []
    for cfg in enumerate_configs(max_gpus, catalog):
        n = len(cfg)
        if best_n is not None and n > best_n:
            break                                   # N is non-decreasing
        if not M.covers(cfg, tpls):                 # (F) hard feasibility
            continue
        cap = M.capacity(cfg, mix)
        if cap < rate:                              # (C) must carry the load
            continue
        if best_n is None or n < best_n:
            best_n, winners = n, []
        winners.append((cap, cfg))
    if not winners:
        return None
    winners.sort(key=lambda x: -x[0])               # tie-break: most headroom
    return winners if all_optima else winners[0]


def load_slo(name):
    path = name if name.endswith(".json") else f"slo_configs/{name}.json"
    return {k: float(v) for k, v in json.load(open(path)).items()}


def solve_slo(rate, mix, slo, beta=0.95, alpha=None, max_gpus=8, catalog=None,
              n_requests=150, cv=1.0, seeds=(1, 2, 3), all_optima=False):
    """Cheapest configuration meeting BOTH the rate and the SLO floors.

    Two constraints, and both are needed:

      stability   simulated capacity >= lambda.  A finite trace hides marginal
                  overload -- 150 requests at 2% over capacity still finish and
                  still meet their deadlines -- but a service runs indefinitely,
                  so an unstable queue is not a solution however good the
                  simulated SAR looks.
      quality     SAR >= beta, and optionally a per-template floor.

    The closed-form capacity is only a cheap pre-filter (with a 15% margin,
    since it can under-predict a sliced pool by ~6%); the binding capacity
    number comes from the simulation, which handles heterogeneous pools.
    Several arrival seeds are used and the *worst* has to clear both floors --
    a provisioning decision that only holds on a lucky trace is not a decision.
    """
    tpls = list(mix)
    traces = [SIM.gamma_arrivals(n_requests, rate, cv, mix, seed=s) for s in seeds]
    best_n, winners = None, []
    for cfg in enumerate_configs(max_gpus, catalog):
        n = len(cfg)
        if best_n is not None and n > best_n:
            break
        if not M.covers(cfg, tpls):
            continue
        if M.capacity(cfg, mix) < rate * 0.85:          # cheap pre-filter
            continue
        sars, worsts, p95s, caps = [], [], [], []
        for arr in traces:
            r = SIM.simulate(cfg, arr)
            if r is None:
                break
            m = SIM.metrics(r, slo)
            sars.append(m["sar"])
            worsts.append(m["sar_worst"])
            p95s.append(m["p95_s"])
            caps.append(m["capacity"])
        if len(sars) != len(traces):
            continue
        if min(caps) < rate:                            # (C) stability
            continue
        if min(sars) < beta:                            # (S) aggregate floor
            continue
        if alpha is not None and min(worsts) < alpha:   # (A) per-class floor
            continue
        if best_n is None or n < best_n:
            best_n, winners = n, []
        winners.append({"cfg": cfg, "sar": min(sars), "sar_worst": min(worsts),
                        "p95": max(p95s), "cap": min(caps)})
    if not winners:
        return None
    winners.sort(key=lambda w: (-w["sar"], w["p95"]))
    return winners if all_optima else winners[0]


def full_only(rate, mix, max_gpus=8):
    """The naive answer: never partition."""
    return solve(rate, mix, max_gpus, catalog=[M.Layout(("7g.269gb",))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, default=1.1)
    ap.add_argument("--mix", default="short", choices=list(MIXES))
    ap.add_argument("--max-gpus", type=int, default=8)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--show", type=int, default=3, help="alternatives to list")
    ap.add_argument("--validated-only", action="store_true",
                    help="restrict to layouts we have measured")
    ap.add_argument("--slo", default=None,
                    help="SLO config name, e.g. video_3.0x -- enables the SAR constraint")
    ap.add_argument("--beta", type=float, default=0.95, help="aggregate SAR floor")
    ap.add_argument("--alpha", type=float, default=None,
                    help="per-template SAR floor (fairness)")
    ap.add_argument("-n", "--n-requests", type=int, default=300)
    ap.add_argument("--cv", type=float, default=1.0)
    args = ap.parse_args()
    mix = MIXES[args.mix]
    cat = ([L for L in M.layout_catalog() if L.profiles in VALIDATED]
           if args.validated_only else None)

    if args.slo and not args.sweep:
        slo = load_slo(args.slo)
        res = solve_slo(args.rate, mix, slo, args.beta, args.alpha,
                        args.max_gpus, cat, args.n_requests, args.cv,
                        all_optima=True)
        cap_only = solve(args.rate, mix, args.max_gpus, catalog=cat)
        fo = full_only(args.rate, mix, args.max_gpus)
        print(f"λ = {args.rate} req/s  ·  '{args.mix}'  ·  마감 {args.slo}"
              f"  ·  SAR ≥ {args.beta}"
              + (f", 종류별 ≥ {args.alpha}" if args.alpha else "") + "\n")
        if res is None:
            print(f"  해 없음 — GPU {args.max_gpus}장으로 이 SLO 를 못 지킴")
            if cap_only:
                print(f"  (용량 제약만이면 GPU {len(cap_only[1])}장으로 충분)")
            return
        w = res[0]
        n = len(w["cfg"])
        mark = "" if is_validated(w["cfg"]) else "   ⚠ 미측정 구성"
        print(f"  해:        GPU {n}장   {name(w['cfg'])}{mark}")
        print(f"             SAR {w['sar']:.3f}   최악 종류 {w['sar_worst']:.3f}"
              f"   p95 {w['p95']:.1f}s   용량 {w['cap']:.2f}/s")
        if cap_only:
            nc = len(cap_only[1])
            print(f"  용량 제약만:  GPU {nc}장   {name(cap_only[1])}")
            if n > nc:
                print(f"  → SLO 를 지키려면 GPU {n-nc}장 더 필요")
        if fo:
            nf = len(fo[1])
            print(f"  Full 만:     GPU {nf}장 (용량 기준)")
            if nf > n:
                print(f"  → GPU {nf-n}장 절약  ({(1-n/nf)*100:.0f}%)")
        if len(res) > 1:
            print(f"\n  같은 {n}장짜리 대안:")
            for x in res[1:args.show]:
                print(f"      SAR {x['sar']:.3f}  p95 {x['p95']:>6.1f}s   {name(x['cfg'])}")
        return

    if not args.sweep:
        res = solve(args.rate, mix, args.max_gpus, catalog=cat, all_optima=True)
        if res is None:
            print(f"λ={args.rate}: GPU {args.max_gpus}장으로 불가능")
            return
        n = len(res[0][1])
        fo = full_only(args.rate, mix, args.max_gpus)
        print(f"λ = {args.rate} req/s   ·   워크로드 '{args.mix}'\n")
        mark = "" if is_validated(res[0][1]) else "   ⚠ 미측정 구성 (외삽)"
        print(f"  해:        GPU {n}장   {name(res[0][1])}{mark}")
        print(f"             용량 {res[0][0]:.3f} req/s  (여유 {(res[0][0]/args.rate-1)*100:+.0f}%)")
        if fo:
            nf = len(fo[1])
            print(f"  Full만:    GPU {nf}장   용량 {fo[0]:.3f} req/s")
            if nf > n:
                print(f"  → GPU {nf-n}장 절약  ({(1-n/nf)*100:.0f}%)")
            else:
                print(f"  → 절약 없음 (Full로도 {nf}장이면 충분)")
        if len(res) > 1:
            print(f"\n  같은 {n}장짜리 대안 (용량 순):")
            for cap, cfg in res[1:args.show]:
                print(f"      {cap:>6.3f} req/s   {name(cfg)}")
        return

    # ------------------------------------------------------------ sweep
    print(f"워크로드 '{args.mix}'   ·   λ 를 훑으며 solver 의 답이 어떻게 바뀌는가\n")
    print(f"{'λ (req/s)':>10} {'해':>34} {'GPU':>4} {'용량':>9} | "
          f"{'Full만':>7} {'절약':>6}")
    prev = None
    for i in range(1, 61):
        rate = round(0.15 * i, 2)
        res = solve(rate, mix, args.max_gpus, catalog=cat)
        fo = full_only(rate, mix, args.max_gpus)
        if res is None:
            print(f"{rate:>10} {'— GPU 8장으로 불가 —':>34}")
            break
        cap, cfg = res
        n, nf = len(cfg), (len(fo[1]) if fo else None)
        key = (n, name(cfg), nf)
        if key == prev:                         # only print where the answer changes
            continue
        prev = key
        save = f"−{nf-n}장" if nf and nf > n else "—"
        print(f"{rate:>10} {name(cfg):>34} {n:>4} {cap:>8.2f}/s | "
              f"{(str(nf)+'장') if nf else '불가':>7} {save:>6}")


if __name__ == "__main__":
    main()
