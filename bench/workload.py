#!/usr/bin/env python
"""
Workload generation, following TetriServe's benchmark conventions.

Two independent knobs, as in TetriServe:

  arrival  -- gamma process with (rate, cv).  cv=1 is Poisson, cv>1 bursty,
              cv<1 more regular.  shape k = 1/cv^2, scale = cv^2/rate, so the
              mean inter-arrival is 1/rate and the CV is exactly `cv`.

  shape    -- which request template arrives.  TetriServe sweeps `evenly` and
              `long_seqlen_high` (p ~ exp(seqlen/16384), i.e. skewed toward the
              most expensive requests).  We keep the same functional forms and
              substitute video templates for their image resolutions.

Templates are chosen so that every one of them fits a 1g.34gb MIG slice once
the text encoder is disaggregated (measured: 6.5 / 12.2 / 12.3 / 23.9 GiB
against a 30.5 GiB slice).  1920x1088 is deliberately excluded -- it needs
49.5 GiB and is simply infeasible on the small slice, which is a separate
(already measured) result rather than something to average into SAR.
"""

import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != "/" and not _os.path.isdir(_os.path.join(_d, "results")):
    _d = _os.path.dirname(_d)
_sys.path[:0] = [_os.path.join(_d, "solver"), _os.path.join(_d, "bench")]
_os.chdir(_d)            # results/ and slo_configs/ are addressed from the repo root

import argparse
import json
import math
import os

import numpy as np

# (width, height, frames) -- spans ~15x in cost, all feasible on 1g.34gb
TEMPLATES = [
    (480, 272, 33),
    (832, 480, 33),
    (832, 480, 49),
    (1280, 720, 49),
]


def tpl_name(t):
    return "x".join(str(v) for v in t)


def seq_len(t, patch=(1, 2, 2), sp=8, tp=4):
    """Post-patchify sequence length -- the quantity attention is O(N^2) in."""
    w, h, f = t
    lf, lh, lw = (f - 1) // tp + 1, h // sp, w // sp
    return (lf // patch[0]) * (lh // patch[1]) * (lw // patch[2])


# --------------------------------------------------------------------------
def explicit_probs(spec):
    """'0.5,0.25,0.15,0.10' -> normalised probabilities over TEMPLATES."""
    v = [float(x) for x in spec.split(",")]
    assert len(v) == len(TEMPLATES), f"need {len(TEMPLATES)} values"
    t = sum(v)
    return [x / t for x in v]


def shape_probs(kind):
    """TetriServe's distributions, evaluated on our templates."""
    s = [seq_len(t) for t in TEMPLATES]
    if kind == "evenly":
        raw = [1.0] * len(s)
    elif kind == "long_seqlen_high":        # favours the expensive templates
        raw = [math.exp(x / max(s)) for x in s]
    elif kind == "short_seqlen_high":       # favours the cheap ones
        raw = [1.0 / (1.0 + math.exp(x / (max(s) / 4))) for x in s]
    else:
        raise ValueError(f"unknown shape distribution {kind!r}")
    tot = sum(raw)
    return [r / tot for r in raw]


def gamma_arrivals(n, rate, cv, rng):
    """Inter-arrival times with mean 1/rate and coefficient of variation cv."""
    k = 1.0 / (cv * cv)
    scale = (cv * cv) / rate
    gaps = rng.gamma(shape=k, scale=scale, size=n)
    return np.cumsum(gaps) - gaps[0]        # first request arrives at t=0


def generate(n, rate, cv, shape_dist, seed, probs=None):
    rng = np.random.default_rng(seed)
    probs = probs or shape_probs(shape_dist)
    idx = rng.choice(len(TEMPLATES), size=n, p=probs)
    at = gamma_arrivals(n, rate, cv, rng)
    return [{"id": i, "tpl": tpl_name(TEMPLATES[j]), "arrival_s": round(float(a), 4),
             "seq": seq_len(TEMPLATES[j])}
            for i, (j, a) in enumerate(zip(idx, at))]


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--num", type=int, default=100)
    ap.add_argument("--rate", type=float, default=0.15, help="requests/second")
    ap.add_argument("--cv", type=float, default=1.0, help="1=Poisson, >1 bursty")
    ap.add_argument("--shape-dist", default="evenly",
                    choices=["evenly", "long_seqlen_high", "short_seqlen_high"])
    ap.add_argument("--probs", default=None,
                    help="explicit per-template probabilities, e.g. 0.5,0.25,0.15,0.10")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    pr = explicit_probs(args.probs) if args.probs else None
    reqs = generate(args.num, args.rate, args.cv, args.shape_dist, args.seed, pr)

    if args.stats or not args.out:
        probs = pr or shape_probs(args.shape_dist)
        print(f"distribution '{args.shape_dist}':")
        print(f"  {'template':>14} {'seq':>8} {'p(target)':>10} {'p(actual)':>10} {'n':>5}")
        for t, p in zip(TEMPLATES, probs):
            c = sum(1 for r in reqs if r["tpl"] == tpl_name(t))
            print(f"  {tpl_name(t):>14} {seq_len(t):>8} {p:>10.3f} "
                  f"{c/len(reqs):>10.3f} {c:>5}")
        gaps = np.diff([r["arrival_s"] for r in reqs])
        print(f"arrival: n={args.num} rate={args.rate} cv={args.cv} -> "
              f"span {reqs[-1]['arrival_s']:.0f}s, "
              f"mean gap {gaps.mean():.2f}s, cv {gaps.std()/gaps.mean():.2f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump({"config": vars(args), "requests": reqs}, open(args.out, "w"),
                  indent=1)
        print(f"wrote {args.out}  ({len(reqs)} requests)")


if __name__ == "__main__":
    main()
