#!/usr/bin/env python
"""
Pool-split co-solver, step 1 of the DiT + LLM integration.

A fixed pool of G identical GPUs is split into g_dit DiT workers and
G - g_dit LLM instances.  Each half is evaluated by its own simulator:

  DiT  -> solver/sim.py            (discrete-event, validated on B300, 2% MAPE)
  LLM  -> LLMServingSim 2.0        (results pre-computed by scripts/sweep_llm_pool.py)

For every (DiT rate, LLM rate) pair the solver enumerates the splits, scores
per-tenant SLO attainment, and picks the cheapest feasible split (fewest GPUs
that satisfy both tenants) or reports "infeasible".  Colocation on one GPU
(SM sharing) is deliberately *not* modelled here; that is step 2.

Hardware: the LLM side is the profiled RTX 4090.  The DiT side has no 4090
profile yet, so B300 service times are scaled by --dit-scale (default 10x,
~ the bf16 dense-throughput ratio 1710 / 165 TFLOP/s).  Re-profile on the
desktop and replace the scale with measured T_FULL.
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

import model as M
import sim as SIM
from solver import MIXES

DIT_SLO_SLACK = 3.0          # deadline = slack x solo latency on the pool's hardware
ALPHA = 0.90                 # per-tenant SLO attainment floor


def dit_sar(g, rate, mix, scale, n=200, seed=7):
    """SLO attainment of a DiT tenant on g Full workers of the scaled hardware."""
    if rate <= 0:
        return 1.0
    if g == 0:
        return 0.0
    saved = dict(M.T_FULL)
    M.T_FULL = {k: v * scale for k, v in saved.items()}
    try:
        slo = {k: DIT_SLO_SLACK * v for k, v in M.T_FULL.items()}
        arr = SIM.gamma_arrivals(n, rate, 1.0, mix, seed=seed)
        s = SIM.simulate([M.Layout(("7g.269gb",))] * g, arr, queue="srpt", select="match")
        return SIM.metrics(s, slo)["sar"] if s else 0.0
    finally:
        M.T_FULL = saved


def load_llm(path="results/servingsim/pool/summary.csv"):
    tab = {}
    if not _os.path.exists(path):
        return tab
    for r in csv.DictReader(open(path)):
        tab[(int(r["gpus"]), float(r["rate"]))] = float(r["sar"])
    return tab


def llm_sar(tab, g, rate):
    if g == 0:
        return 0.0 if rate > 0 else 1.0
    if rate == 0:
        return 1.0
    v = tab.get((g, float(rate)))
    return v if v is not None else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--dit-scale", type=float, default=10.0)
    ap.add_argument("--dit-mix", default="short", choices=list(MIXES))
    ap.add_argument("--dit-rates", default="0,0.01,0.02,0.04,0.06")
    ap.add_argument("--llm-rates", default="0,1,2,3,4,6")
    ap.add_argument("-o", "--out", default="results/servingsim/pool/cosolve.md")
    a = ap.parse_args()
    G = a.gpus
    dr = [float(x) for x in a.dit_rates.split(",")]
    lr = [float(x) for x in a.llm_rates.split(",")]
    mix = MIXES[a.dit_mix]
    tab = load_llm()

    # per-half tables
    D = {(g, r): dit_sar(g, r, mix, a.dit_scale) for g in range(G + 1) for r in dr}
    L = {(g, r): llm_sar(tab, g, r) for g in range(G + 1) for r in lr}

    lines = [f"# Pool-split co-solver  ({G} x RTX 4090-class, DiT x{a.dit_scale:g} of B300, mix={a.dit_mix}, "
             f"SLO: DiT {DIT_SLO_SLACK}x solo, LLM TTFT<=3s & TPOT<=60ms, floor {ALPHA})", ""]
    lines += ["## DiT tenant alone: SAR by (workers, req/s)", "",
              "| workers | " + " | ".join(f"{r:g}" for r in dr) + " |", "|---|" + "---|" * len(dr)]
    for g in range(1, G + 1):
        lines.append(f"| {g} | " + " | ".join(f"{D[(g, r)]:.2f}" for r in dr) + " |")
    lines += ["", "## LLM tenant alone: SAR by (GPUs, req/s)  (LLMServingSim, colo)", "",
              "| GPUs | " + " | ".join(f"{r:g}" for r in lr) + " |", "|---|" + "---|" * len(lr)]
    for g in range(1, G + 1):
        lines.append(f"| {g} | " + " | ".join((f"{L[(g, r)]:.2f}" if L[(g, r)] == L[(g, r)] else "-") for r in lr) + " |")

    lines += ["", f"## Co-solver: best split (DiT:LLM GPUs) per demand pair, floor {ALPHA}", "",
              "| DiT req/s \\ LLM req/s | " + " | ".join(f"{r:g}" for r in lr) + " |", "|---|" + "---|" * len(lr)]
    n_feas = n_tot = 0
    for rd in dr:
        cells = []
        for rl in lr:
            best = None
            for gd in range(0, G + 1):
                for gl in range(0, G - gd + 1):       # idle GPUs allowed: they are the saving
                    sd, sl = D[(gd, rd)], L[(gl, rl)]
                    if sl != sl:                       # nan: LLM table missing this rate
                        continue
                    ok = sd >= ALPHA and sl >= ALPHA
                    used = (gd if rd > 0 else 0) + (gl if rl > 0 else 0)
                    # feasible: fewest GPUs, then highest attainment; infeasible: least bad
                    key = (0, used, -(sd + sl)) if ok else (1, -(sd + sl), used)
                    if best is None or key < best[0]:
                        best = (key, gd, gl, sd, sl, ok)
            if best is None:
                cells.append("-"); continue
            _, gd, gl, sd, sl, ok = best
            n_tot += 1; n_feas += ok
            idle = G - gd - gl
            cells.append((f"{gd}:{gl} ✓" + (f" +{idle} idle" if idle else "")) if ok else f"{gd}:{gl} ✗ ({sd:.2f}/{sl:.2f})")
        lines.append(f"| {rd:g} | " + " | ".join(cells) + " |")
    lines += ["", f"feasible pairs: {n_feas}/{n_tot}.  ✗ shows the least-bad split with (DiT SAR / LLM SAR).",
              "", "Static baselines for comparison (same table, fixed split):", ""]
    for gd in (1, 2):
        gl = G - gd; f = 0
        for rd in dr:
            for rl in lr:
                if L[(gl, rl)] == L[(gl, rl)]:
                    f += (D[(gd, rd)] >= ALPHA and L[(gl, rl)] >= ALPHA)
        lines.append(f"- fixed {gd}:{gl} → feasible pairs {f}/{n_tot}")
    txt = "\n".join(lines)
    _os.makedirs(_os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
