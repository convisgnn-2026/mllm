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

IMG_TOKS = 576               # LLaVA-1.5 @336px, one image per request
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


def load_vlm(enc_ms, root="results/servingsim/pool_vlm", ttft_ms=3000, tpot_ms=60):
    """VLM tenant = image-token-inflated LLM run + fixed vision-encoder latency per request.
    The encoder runs on the same GPUs as the language model (ms-scale, never the bottleneck);
    its latency is charged against the TTFT budget."""
    tab = {}
    if not _os.path.isdir(root):
        return tab
    for d in _os.listdir(root):
        p = _os.path.join(root, d, "sim.csv")
        if not d.startswith("vlm_colo") or not _os.path.exists(p):
            continue
        g = int(d.split("_")[1][4:]); rate = float(d.split("_sps")[1])
        rows = list(csv.DictReader(open(p)))
        ok = sum(1 for r in rows if float(r["TTFT"]) / 1e6 + enc_ms <= ttft_ms
                 and float(r["TPOT"]) / 1e6 <= tpot_ms)
        tab[(g, rate)] = ok / len(rows)
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
    ap.add_argument("--vlm-rates", default="0", help="VLM tenant req/s list; each value gets its own table")
    ap.add_argument("--enc-ms", type=float, default=25.0, help="vision encoder latency per image on the pool GPU (assumed)")
    ap.add_argument("-o", "--out", default="results/servingsim/pool/cosolve.md")
    a = ap.parse_args()
    G = a.gpus
    dr = [float(x) for x in a.dit_rates.split(",")]
    lr = [float(x) for x in a.llm_rates.split(",")]
    vr = [float(x) for x in a.vlm_rates.split(",")]
    mix = MIXES[a.dit_mix]
    tab = load_llm()
    vtab = load_vlm(a.enc_ms)

    # per-half tables
    D = {(g, r): dit_sar(g, r, mix, a.dit_scale) for g in range(G + 1) for r in dr}
    L = {(g, r): llm_sar(tab, g, r) for g in range(G + 1) for r in lr}
    V = {(g, r): llm_sar(vtab, g, r) for g in range(G + 1) for r in vr}

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

    if any(r > 0 for r in vr):
        lines += ["", f"## VLM tenant alone: SAR by (GPUs, req/s)  (+{IMG_TOKS} image tokens, encoder {a.enc_ms:g} ms)", "",
                  "| GPUs | " + " | ".join(f"{r:g}" for r in vr) + " |", "|---|" + "---|" * len(vr)]
        for g in range(1, G + 1):
            lines.append(f"| {g} | " + " | ".join((f"{V[(g, r)]:.2f}" if V[(g, r)] == V[(g, r)] else "-") for r in vr) + " |")

    lines += ["", f"## Co-solver: best split (DiT:LLM:VLM GPUs) per demand triple, floor {ALPHA}"]
    grand = [0, 0]
    for rv in vr:
        lines += ["", f"### VLM = {rv:g} req/s", "",
                  "| DiT req/s \\ LLM req/s | " + " | ".join(f"{r:g}" for r in lr) + " |", "|---|" + "---|" * len(lr)]
        n_feas = n_tot = 0
        fixed = {}
        for rd in dr:
            cells = []
            for rl in lr:
                best = None
                for gd in range(0, G + 1):
                    for gl in range(0, G - gd + 1):
                        for gv in range(0, G - gd - gl + 1):     # idle GPUs allowed: they are the saving
                            sd, sl, sv = D[(gd, rd)], L[(gl, rl)], V[(gv, rv)]
                            if sl != sl or sv != sv:
                                continue
                            ok = sd >= ALPHA and sl >= ALPHA and sv >= ALPHA
                            used = (gd if rd > 0 else 0) + (gl if rl > 0 else 0) + (gv if rv > 0 else 0)
                            tot = sd + sl + sv
                            key = (0, used, -tot) if ok else (1, -tot, used)
                            if best is None or key < best[0]:
                                best = (key, gd, gl, gv, sd, sl, sv, ok)
                            for fx in ((1, 2, 1), (2, 1, 1), (1, 1, 2), (2, 2, 0), (1, 3, 0)):
                                if (gd, gl, gv) == fx:
                                    fixed[fx] = fixed.get(fx, 0) + ok
                if best is None:
                    cells.append("-"); continue
                _, gd, gl, gv, sd, sl, sv, ok = best
                n_tot += 1; n_feas += ok
                idle = G - gd - gl - gv
                tag = f"{gd}:{gl}:{gv}" if rv > 0 else f"{gd}:{gl}"
                cells.append((f"{tag} ✓" + (f" +{idle} idle" if idle else "")) if ok
                             else f"{tag} ✗ ({sd:.2f}/{sl:.2f}" + (f"/{sv:.2f})" if rv > 0 else ")"))
            lines.append(f"| {rd:g} | " + " | ".join(cells) + " |")
        grand[0] += n_feas; grand[1] += n_tot
        lines += ["", f"feasible: {n_feas}/{n_tot}.  ✗ = least-bad split with (DiT/LLM" + ("/VLM" if rv > 0 else "") + " SAR).",
                  "fixed splits: " + ", ".join(f"{a_}:{b_}:{c_} → {fixed.get((a_, b_, c_), 0)}/{n_tot}"
                                             for (a_, b_, c_) in ((1, 2, 1), (2, 1, 1), (1, 1, 2), (2, 2, 0), (1, 3, 0))
                                             if (rv > 0) == (c_ > 0))]
    lines += ["", f"**all tables: {grand[0]}/{grand[1]} demand points feasible with the adaptive split.**"]
    txt = "\n".join(lines)
    _os.makedirs(_os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
