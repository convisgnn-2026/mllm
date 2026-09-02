#!/usr/bin/env python
"""
Experiment D — rescore every fairness run across the SLO sweep.  No GPU.

Each run in results/fair/ carries per-request arrival/end timestamps, so the
same run can be judged against any deadline after the fact.  This produces
the second axis of the decision map (load × SLO slack): at which slack does
partition stop helping the worst class?

Writes results/fair_slo_sweep.md and prints it.
"""
import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != "/" and not _os.path.isdir(_os.path.join(_d, "results")):
    _d = _os.path.dirname(_d)
_sys.path[:0] = [_os.path.join(_d, "solver"), _os.path.join(_d, "bench")]
_os.chdir(_d)

import glob
import json
from collections import defaultdict

USER = {"480x272x33": "A", "832x480x33": "B", "832x480x49": "C", "1280x720x49": "D"}
SLOS = ["video_1.0x", "video_1.5x", "video_2.0x", "video_3.0x", "video_4.0x", "video_6.0x"]
CAP_2F = 0.687


def score(path, slo):
    d = json.load(open(path))
    per = defaultdict(list)
    for r in d["requests"]:
        per[USER[r["tpl"]]].append((r["end_s"] - r["arrival_s"]) <= slo[r["tpl"]])
    sar = {k: sum(v) / len(v) for k, v in per.items()}
    n = sum(len(v) for v in per.values())
    return sum(sum(v) for v in per.values()) / n, min(sar.values()), sar


def main(out="results/fair_slo_sweep.md"):
    runs = {}
    for f in sorted(glob.glob("results/fair/*.json")):
        d = json.load(open(f))
        runs[(d["policy"], d["queue"], d["workload_config"]["rate"])] = f
    rates = sorted({k[2] for k in runs})
    lines = ["# Experiment D — fairness runs rescored across SLO slack", "",
             "same 24 runs, same trace; only the deadline changes.  "
             "cells: aggregate / worst-class SAR.  Δ = 2x2way − 2F.", ""]
    for slon in SLOS:
        slo = {k: float(v) for k, v in json.load(open(f"slo_configs/{slon}.json")).items()}
        lines += [f"## SLO {slon.split('_')[1]}", "",
                  "| load | queue | Full ×2 | 2-way ×2 | Δ agg | Δ worst | 2-way wins? |",
                  "|---|---|---|---|---|---|---|"]
        wins = tot = 0
        for r in rates:
            for q in ["fcfs", "sjf", "edf", "vtc"]:
                a, b = runs.get(("2F", q, r)), runs.get(("2x2way", q, r))
                if not (a and b):
                    continue
                A, B = score(a, slo), score(b, slo)
                da, dw = B[0] - A[0], B[1] - A[1]
                w = "both ↑" if da > 0.005 and dw > 0.005 else ("worst ↑" if dw > 0.005 else
                    ("agg ↑" if da > 0.005 else "no"))
                wins += dw > 0.005; tot += 1
                lines.append(f"| {r/CAP_2F*100:.0f}% | {q} | {A[0]:.2f} / {A[1]:.2f} | "
                             f"{B[0]:.2f} / {B[1]:.2f} | {da:+.2f} | {dw:+.2f} | {w} |")
        lines += ["", f"worst-class improved in **{wins}/{tot}** cells", ""]
    txt = "\n".join(lines)
    open(out, "w").write(txt)
    print(txt)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
