#!/usr/bin/env python
"""
Check model.py against every request we have actually served.

Two things are validated separately, because they can fail for different reasons:

  service time  -- per (template, SM) mean, over 7,130 served requests
  capacity      -- the sustainable rate a configuration can hold, recovered
                   from each run as  throughput / utilisation

Capacity is the quantity the solver's provisioning decision rests on, so it is
the one that has to be right.
"""

import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != "/" and not _os.path.isdir(_os.path.join(_d, "results")):
    _d = _os.path.dirname(_d)
_sys.path[:0] = [_os.path.join(_d, "solver"), _os.path.join(_d, "bench")]
_os.chdir(_d)            # results/ and slo_configs/ are addressed from the repo root

import glob
import json
import statistics as st
from collections import defaultdict

import model as M


def measured_service():
    obs = defaultdict(list)
    for f in (glob.glob("results/runs/*.json") + glob.glob("results/expB/*.json")
              + glob.glob("results/small/*.json")):
        for r in json.load(open(f))["requests"]:
            if "service_s" in r and "sm" in r:
                obs[(r["tpl"], r["sm"])].append(r["service_s"])
    return obs


def run_mix(path):
    """Realised template mix of a run (not the nominal probabilities)."""
    reqs = json.load(open(path))["requests"]
    c = defaultdict(int)
    for r in reqs:
        c[r["tpl"]] += 1
    return {k: v / len(reqs) for k, v in c.items()}


def layouts_of(run):
    """Recover the partitioning from the worker SM counts a run recorded."""
    sms = sorted((w["sm"] for w in run["workers"].values()), reverse=True)
    by_sm = {v[0]: k for k, v in M.PROFILES.items()}
    per_gpu = run["n_gpus"]
    profs = [by_sm[s] for s in sms if s in by_sm]
    if len(profs) != len(sms):
        return None
    # group workers into GPUs: equal-sized groups when the pool is uniform
    if len(set(profs)) == 1:
        k = len(profs) // per_gpu
        return [M.Layout(tuple(profs[i * k:(i + 1) * k])) for i in range(per_gpu)]
    full = [p for p in profs if p == "7g.269gb"]
    rest = [p for p in profs if p != "7g.269gb"]
    if not rest or len(rest) % (per_gpu - len(full)) != 0:
        return None
    k = len(rest) // (per_gpu - len(full))
    out = [M.Layout(("7g.269gb",)) for _ in full]
    out += [M.Layout(tuple(rest[i * k:(i + 1) * k]))
            for i in range(per_gpu - len(full))]
    return out


def main():
    # ---------------------------------------------------- service time
    obs = measured_service()
    print("① 서비스 시간   (7,130건)")
    print(f"{'템플릿':>14} {'SM':>5} {'n':>6} {'실측':>9} {'예측':>9} {'오차':>8}")
    errs = []
    for (t, sm), v in sorted(obs.items(), key=lambda x: (x[0][0], -x[0][1])):
        m, p = st.mean(v), M.service(t, sm)
        e = (p - m) / m
        errs.append(abs(e))
        print(f"{t:>14} {sm:>5} {len(v):>6} {m:>8.2f}s {p:>8.2f}s {e*100:>+7.1f}%")
    print(f"{'':>14} {'':>5} {'':>6} {'':>9} {'MAPE':>9} {sum(errs)/len(errs)*100:>7.1f}%"
          f"   최대 {max(errs)*100:.1f}%")

    # ---------------------------------------------------- capacity
    print("\n② 처리 용량   (throughput / utilisation 으로 실측 복원)")
    print(f"{'구성':>22} {'워크로드':>18} {'GPU':>4} {'실측':>9} {'예측':>9} {'오차':>8}")
    rows, cerrs = [], []
    for f in sorted(glob.glob("results/small/*.json") + glob.glob("results/expB/*.json")):
        run = json.load(open(f))
        u = run.get("utilization")
        if not u or u < 0.55:               # far from saturation: capacity not observable
            continue
        Ls = layouts_of(run)
        if Ls is None:
            continue
        mix = run_mix(f)
        meas = len(run["requests"]) / run["makespan_s"] / u
        pred = M.capacity(Ls, mix)
        e = (pred - meas) / meas
        cerrs.append(abs(e))
        wl = run["workload_config"]["shape_dist"][:12]
        rows.append((run["policy"], f"{wl}/r{run['workload_config']['rate']}",
                     run["n_gpus"], meas, pred, e))
    for pol, wl, g, meas, pred, e in sorted(rows, key=lambda r: (r[0], r[1])):
        print(f"{pol:>22} {wl:>18} {g:>4} {meas:>8.2f}/s {pred:>8.2f}/s {e*100:>+7.1f}%")
    if cerrs:
        print(f"{'':>22} {'':>18} {'':>4} {'':>9} {'MAPE':>9} "
              f"{sum(cerrs)/len(cerrs)*100:>7.1f}%   최대 {max(cerrs)*100:.1f}%   n={len(cerrs)}")


if __name__ == "__main__":
    main()
