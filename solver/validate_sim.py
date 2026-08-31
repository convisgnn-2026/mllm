#!/usr/bin/env python
"""
Validate the simulator by replaying every real run's own arrival trace.

Each recorded run carries each request's arrival time and template, so the
replay is exact -- same trace, same configuration, same dispatch policy.  The
only thing the simulator supplies is the service time, from model.py.

The test that matters is the heterogeneous pool: the closed-form capacity model
over-predicts `2 Full + one 7-way GPU` by 11% because it assumes every worker
sees the same mix.  If the simulator lands on the measurement instead, the
whole layout catalogue opens up to the solver.
"""
import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != "/" and not _os.path.isdir(_os.path.join(_d, "results")):
    _d = _os.path.dirname(_d)
_sys.path[:0] = [_os.path.join(_d, "solver"), _os.path.join(_d, "bench")]
_os.chdir(_d)

import glob
import json

import model as M
import sim as SIM
from validate import layouts_of


def replay(path):
    run = json.load(open(path))
    Ls = layouts_of(run)
    if Ls is None:
        return None
    arr = [(r["arrival_s"], r["tpl"]) for r in run["requests"]]
    policy = run.get("queue", "srpt"), run.get("select", "match")
    s = SIM.simulate(Ls, arr, queue=policy[0], select=policy[1])
    if s is None:
        return None

    lat = sorted(r["end_s"] - r["arrival_s"] for r in run["requests"])
    real = {"makespan": run["makespan_s"],
            "p95": lat[int(len(lat) * .95) - 1],
            "mean": sum(lat) / len(lat),
            "cap": (len(run["requests"]) / run["makespan_s"] / run["utilization"]
                    if run.get("utilization") else None)}
    m = SIM.metrics(s)
    pred = {"makespan": m["makespan_s"], "p95": m["p95_s"],
            "mean": m["mean_s"], "cap": m["capacity"]}
    return run, real, pred


def main():
    files = sorted(glob.glob("results/small/*.json") + glob.glob("results/expB/*.json"))
    rows = []
    for f in files:
        r = replay(f)
        if r:
            rows.append((f,) + r)

    print(f"{'구성':>22} {'워크로드':>17} | " +
          "".join(f"{k:>21}" for k in ["makespan", "p95", "용량"]))
    print(f"{'':>22} {'':>17} | " + "".join(f"{'실측 → 예측 (오차)':>21}" for _ in range(3)))
    errs = {k: [] for k in ("makespan", "p95", "cap")}
    for f, run, real, pred in rows:
        wc = run["workload_config"]
        cells = []
        for k in ("makespan", "p95", "cap"):
            if real[k] is None:
                cells.append(f"{'-':>21}")
                continue
            e = (pred[k] - real[k]) / real[k]
            errs[k].append(abs(e))
            u = "/s" if k == "cap" else "s"
            cells.append(f"{real[k]:>6.2f}→{pred[k]:>6.2f}{u} {e*100:>+5.0f}%".rjust(21))
        print(f"{run['policy']:>22} "
              f"{wc['shape_dist'][:11]+'/r'+str(wc['rate']):>17} | " + "".join(cells))

    print()
    for k, label in (("makespan", "makespan"), ("p95", "p95 지연"), ("cap", "처리 용량")):
        if errs[k]:
            print(f"  {label:>10}  MAPE {sum(errs[k])/len(errs[k])*100:5.1f}%   "
                  f"최대 {max(errs[k])*100:5.1f}%   n={len(errs[k])}")

    # the decisive case
    print("\n=== 이종 풀 (닫힌 수식이 +11% 틀린 곳) ===")
    for f, run, real, pred in rows:
        if "1x7way" not in run["policy"]:
            continue
        cf = M.capacity(layouts_of(run),
                        {t: sum(1 for r in run["requests"] if r["tpl"] == t) / len(run["requests"])
                         for t in {r["tpl"] for r in run["requests"]}})
        print(f"  {run['policy']} / {run['workload_config']['shape_dist'][:11]}")
        print(f"     실측 {real['cap']:.3f}/s |  닫힌 수식 {cf:.3f}/s ({(cf/real['cap']-1)*100:+.1f}%)"
              f" |  시뮬레이터 {pred['cap']:.3f}/s ({(pred['cap']/real['cap']-1)*100:+.1f}%)")


if __name__ == "__main__":
    main()
