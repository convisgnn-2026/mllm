#!/usr/bin/env python
"""S1 gate: DiT-only cluster must reproduce (a) the legacy solver/sim.py replay
exactly and (b) the 39 measured B300 runs within the legacy error
(makespan MAPE 1.6%, capacity 2.1%, p95 7.6%).

    ~/venv/jinwoo/bin/python -m hetsim.gates.s1_dit
"""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from hetsim.spec import CardType, Card
from hetsim.core import SlotInstance, Group, Simulator
from hetsim.model import dit
from hetsim.score import metrics
from hetsim.workload import from_records

BIG = frozenset({"1280x720x49"})


def dit_group(worker_sms, model=dit.B300, discipline="srpt", selector="match"):
    """One DiT group whose workers are MIG slices identified by SM count."""
    by_sm = {v[0]: (k, v[1]) for k, v in dit.MIG_PROFILES.items()}
    inst = []
    for i, sm in enumerate(worker_sms):
        name, mem = by_sm[sm]
        ct = CardType(name, sm, mem)
        inst.append(SlotInstance(f"dit{i}", Card(f"g{i}", ct), model))
    return Group("dit/full", "dit", inst, discipline, selector,
                 rank=lambda r: model.t_full[r.cls], big=BIG)


def replay(path):
    run = json.load(open(path))
    sms = sorted((w["sm"] for w in run["workers"].values()), reverse=True)
    reqs = from_records("dit", run["requests"])
    g = dit_group(sms, discipline=run.get("queue", "srpt"), selector=run.get("select", "match"))
    res = Simulator([g]).run(reqs)
    m = metrics(res)
    lat = sorted(r["end_s"] - r["arrival_s"] for r in run["requests"])
    real = {"makespan": run["makespan_s"], "p95": lat[int(len(lat) * .95) - 1],
            "cap": len(run["requests"]) / run["makespan_s"] / run["utilization"]}
    pred = {"makespan": m["makespan_s"], "p95": m["p95_s"], "cap": m["capacity"]}
    return run, real, pred, res


def legacy(path):
    """Same replay through the old solver/sim.py, for the exactness check."""
    sys.path.insert(0, os.path.join(ROOT, "solver"))
    import sim as SIM
    from validate import layouts_of
    run = json.load(open(path))
    s = SIM.simulate(layouts_of(run), [(r["arrival_s"], r["tpl"]) for r in run["requests"]],
                     queue=run.get("queue", "srpt"), select=run.get("select", "match"))
    return SIM.metrics(s)


def main():
    os.chdir(ROOT)
    files = sorted(glob.glob("results/small/*.json") + glob.glob("results/expB/*.json"))
    errs = {k: [] for k in ("makespan", "p95", "cap")}
    max_dev = 0.0
    print(f"{'구성':>22} {'워크로드':>17} | {'makespan':>21}{'p95':>21}{'용량':>21} | vs legacy")
    for f in files:
        run, real, pred, res = replay(f)
        leg = legacy(f)
        dev = max(abs(pred["makespan"] - leg["makespan_s"]), abs(pred["p95"] - leg["p95_s"]))
        max_dev = max(max_dev, dev)
        cells = []
        for k in ("makespan", "p95", "cap"):
            e = (pred[k] - real[k]) / real[k]
            errs[k].append(abs(e))
            cells.append(f"{real[k]:6.2f}→{pred[k]:6.2f} {e*100:+5.0f}%".rjust(21))
        wc = run["workload_config"]
        print(f"{run['policy']:>22} {wc['shape_dist'][:11]+'/r'+str(wc['rate']):>17} | "
              + "".join(cells) + f" | {dev:.2e}")
    print()
    ok = True
    for k, lim in (("makespan", 0.02), ("cap", 0.03), ("p95", 0.08)):
        mape = sum(errs[k]) / len(errs[k])
        flag = "PASS" if mape <= lim else "FAIL"
        ok &= mape <= lim
        print(f"  {k:>8}  MAPE {mape*100:5.1f}%  max {max(errs[k])*100:5.1f}%  n={len(errs[k])}  [{flag} ≤{lim*100:.0f}%]")
    exact = max_dev < 1e-6
    ok &= exact
    print(f"  legacy sim.py exactness: max |Δ| = {max_dev:.2e}  [{'PASS' if exact else 'FAIL'}]")
    print("\nS1 GATE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
