#!/usr/bin/env python
"""
Discrete-event queue simulator -- the fast stand-in for a real serving run.

`bench/serve_bench.py` already produces real queues on real GPUs; that is the
ground truth and we have 55 runs of it.  What it cannot do is evaluate the
thousands of configurations the solver enumerates but never runs.  This module
reproduces that dispatcher exactly, in milliseconds instead of minutes, so the
search can score configurations it has not measured.

It matters for two things the closed-form capacity model cannot express:

  SAR / p95   need waiting time, which needs a queue.
  heterogeneous pools   the closed form assumes every worker sees the same
                        template mix.  Under size-matched dispatch it does not:
                        Full GPUs take the big requests and slices take the
                        small ones.  That is why the one heterogeneous
                        configuration we measured came in 11% under prediction.

The scheduler below mirrors serve_bench.py line for line -- SRPT queue order,
and for `match` the biggest request goes to the highest-SM free worker while
everything else goes to the lowest-SM free worker that fits.
"""
import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != "/" and not _os.path.isdir(_os.path.join(_d, "results")):
    _d = _os.path.dirname(_d)
_sys.path[:0] = [_os.path.join(_d, "solver"), _os.path.join(_d, "bench")]
_os.chdir(_d)

import heapq

import numpy as np

import model as M

BIG_DEFAULT = frozenset({"1280x720x49"})


def simulate(layouts, arrivals, queue="srpt", select="match",
             big=BIG_DEFAULT, disaggregated_t5=True):
    """Replay `arrivals` = [(arrival_s, template), ...] against a configuration.

    Returns one record per request with its completion time, plus the pool's
    busy time so utilisation and capacity can be recovered the same way we
    recover them from a real run.
    """
    workers = M.config_workers(layouts)
    if not workers:
        return None
    # precompute service times; a worker that cannot hold a template never gets it
    svc = {(t, i): (M.service(t, w.sm)
                    if M.feasible(t, w, disaggregated_t5=disaggregated_t5) else None)
           for t in {t for _, t in arrivals}
           for i, w in enumerate(workers)}
    if any(all(svc[(t, i)] is None for i in range(len(workers)))
           for _, t in arrivals):
        return None                                  # some template unservable

    order = sorted(arrivals, key=lambda a: a[0])
    free = list(range(len(workers)))
    busy = []                                        # heap of (finish, worker)
    pend, done = [], []
    nxt, t, busy_time = 0, 0.0, 0.0

    while len(done) < len(order):
        # nothing to do and nothing running -> jump to the next arrival
        if not busy and not (pend and free) and nxt < len(order):
            t = max(t, order[nxt][0])
        while nxt < len(order) and order[nxt][0] <= t + 1e-12:
            pend.append(order[nxt])
            nxt += 1
        while busy and busy[0][0] <= t + 1e-12:
            fin, w, a, tpl = heapq.heappop(busy)
            free.append(w)
            done.append({"tpl": tpl, "arrival_s": a, "end_s": fin,
                         "sm": workers[w].sm, "service_s": svc[(tpl, w)]})

        progressed = False
        while free and pend:
            # queue discipline
            i = (min(range(len(pend)), key=lambda k: M.T_FULL[pend[k][1]])
                 if queue == "srpt" else 0)
            a, tpl = pend[i]
            cand = [w for w in free if svc[(tpl, w)] is not None]
            if not cand:
                break                                # nothing free can hold it
            pend.pop(i)
            if select == "any":
                w = min(cand)
            else:                                    # size matching
                w = (max(cand, key=lambda w: workers[w].sm) if tpl in big
                     else min(cand, key=lambda w: workers[w].sm))
            free.remove(w)
            fin = t + svc[(tpl, w)]
            busy_time += svc[(tpl, w)]
            heapq.heappush(busy, (fin, w, a, tpl))
            progressed = True

        if progressed:
            continue
        cands = [busy[0][0]] if busy else []
        if nxt < len(order):
            cands.append(order[nxt][0])
        if not cands:
            break
        t = max(t, min(cands))

    makespan = max(r["end_s"] for r in done) - min(a for a, _ in order)
    return {"requests": done, "makespan_s": makespan,
            "n_workers": len(workers), "busy_s": busy_time,
            "utilization": busy_time / (len(workers) * makespan)}


def metrics(sim, slo=None):
    """SAR / tail / capacity from a simulate() result."""
    reqs = sim["requests"]
    lat = sorted(r["end_s"] - r["arrival_s"] for r in reqs)
    out = {"makespan_s": sim["makespan_s"],
           "utilization": sim["utilization"],
           "capacity": len(reqs) / sim["makespan_s"] / max(sim["utilization"], 1e-9),
           "mean_s": sum(lat) / len(lat),
           "p95_s": lat[int(len(lat) * 0.95) - 1]}
    if slo:
        hit = [(r["end_s"] - r["arrival_s"]) <= slo[r["tpl"]] for r in reqs]
        out["sar"] = sum(hit) / len(hit)
        per = {}
        for r, ok in zip(reqs, hit):
            per.setdefault(r["tpl"], []).append(ok)
        out["sar_per_template"] = {k: sum(v) / len(v) for k, v in per.items()}
        out["sar_worst"] = min(out["sar_per_template"].values())
    return out


def gamma_arrivals(n, rate, cv, mix, seed=42):
    """Same generator as bench/workload.py, for solver-side what-ifs."""
    rng = np.random.default_rng(seed)
    k, scale = 1.0 / (cv * cv), (cv * cv) / rate
    gaps = rng.gamma(k, scale, n)
    at = np.cumsum(gaps) - gaps[0]
    tpls = list(mix)
    p = np.array([mix[t] for t in tpls], float)
    p /= p.sum()
    idx = rng.choice(len(tpls), size=n, p=p)
    return [(float(a), tpls[i]) for a, i in zip(at, idx)]


if __name__ == "__main__":
    import solver as S
    mix = S.MIXES["short"]
    arr = gamma_arrivals(150, 1.0, 1.0, mix)
    for lab, cfg in [("Full 2장", [M.Layout(("7g.269gb",))] * 2),
                     ("2장 모두 2분할", [M.Layout(("3g.135gb",) * 2)] * 2),
                     ("Full 1 + 2분할 1", [M.Layout(("7g.269gb",)),
                                           M.Layout(("3g.135gb",) * 2)])]:
        m = metrics(simulate(cfg, arr))
        print(f"{lab:>16}  용량 {m['capacity']:.3f}/s  이용률 {m['utilization']*100:.0f}%  "
              f"평균 {m['mean_s']:.1f}s  p95 {m['p95_s']:.1f}s")
