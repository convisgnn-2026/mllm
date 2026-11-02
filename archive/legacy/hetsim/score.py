"""Score a finished run.  All metrics are post-hoc so one run can be scored
against many SLOs (bench/score_slo.py convention)."""


def metrics(result, slo=None, work=None):
    """slo   : cls -> deadline seconds (latency <= deadline is a hit), or a
               callable(req) -> bool for compound SLOs (TTFT & TPOT)
       work  : cls -> reference GPU-seconds of one request (for work-weighted goodput)
    """
    reqs = result.requests
    lat = sorted(r.latency for r in reqs)
    out = {
        "n": len(reqs),
        "makespan_s": result.makespan,
        "utilization": result.utilization,
        "capacity": len(reqs) / result.makespan / max(result.utilization, 1e-9),
        "mean_s": sum(lat) / len(lat),
        "p95_s": lat[int(len(lat) * 0.95) - 1],
    }
    if slo:
        hit = [slo(r) for r in reqs] if callable(slo) else [r.latency <= slo[r.cls] for r in reqs]
        out["sar"] = sum(hit) / len(hit)
        per = {}
        for r, ok in zip(reqs, hit):
            per.setdefault(r.cls, []).append(ok)
        out["sar_per_class"] = {k: sum(v) / len(v) for k, v in per.items()}
        out["sar_worst"] = min(out["sar_per_class"].values())
        span = out["makespan_s"]
        out["goodput_req"] = sum(hit) / span                       # met requests / s
        if work:
            w = [work[r.cls] for r in reqs]
            out["goodput_work"] = sum(wi for wi, ok in zip(w, hit) if ok) / span   # met GPU-s / s
            out["work_sat"] = sum(wi for wi, ok in zip(w, hit) if ok) / sum(w)      # share of work met
    return out
