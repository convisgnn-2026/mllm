"""Arrivals and request classes, TetriServe conventions (bench/workload.py).

gamma arrivals with (rate, cv): shape 1/cv^2, scale cv^2/rate.  cv=1 Poisson.
Draw order is class indices first, then gaps -- identical to bench/workload.py
so a seed reproduces the traces the real B300 runs were served with.
"""
import numpy as np

from .spec import Request


def gamma_gaps(n, rate, cv, rng):
    k, scale = 1.0 / (cv * cv), (cv * cv) / rate
    gaps = rng.gamma(shape=k, scale=scale, size=n)
    return np.cumsum(gaps) - gaps[0]


def generate(tenant, n, rate, cv, mix, seed=42, id0=0):
    """mix: cls -> probability."""
    rng = np.random.default_rng(seed)
    classes = list(mix)
    p = np.array([mix[c] for c in classes], float)
    p /= p.sum()
    idx = rng.choice(len(classes), size=n, p=p)
    at = gamma_gaps(n, rate, cv, rng)
    return [Request(id0 + i, tenant, classes[j], float(a)) for i, (j, a) in enumerate(zip(idx, at))]


def from_records(tenant, records, cls_key="tpl", t_key="arrival_s"):
    """Requests from a recorded run's request list (exact replay)."""
    return [Request(i, tenant, r[cls_key], float(r[t_key])) for i, r in enumerate(records)]


def from_jsonl(tenant, path, n=None):
    """LLMServingSim request trace: input_toks, output_toks, arrival_time_ns."""
    import json
    out = []
    for i, line in enumerate(open(path)):
        if n is not None and i >= n:
            break
        r = json.loads(line)
        out.append(Request(i, tenant, "llm", r["arrival_time_ns"] / 1e9,
                           {"input": int(r["input_toks"]), "output": int(r["output_toks"])}))
    return out
