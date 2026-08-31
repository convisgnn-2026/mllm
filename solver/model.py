#!/usr/bin/env python
"""
Cost / capacity / feasibility model for MIG-partitioned video DiT serving.

Everything here is fitted to measurement, and `validate.py` re-checks it against
the 7,130 served requests we have on disk.

Service time
    t(tpl, S) = t_full(tpl) * ( 148*(1-B)/S + B )

    A fraction B of a request's work does not scale with SM count -- kernel
    launches, the memory-bound part of VAE decode, host-side serialisation.
    The rest scales inversely with SM.  B = 0.184 fitted globally; the form
    passes through t_full exactly at S = 148, so a new template needs only one
    Full-GPU measurement to be usable.  MAPE 3.0% over 12 (template, SM) pairs.

Feasibility
    A slice must hold the resident weights plus the request's peak activation.
    This is a hard, discrete constraint: violating it does not make the request
    slow, it makes it unservable.  It is what turns partition choice into a
    combinatorial problem rather than a smooth trade-off.

Capacity
    Work-conserving pool: every worker can take any request it fits, so the
    sustainable arrival rate is the sum of per-worker service rates.
"""

import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != "/" and not _os.path.isdir(_os.path.join(_d, "results")):
    _d = _os.path.dirname(_d)
_sys.path[:0] = [_os.path.join(_d, "solver"), _os.path.join(_d, "bench")]
_os.chdir(_d)            # results/ and slo_configs/ are addressed from the repo root

from dataclasses import dataclass

# ---------------------------------------------------------------- measured
FULL_SM = 148
B_SCALE = 0.184                 # share of work that does not scale with SM

# Full-GPU latency, 20 denoising steps + VAE decode  (seconds)
T_FULL = {
    "480x272x33": 0.85,
    "832x480x33": 2.62,
    "832x480x49": 4.06,
    "1280x720x49": 12.22,
}
# per-template refinement; falls back to B_SCALE when absent (MAPE 2.0% -> 3.0%)
B_TPL = {"480x272x33": 0.232, "832x480x33": 0.154,
         "832x480x49": 0.154, "1280x720x49": 0.156}

# peak activation, i.e. peak allocation minus resident weights  (GiB)
ACT_PEAK = {
    "480x272x33": 3.30, "832x480x33": 9.10, "832x480x49": 9.20,
    "1280x720x49": 20.40, "1920x1088x81": 46.40,
}
W_DIT, W_VAE, W_T5 = 2.65, 0.47, 10.58        # resident weights (GiB)

# MIG profiles measured on B300: name -> (SM, memory GiB, compute slices, memory slices)
# A GPU has 7 compute slices and 8 memory slices; a layout must fit in BOTH,
# which is why e.g. 1g.67gb x2 + 1g.34gb x5 is illegal (9 memory slices) even
# though its compute slices add up to 7.
PROFILES = {
    "7g.269gb": (148, 268.0, 7, 8),
    "4g.135gb": (72, 133.5, 4, 4),
    "3g.135gb": (70, 133.5, 3, 4),
    "2g.67gb": (36, 66.5, 2, 2),
    "1g.67gb": (30, 66.5, 1, 2),
    "1g.34gb": (18, 30.5, 1, 1),
}
COMPUTE_SLICES, MEMORY_SLICES = 7, 8           # per physical GPU


@dataclass(frozen=True)
class Worker:
    sm: int
    mem: float
    profile: str


@dataclass(frozen=True)
class Layout:
    """One physical GPU's partitioning, as a tuple of profile names."""
    profiles: tuple

    @property
    def workers(self):
        return tuple(Worker(*PROFILES[p][:2], p) for p in self.profiles)

    @property
    def label(self):
        if self.profiles == ("7g.269gb",):
            return "Full"
        n = len(self.profiles)
        return f"{n}분할" if len(set(self.profiles)) == 1 else "+".join(self.profiles)

    def valid(self):
        """MIG placement: fit within both the compute and the memory slices."""
        return (sum(PROFILES[p][2] for p in self.profiles) <= COMPUTE_SLICES and
                sum(PROFILES[p][3] for p in self.profiles) <= MEMORY_SLICES)


def service(tpl, sm, per_template=True):
    """Predicted service time (s) for one request on a worker with `sm` SMs."""
    b = B_TPL.get(tpl, B_SCALE) if per_template else B_SCALE
    return T_FULL[tpl] * (FULL_SM * (1 - b) / sm + b)


def resident_weights(disaggregated_t5=True):
    return W_DIT + W_VAE + (0.0 if disaggregated_t5 else W_T5)


def feasible(tpl, worker, disaggregated_t5=True):
    """Can this worker run this request at all?  Hard constraint."""
    return ACT_PEAK[tpl] + resident_weights(disaggregated_t5) <= worker.mem


def config_workers(layouts):
    return [w for L in layouts for w in L.workers]


def covers(layouts, templates, **kw):
    """Every template must be servable somewhere, or the config is invalid."""
    ws = config_workers(layouts)
    return all(any(feasible(t, w, **kw) for w in ws) for t in templates)


def capacity(layouts, mix, **kw):
    """Sustainable arrival rate (req/s) of a work-conserving pool.

    A worker contributes 1 / E[service] only over the templates it can hold;
    if it cannot hold some template, that template's share of the traffic must
    be carried by the others, so the worker's usable rate shrinks accordingly.
    """
    total = 0.0
    for w in config_workers(layouts):
        share = sum(p for t, p in mix.items() if feasible(t, w, **kw))
        if share <= 0:
            continue
        exp_t = sum(p * service(t, w.sm) for t, p in mix.items()
                    if feasible(t, w, **kw)) / share
        total += share / exp_t
    return total


def n_gpus(layouts):
    return len(layouts)


# ------------------------------------------------------------------ catalog
def layout_catalog():
    """Partitionings worth considering, valid under MIG placement rules."""
    cands = [
        ("7g.269gb",),                                   # Full
        ("3g.135gb",) * 2,                               # 2분할
        ("2g.67gb",) * 3,                                # 3분할
        ("1g.67gb",) * 4,                                # 4분할
        ("1g.34gb",) * 7,                                # 7분할
        ("3g.135gb",) + ("1g.34gb",) * 4,                # 큰 조각 1 + 작은 조각 4
        ("2g.67gb",) * 2 + ("1g.34gb",) * 3,
        ("1g.67gb",) * 2 + ("1g.34gb",) * 5,           # illegal: 9 memory slices
        ("1g.67gb",) * 2 + ("1g.34gb",) * 4,
        ("4g.135gb",) + ("1g.34gb",) * 3,
    ]
    return [L for L in (Layout(c) for c in cands) if L.valid()]


if __name__ == "__main__":
    print(f"{'레이아웃':>34} {'워커':>4} {'총SM':>5} {'총메모리':>9}  실행 가능 템플릿")
    tpls = list(T_FULL) + ["1920x1088x81"]
    for L in layout_catalog():
        ws = L.workers
        ok = [t.split("x")[0] + "p" for t in tpls if any(feasible(t, w) for w in ws)]
        name = (f"{len(ws)}x{L.profiles[0]}" if len(set(L.profiles)) == 1
                else "+".join(f"{L.profiles.count(p)}x{p}" for p in dict.fromkeys(L.profiles)))
        print(f"{name:>34} {len(ws):>4} "
              f"{sum(w.sm for w in ws):>5} {sum(w.mem for w in ws):>8.1f}G  "
              f"{'전부' if len(ok) == len(tpls) else ','.join(ok) or '없음'}")
