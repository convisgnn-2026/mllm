#!/usr/bin/env python
"""S3 gate: PD disaggregation and KV parking against LLMServingSim 2.0.

  (a) PD  : 4 x RTX 4090 as P:D = 1:3, 2:2, 3:1 at 2..10 req/s
            (results/servingsim/sweep1/summary.csv, rows pd{P}{D}_f000)
  (b) park: colo 1..4 GPUs with the host parking tier on
            (results/servingsim/pool_park/summary.csv)

    ~/venv/jinwoo/bin/python -m hetsim.gates.s3_pd_park
"""
import csv
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from hetsim.core import Simulator, Router
from hetsim.engine_llm import HostTier
from hetsim.builders import llm_colo, llm_pd, pd_rule
from hetsim.score import metrics
from hetsim.workload import from_jsonl
from hetsim.gates.s2_llm import slo_hit

WL = os.path.join(ROOT, "results/servingsim/workloads")


def run(groups, rate, wl_prefix="sharegpt-8b-300"):
    reqs = from_jsonl("llm", os.path.join(WL, f"{wl_prefix}-sps{rate}.jsonl"), n=300)
    sim = Simulator(groups, Router(groups, pd_rule(groups)))
    res = sim.run(reqs)
    m = metrics(res, slo=slo_hit)
    ttft = sorted(r.attrs["ttft"] for r in res.requests)
    tpot = sorted(r.attrs["tpot"] for r in res.requests)
    p = lambda v, q: v[min(len(v) - 1, int(len(v) * q))]
    m["ttft_p90"], m["tpot_p90"] = p(ttft, .9) * 1e3, p(tpot, .9) * 1e3
    m["preempt"] = sum(i.preemptions for i in res.instances)
    m["recomp"] = sum(i.recomputed for i in res.instances)
    m["recall"] = sum(i.recalled for i in res.instances)
    return m


def table(title, rows):
    print(f"\n== {title}")
    print(f"{'cfg':>10} {'rate':>4} | {'SAR ref':>7} {'SAR sim':>7} {'Δ':>6} | {'TTFT90 ref':>10} {'sim':>8} | {'TPOT90 ref':>10} {'sim':>6} | {'mk ref':>6} {'sim':>6} | pre recomp recall")
    devs = []
    for name, rate, ref, m in rows:
        d = m["sar"] - ref["sar"]
        devs.append(abs(d))
        print(f"{name:>10} {rate:>4} | {ref['sar']:7.3f} {m['sar']:7.3f} {d:+6.3f} | "
              f"{ref['ttft_p90']:10.0f} {m['ttft_p90']:8.0f} | {ref['tpot_p90']:10.1f} {m['tpot_p90']:6.1f} | "
              f"{ref['makespan_s']:6.1f} {m['makespan_s']:6.1f} | {m['preempt']:3d} {m['recomp']:6d} {m['recall']:6d}")
    # knee cells (oracle SAR strictly between 0.25 and 0.95) sit on a knife edge:
    # the two simulators route 1/3 of requests to different instances there and
    # the per-instance backlogs diverge chaotically.  They get a wider tolerance.
    knee = [abs(m["sar"] - ref["sar"]) for _, _, ref, m in rows if 0.25 < ref["sar"] < 0.95]
    flat = [abs(m["sar"] - ref["sar"]) for _, _, ref, m in rows if not (0.25 < ref["sar"] < 0.95)]
    mean = sum(devs) / len(devs)
    mx_flat, mx_knee = max(flat or [0]), max(knee or [0])
    print(f"   SAR |Δ| mean {mean:.3f} (≤0.05)  max off-knee {mx_flat:.3f} (≤0.10)  max knee {mx_knee:.3f} (≤0.20, n={len(knee)})")
    return mean <= 0.05 and mx_flat <= 0.10 and mx_knee <= 0.20


def main():
    f = lambda r: {k: float(r[k]) for k in ("sar", "ttft_p90", "tpot_p90", "makespan_s")}
    sweep1 = {(r["cfg"], int(r["rate"])): f(r)
              for r in csv.DictReader(open(os.path.join(ROOT, "results/servingsim/sweep1/summary.csv")))}
    park = {(int(r["gpus"]), int(r["rate"])): f(r)
            for r in csv.DictReader(open(os.path.join(ROOT, "results/servingsim/pool_park/summary.csv")))}

    rows = []
    for P, D in ((1, 3), (2, 2), (3, 1)):
        for rate in (2, 4, 6, 8, 10):
            m = run(llm_pd(P, D), rate)
            rows.append((f"pd{P}:{D}", rate, sweep1[(f"pd{P}{D}_f000", rate)], m))
    for rate in (10,):
        rows.append(("colo4", rate, sweep1[("colo", rate)], run(llm_colo(4), rate)))
    a = table("(a) PD disaggregation, no parking", rows)

    rows = []
    for g in (1, 2, 3, 4):
        for rate in (1, 2, 3, 4, 6, 8):
            m = run(llm_colo(g, host=HostTier(bw_gbps=50.0, capacity_gib=128.0)), rate)
            rows.append((f"colo{g}+park", rate, park[(g, rate)], m))
    b = table("(b) colo with KV parking (host tier 50 GB/s)", rows)

    ok = a and b
    print("\nS3 GATE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
