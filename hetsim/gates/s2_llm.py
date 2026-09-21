#!/usr/bin/env python
"""S2 gate: LLM co-located instances (1..4 x RTX 4090, Llama-3.1-8B, ShareGPT
300 requests, 1..8 req/s) must reproduce LLMServingSim 2.0's SAR table
(results/servingsim/pool/summary.csv; SLO = TTFT <= 3 s and TPOT <= 60 ms).

    ~/venv/jinwoo/bin/python -m hetsim.gates.s2_llm
"""
import csv
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from hetsim.spec import CardType, Card
from hetsim.core import Group, Simulator
from hetsim.engine_llm import BatchInstance, least_load
from hetsim.model import llm
from hetsim.score import metrics
from hetsim.workload import from_jsonl

CARD_4090 = CardType("RTX4090", sm=128, mem_gib=24.0, tflops=165.0, bw_gbps=1008.0)
SLO_TTFT, SLO_TPOT = 3.0, 0.060


def slo_hit(req):
    a = req.attrs
    return a["ttft"] <= SLO_TTFT and a["tpot"] <= SLO_TPOT


def llm_group(n, spec=llm.LLAMA31_8B, prof=llm.RTX4090, name="llm/colo"):
    inst = [BatchInstance(f"llm{i}", Card(f"g{i}", CARD_4090), spec, prof) for i in range(n)]
    return Group(name, "llm", inst, discipline="fcfs", selector=least_load(inst))


def run(n, rate, wl_prefix="sharegpt-8b-300"):
    path = os.path.join(ROOT, "results/servingsim/workloads", f"{wl_prefix}-sps{rate}.jsonl")
    reqs = from_jsonl("llm", path, n=300)
    res = Simulator([llm_group(n)]).run(reqs)
    m = metrics(res, slo=slo_hit)
    ttft = sorted(r.attrs["ttft"] for r in res.requests)
    tpot = sorted(r.attrs["tpot"] for r in res.requests)
    p = lambda v, q: v[min(len(v) - 1, int(len(v) * q))]
    m["ttft_p90"], m["tpot_p90"] = p(ttft, .9) * 1e3, p(tpot, .9) * 1e3
    m["preempt"] = sum(i.preemptions for i in res.instances)
    m["recomp"] = sum(i.recomputed for i in res.instances)
    return m


def main():
    ref = {}
    for r in csv.DictReader(open(os.path.join(ROOT, "results/servingsim/pool/summary.csv"))):
        ref[(int(r["gpus"]), int(r["rate"]))] = r
    print(f"{'cfg':>6} {'rate':>4} | {'SAR ref':>7} {'SAR sim':>7} {'Δ':>6} | {'TTFT90 ref':>10} {'sim':>8} | {'TPOT90 ref':>10} {'sim':>6} | {'mk ref':>6} {'sim':>6} | pre  t")
    devs = []
    for (g, rate), r in sorted(ref.items()):
        t0 = time.perf_counter()
        m = run(g, rate)
        dt = time.perf_counter() - t0
        d = m["sar"] - float(r["sar"])
        devs.append(abs(d))
        print(f"colo{g:<2} {rate:>4} | {float(r['sar']):7.3f} {m['sar']:7.3f} {d:+6.3f} | "
              f"{float(r['ttft_p90']):10.0f} {m['ttft_p90']:8.0f} | {float(r['tpot_p90']):10.1f} {m['tpot_p90']:6.1f} | "
              f"{float(r['makespan_s']):6.1f} {m['makespan_s']:6.1f} | {m["preempt"]:3d} {m["recomp"]:6d} {dt*1e3:5.0f}ms")
    mean, mx = sum(devs) / len(devs), max(devs)
    ok = mean <= 0.05 and mx <= 0.10
    print(f"\n  SAR |Δ| mean {mean:.3f} (≤0.05)  max {mx:.3f} (≤0.10)   n={len(devs)}")
    print("S2 GATE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
