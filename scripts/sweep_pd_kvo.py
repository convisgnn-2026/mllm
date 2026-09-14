#!/usr/bin/env python
"""
Sweep 1 — does KV offloading move the optimal prefill:decode ratio?

Pool: 4 x RTX 4090 (one node, PCIe).  Model: Llama-3.1-8B.  Workload: ShareGPT
300 requests, arrival scaled to 10/20/30/40 req/s.

Configurations
  colo        4 co-located instances (pd_type null)               -- vLLM default
  pd{P}:{D}   P prefill + D decode instances, P+D = 4
  x kv        npu   : KV on the 4090 (24 GB, weights 16 GB -> small KV room)
              host  : decode KV on a host-memory tier reached over PCIe 4.0 x16
                      (modelled as a CXL-type expansion: 25 GB/s, 1 us, 128 GB)

Outputs results/servingsim/sweep1/<cfg>_sps<r>/sim.csv and summary.csv
"""
import csv, json, os, subprocess, sys, time
from pathlib import Path

REPO = Path.home() / "papers/refs/code/LLMServingSim"
OUT = Path.home() / "vidbench/results/servingsim/sweep1"
WL = Path.home() / "vidbench/results/servingsim/workloads"
# the simulator resolves paths relative to the repo root; ext/ is a symlink to results/servingsim
REL = "ext"
PY = Path.home() / "venv/servingsim/bin/python"
OUT.mkdir(parents=True, exist_ok=True)

GPU = {"mem_size": 24, "mem_bw": 1008, "mem_latency": 0, "mem_util": 0.833919}
CPU = {"mem_size": 128, "mem_bw": 50, "mem_latency": 0}
HOST_KV = {"mem_size": 128, "mem_bw": 25, "mem_latency": 1000, "num_devices": 1}   # PCIe 4.0 x16
MODEL = "meta-llama/Llama-3.1-8B"
RATES = [2, 4, 6, 8, 10]
WL_PREFIX = "sharegpt-8b-300"       # workload family; sweep_*_pool.py override this
EXTRA_ARGS = []                     # extra serving CLI flags (e.g. parking: --enable-prefix-caching --prefix-storage CPU)
SLO = {"ttft_ms": 3000, "tpot_ms": 60}          # goodput definition; both must hold


def instance(pd, kv, dev=0, frac=1.0):
    inst = {"model_name": MODEL, "hardware": "RTX4090", "npu_mem": dict(GPU),
            "pd_type": pd, "tp_size": 1}
    if kv == "host" and pd in ("decode", None):
        # one host-memory "device" per decode GPU: each has its own PCIe link
        inst["placement"] = {"default": {"weights": "npu", "kv_loc": f"cxl:{dev}",
                                         "kv_evict_loc": "cpu", "kv_host_frac": frac}}
    return inst


def cluster(P, D, kv, frac=1.0):
    if P:
        insts = [instance("prefill", kv) for _ in range(P)] + \
                [instance("decode", kv, dev=i, frac=frac) for i in range(D)]
    else:
        insts = [instance(None, kv, dev=i, frac=frac) for i in range(D)]
    c = {"num_nodes": 1, "link_bw": 16, "link_latency": 20000,
         "nodes": [{"num_instances": len(insts), "cpu_mem": dict(CPU), "instances": insts}]}
    if kv == "host":
        c["cxl_mem"] = dict(HOST_KV); c["cxl_mem"]["num_devices"] = D
    return c


def run(name, cfg, rate):
    d = OUT / f"{name}_sps{rate}"
    if (d / "sim.csv").exists():
        return d
    d.mkdir(exist_ok=True)
    cfg_path = d / "cluster.json"
    cfg_path.write_text(json.dumps(cfg, indent=1))
    cmd = [str(PY), "-m", "serving", "--cluster-config", f"{REL}/{OUT.name}/{d.name}/cluster.json",
           "--dataset", f"{REL}/workloads/{WL_PREFIX}-sps{rate}.jsonl",
           "--output", f"{REL}/{OUT.name}/{d.name}/sim.csv", "--num-reqs", "300",
           "--dtype", "bfloat16", "--kv-cache-dtype", "auto", "--block-size", "16",
           "--max-num-seqs", "256", "--max-num-batched-tokens", "2048",
           "--log-level", "WARNING", "--network-backend", "analytical"] + list(EXTRA_ARGS)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    (d / "sim.log").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr)
    ok = (d / "sim.csv").exists()
    print(f"  {name:>10} sps{rate:<3} {'ok' if ok else 'FAIL'}  {time.time()-t0:5.1f}s", flush=True)
    if not ok:
        print("    " + (r.stderr.strip().splitlines() or ["?"])[-1][:160], flush=True)
    return d


def score(d):
    rows = list(csv.DictReader(open(d / "sim.csv")))
    ttft = sorted(float(r["TTFT"]) / 1e6 for r in rows)
    tpot = sorted(float(r["TPOT"]) / 1e6 for r in rows)
    lat = sorted(float(r["latency"]) / 1e6 for r in rows)
    good = sum(1 for r in rows if float(r["TTFT"]) / 1e6 <= SLO["ttft_ms"]
               and float(r["TPOT"]) / 1e6 <= SLO["tpot_ms"])
    mk = (max(float(r["end_time"]) for r in rows) - min(float(r["arrival"]) for r in rows)) / 1e9
    p = lambda v, q: v[min(len(v) - 1, int(len(v) * q))]
    return dict(n=len(rows), ttft_p50=p(ttft, .5), ttft_p90=p(ttft, .9), tpot_p50=p(tpot, .5),
                tpot_p90=p(tpot, .9), lat_p90=p(lat, .9), goodput=good / mk, sar=good / len(rows),
                makespan_s=mk)


def main():
    configs = [("colo", cluster(0, 4, "npu"))]
    for P, D in [(1, 3), (2, 2), (3, 1)]:
        configs.append((f"pd{P}{D}_f000", cluster(P, D, "npu")))
        configs.append((f"pd{P}{D}_f010", cluster(P, D, "host", frac=0.10)))

    summary = []
    for rate in RATES:
        for name, cfg in configs:
            d = run(name, cfg, rate)
            if (d / "sim.csv").exists():
                s = score(d); s.update(cfg=name, rate=rate); summary.append(s)
    keys = ["cfg", "rate", "sar", "goodput", "ttft_p50", "ttft_p90", "tpot_p50", "tpot_p90", "lat_p90", "makespan_s"]
    with open(OUT / "summary.csv", "w") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(summary)
    print(f"\n{'cfg':>10} {'sps':>4} {'SAR':>5} {'goodput':>8} {'TTFT p90':>9} {'TPOT p90':>9}")
    for s in summary:
        print(f"{s['cfg']:>10} {s['rate']:>4} {s['sar']:>5.2f} {s['goodput']:>8.2f} {s['ttft_p90']:>8.0f}ms {s['tpot_p90']:>8.1f}ms")
    print("wrote", OUT / "summary.csv")


if __name__ == "__main__":
    main()
