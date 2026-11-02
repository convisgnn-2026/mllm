#!/usr/bin/env python
"""LLM side of the pool split: co-located Llama-3.1-8B on 1..4 x RTX 4090 at several arrival rates.
Feeds solver/cosolve.py.  Output results/servingsim/pool/summary.csv"""
import csv, json, sys, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)); import sweep_pd_kvo as S
OUT = Path(__file__).resolve().parents[1] / "results/servingsim/pool"; OUT.mkdir(parents=True, exist_ok=True)
S.OUT = OUT
RATES = [1, 2, 3, 4, 6, 8]
rows = []
for g in (1, 2, 3, 4):
    for r in RATES:
        wl = S.WL / f"sharegpt-8b-300-sps{r}.jsonl"
        if not wl.exists():
            src = S.WL / "sharegpt-8b-300-sps10.jsonl"; rs = [json.loads(l) for l in open(src)]; t0 = min(x["arrival_time_ns"] for x in rs)
            with open(wl, "w") as w:
                for x in rs: q = dict(x); q["arrival_time_ns"] = int((x["arrival_time_ns"] - t0) * 10 / r); w.write(json.dumps(q) + "\n")
        d = S.run(f"colo{g}", S.cluster(0, g, "npu"), r)
        if (d / "sim.csv").exists():
            s = S.score(d); s.update(cfg=f"colo{g}", gpus=g, rate=r); rows.append(s)
keys = ["cfg", "gpus", "rate", "sar", "goodput", "ttft_p90", "tpot_p90", "makespan_s"]
with open(OUT / "summary.csv", "w") as f:
    w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
print("wrote", OUT / "summary.csv")
