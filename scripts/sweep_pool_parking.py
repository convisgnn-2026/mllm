#!/usr/bin/env python
"""LLM and VLM pool tables with KV parking on (prefix cache + CPU storage tier, evict-to-CPU restore).
Outputs results/servingsim/pool_park/summary.csv and results/servingsim/pool_vlm_park/summary.csv"""
import csv, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)); import sweep_pd_kvo as S
S.EXTRA_ARGS = ["--enable-prefix-caching", "--prefix-storage", "CPU"]
RATES = [1, 2, 3, 4, 6, 8]; IMG_TOKS = 576
base = [json.loads(l) for l in open(S.WL / "sharegpt-8b-300-sps10.jsonl")]; t0 = min(x["arrival_time_ns"] for x in base)
def sweep(outdir, prefix, tag, vlm):
    S.OUT = Path.home() / f"vidbench/results/servingsim/{outdir}"; S.OUT.mkdir(parents=True, exist_ok=True); S.WL_PREFIX = prefix
    for r in RATES:
        wl = S.WL / f"{prefix}-sps{r}.jsonl"
        if not wl.exists():
            with open(wl, "w") as w:
                for i, x in enumerate(base):
                    q = dict(x); q["arrival_time_ns"] = int((x["arrival_time_ns"] - t0) * 10 / r)
                    if vlm:   # image tokens: unique ids per request -> no cross-request prefix hits, eviction restore still works
                        q["input_toks"] = x["input_toks"] + IMG_TOKS
                        q["input_tok_ids"] = [10_000_000 + i * IMG_TOKS + k for k in range(IMG_TOKS)] + list(x["input_tok_ids"])
                    w.write(json.dumps(q) + "\n")
    rows = []
    for g in (1, 2, 3, 4):
        for r in RATES:
            d = S.run(f"{tag}{g}", S.cluster(0, g, "npu"), r)
            if (d / "sim.csv").exists():
                s = S.score(d); s.update(cfg=f"{tag}{g}", gpus=g, rate=r); rows.append(s)
    with open(S.OUT / "summary.csv", "w") as f:
        w = csv.DictWriter(f, fieldnames=["cfg", "gpus", "rate", "sar", "goodput", "ttft_p90", "tpot_p90", "makespan_s"], extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    print("wrote", S.OUT / "summary.csv", flush=True)
sweep("pool_park", "sharegpt-8b-300", "colo", vlm=False)
sweep("pool_vlm_park", "vlm-ids-8b-300", "vlm_colo", vlm=True)
