#!/usr/bin/env python
"""VLM tenant, LLM side: same ShareGPT trace with +IMG_TOKS prompt tokens per request
(LLaVA-1.5 @336px = 576 image tokens), language backbone = Llama-3.1-8B profile (proxy for a 7B VLM).
The vision encoder is a fixed per-image cost added analytically in solver/cosolve.py.
Output results/servingsim/pool_vlm/summary.csv"""
import csv, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)); import sweep_pd_kvo as S
IMG_TOKS = 576
S.OUT = Path(__file__).resolve().parents[1] / "results/servingsim/pool_vlm"; S.OUT.mkdir(parents=True, exist_ok=True)
S.WL_PREFIX = "vlm-8b-300"
RATES = [1, 2, 3, 4, 6, 8]
src = S.WL / "sharegpt-8b-300-sps10.jsonl"; base = [json.loads(l) for l in open(src)]; t0 = min(x["arrival_time_ns"] for x in base)
for r in RATES:
    wl = S.WL / f"{S.WL_PREFIX}-sps{r}.jsonl"
    if not wl.exists():
        with open(wl, "w") as w:
            for x in base:
                q = dict(x); q["arrival_time_ns"] = int((x["arrival_time_ns"] - t0) * 10 / r)
                q["input_toks"] = x["input_toks"] + IMG_TOKS
                q.pop("input_tok_ids", None); q.pop("output_tok_ids", None)   # no prefix caching for image tokens
                w.write(json.dumps(q) + "\n")
rows = []
for g in (1, 2, 3, 4):
    for r in RATES:
        d = S.run(f"vlm_colo{g}", S.cluster(0, g, "npu"), r)
        if (d / "sim.csv").exists():
            s = S.score(d); s.update(cfg=f"vlm_colo{g}", gpus=g, rate=r); rows.append(s)
keys = ["cfg", "gpus", "rate", "sar", "goodput", "ttft_p90", "tpot_p90", "makespan_s"]
with open(S.OUT / "summary.csv", "w") as f:
    w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
print("wrote", S.OUT / "summary.csv")
