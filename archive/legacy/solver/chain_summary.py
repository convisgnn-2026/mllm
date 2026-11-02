#!/usr/bin/env python
"""
Score every block of the experiment chain against the criteria fixed in
docs/research-flow.md §14.  No GPU.  Writes results/chain/summary.md.
"""
import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != "/" and not _os.path.isdir(_os.path.join(_d, "results")):
    _d = _os.path.dirname(_d)
_sys.path[:0] = [_os.path.join(_d, "solver"), _os.path.join(_d, "bench")]
_os.chdir(_d)

import csv
import glob
import json
import math
import statistics as st
from collections import defaultdict

U = {"480x272x33": "A", "832x480x33": "B", "832x480x49": "C", "1280x720x49": "D"}
SLO = {k: float(v) for k, v in json.load(open("slo_configs/video_3.0x.json")).items()}
CAP_2F = 0.687
OUT = []


def P(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s); print(s)


def load(tag, q, r="0.85"):
    p = f"results/fair/{tag}_{q}_r{r}.json"
    return json.load(open(p)) if _os.path.exists(p) else None


def score(d):
    per = defaultdict(list)
    for x in d["requests"]:
        per[U[x["tpl"]]].append((x["end_s"] - x["arrival_s"]) <= SLO[x["tpl"]])
    sar = {k: sum(v) / len(v) for k, v in per.items()}
    n = sum(len(v) for v in per.values())
    lat = sorted(x["end_s"] - x["arrival_s"] for x in d["requests"])
    return dict(agg=sum(sum(v) for v in per.values()) / n, worst=min(sar.values()),
                sar=sar, p95=lat[int(len(lat) * .95) - 1], mk=d["makespan_s"],
                util=d.get("utilization", float("nan")), n=n)


def paired_D(a, b):
    A = {x["id"]: x for x in a["requests"] if x["tpl"] == "1280x720x49"}
    B = {x["id"]: x for x in b["requests"] if x["tpl"] == "1280x720x49"}
    hit = lambda x: (x["end_s"] - x["arrival_s"]) <= SLO[x["tpl"]]
    worse = sum(1 for i in A if i in B and hit(A[i]) and not hit(B[i]))
    better = sum(1 for i in A if i in B and hit(B[i]) and not hit(A[i]))
    return worse, better


def row(label, s):
    return (f"| {label} | {s['agg']:.2f} | {s['worst']:.2f} | "
            + " | ".join(f"{s['sar'].get(k, float('nan')):.2f}" for k in "ABCD")
            + f" | {s['p95']:.1f}s | {s['util']:.2f} |")


HDR = "| config | agg | worst | A | B | C | D | p95 | util |\n|---|---|---|---|---|---|---|---|---|"

# ============================================================ A
P("# Chain summary — 2026-09-03\n")
P("## A — Full ×3 vs 2-way ×2: partition vs buying  (fair trace, 124% of 2F, 3.0x)\n")
P(HDR)
for q in ["fcfs", "sjf", "edf", "vtc"]:
    for tag in ["2F", "2x2way", "3F"]:
        d = load(tag, q)
        if d: P(row(f"{tag} {q}", score(d)))
P()
P("verdict (criterion: 2x2way D ≥ 0.7 × 3F D):")
for q in ["sjf", "edf", "vtc"]:
    a, b = load("2x2way", q), load("3F", q)
    if a and b:
        da, db = score(a)["sar"]["D"], score(b)["sar"]["D"]
        P(f"  {q}: 2x2way D {da:.2f} vs 3F D {db:.2f} → ratio {da/db if db else float('nan'):.2f}"
          f"  {'PASS' if db and da >= 0.7 * db else 'FAIL'}   (3F costs 1.5× the GPUs)")

# ============================================================ U2
P("\n## U2 — MIG reconfiguration cost (5 reps, seconds, median)\n")
rows = list(csv.DictReader(open("results/mig_reconfig.csv")))
for op in ["mig_off", "mig_on_2x3g", "worker_cold_start_1req"]:
    v = sorted(float(r["seconds"]) for r in rows if r["op"] == op)
    P(f"  {op:<26} median {v[len(v)//2]:6.1f}   min {v[0]:6.1f}   max {v[-1]:6.1f}")
P("  → reconfigure ≈ 3 s; the cost is the worker cold start (weights + first step) ≈ 33 s.")

# ============================================================ U1
P("\n## U1 — heterogeneous layouts on the fair trace (same 2 GPUs)\n")
P(HDR)
for q in ["sjf", "edf"]:
    for tag in ["2F", "2x2way", "1F1x2", "2x3g1g4", "2x4g1g3"]:
        d = load(tag, q)
        if d: P(row(f"{tag} {q}", score(d)))
P()
P("verdict (criterion: some heterogeneous layout beats 2x2way on worst without losing agg):")
for q in ["sjf", "edf"]:
    base = score(load("2x2way", q))
    for tag in ["1F1x2", "2x3g1g4", "2x4g1g3"]:
        d = load(tag, q)
        if not d: continue
        s = score(d)
        P(f"  {q} {tag:<9} worst {s['worst']:.2f} vs 2x2way {base['worst']:.2f}  "
          f"agg {s['agg']:.2f} vs {base['agg']:.2f}  "
          f"{'BEATS' if s['worst'] > base['worst'] + 0.04 and s['agg'] >= base['agg'] - 0.02 else 'no'}")

# ============================================================ B
P("\n## B — mix sweep: D share of requests 5 / 11 / 22 / 37 %  (124%, 3.0x)\n")
P("| D share | queue | 2F agg/D | 2x2way agg/D | Δ agg | Δ D | D worse/better (paired) |")
P("|---|---|---|---|---|---|---|")
for lab, tagF, tagP in [("5%", "2F_d05", "2x2way_d05"), ("11%", "2F", "2x2way"),
                        ("22%", "2F_d22", "2x2way_d22"), ("37%", "2F_d37", "2x2way_d37")]:
    for q in ["sjf", "edf"]:
        a, b = load(tagF, q), load(tagP, q)
        if not (a and b): continue
        A, B = score(a), score(b)
        w, bt = paired_D(a, b)
        P(f"| {lab} | {q} | {A['agg']:.2f} / {A['sar']['D']:.2f} | {B['agg']:.2f} / {B['sar']['D']:.2f} | "
          f"{B['agg']-A['agg']:+.2f} | {B['sar']['D']-A['sar']['D']:+.2f} | {w} / {bt} |")

# ============================================================ C
P("\n## C — load sweep: where is the knee  (fair mix, 3.0x)\n")
P("| load | queue | 2F p95 | 2x2way p95 | Δ p95 | 2F util | 2F D | 2x2way D | Δ D |")
P("|---|---|---|---|---|---|---|---|---|")
for r in ["0.55", "0.60", "0.65", "0.70", "0.75", "0.85", "1.10"]:
    for q in ["sjf", "edf"]:
        a, b = load("2F", q, r), load("2x2way", q, r)
        if not (a and b): continue
        A, B = score(a), score(b)
        P(f"| {float(r)/CAP_2F*100:.0f}% | {q} | {A['p95']:.1f} | {B['p95']:.1f} | "
          f"{(B['p95']-A['p95'])/A['p95']*100:+.0f}% | {A['util']:.2f} | {A['sar']['D']:.2f} | {B['sar']['D']:.2f} | {B['sar']['D']-A['sar']['D']:+.2f} |")

# ============================================================ E
P("\n## E — seeds 7 (orig), 8, 9, 10 at 124%  (D worst-class, paired)\n")
P("| seed | queue | 2F D | 2x2way D | Δ | worse / better |")
P("|---|---|---|---|---|---|")
pool = defaultdict(lambda: [0, 0])
for s in ["7", "8", "9", "10"]:
    for q in ["sjf", "edf"]:
        tF, tP = ("2F", "2x2way") if s == "7" else (f"2F_s{s}", f"2x2way_s{s}")
        a, b = load(tF, q), load(tP, q)
        if not (a and b): continue
        A, B = score(a), score(b)
        w, bt = paired_D(a, b); pool[q][0] += w; pool[q][1] += bt
        P(f"| {s} | {q} | {A['sar']['D']:.2f} | {B['sar']['D']:.2f} | {B['sar']['D']-A['sar']['D']:+.2f} | {w} / {bt} |")
P()
for q, (w, bt) in pool.items():
    n = w + bt
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(w, bt) + 1)) / 2 ** n) if n else 1.0
    P(f"  pooled {q}: D requests worse {w}, better {bt} over 4 seeds → sign test p = {p:.4f}")

# ============================================================ U5
P("\n## U5 — production-shaped skew: A 65% B 20% C 10% D 5%  (124%, 3.0x)\n")
P(HDR)
for q in ["sjf", "edf"]:
    for tag in ["2F_skewreal", "2x2way_skewreal"]:
        d = load(tag, q)
        if d: P(row(f"{tag} {q}", score(d)))

# ============================================================ U6
P("\n## U6 — 3-GPU scale: Full ×3 vs 2-way ×3  (fair trace)\n")
P(HDR)
for q in ["sjf", "edf"]:
    for tag in ["3F", "3x2way"]:
        d = load(tag, q)
        if d: P(row(f"{tag} {q}", score(d)))
P()
for q in ["sjf", "edf"]:
    a2, b2, a3, b3 = load("2F", q), load("2x2way", q), load("3F", q), load("3x2way", q)
    if all([a2, b2, a3, b3]):
        g2 = score(b2)["sar"]["D"] - score(a2)["sar"]["D"]
        g3 = score(b3)["sar"]["D"] - score(a3)["sar"]["D"]
        P(f"  {q}: partition gain on D   2 GPUs {g2:+.2f}   3 GPUs {g3:+.2f}   "
          f"(note: 3 GPUs = 83% load, not 124% — different regime)")

# ============================================================ F
P("\n## F — torch.compile vs eager, 148 SM  (DiT step ms; overhead = 1 − compile/eager)\n")
eager = {(r["width"], r["height"], r["frames"]): r for r in csv.DictReader(open("results/stages.csv")) if r["tag"] == "full"}
comp = {(r["width"], r["height"], r["frames"]): r for r in csv.DictReader(open("results/stages_compile.csv"))}
P("| template | eager ms | compile ms | overhead |")
P("|---|---|---|---|")
for k in sorted(eager, key=lambda k: int(k[0]) * int(k[1]) * int(k[2])):
    if k in comp and comp[k]["status"] == "ok":
        e, c = float(eager[k]["step_ms_mean"]), float(comp[k]["step_ms_mean"])
        P(f"| {'x'.join(k)} | {e:.1f} | {c:.1f} | {(1-c/e)*100:+.1f}% |")

# ============================================================ G
P("\n## G — Wan2.1-14B vs 1.3B, 148 SM, 20 steps\n")
b14 = {(r["width"], r["height"], r["frames"]): r for r in csv.DictReader(open("results/stages_14b.csv"))}
P("| template | 1.3B step ms | 14B step ms | ×slower | 14B DiT peak | 14B VAE peak | fits 1g.34gb? | fits 1g.67gb? |")
P("|---|---|---|---|---|---|---|---|")
W14 = None
for k in sorted(eager, key=lambda k: int(k[0]) * int(k[1]) * int(k[2])):
    r = b14.get(k)
    if not r or r["status"] != "ok":
        if r: P(f"| {'x'.join(k)} | | | | | | {r['status']} | |")
        continue
    W14 = W14 or float(r["dit_w_gib"])
    e, c = float(eager[k]["step_ms_mean"]), float(r["step_ms_mean"])
    dp, vp = float(r["dit_peak_gib"]), float(r["vae_peak_gib"])
    need = max(dp, vp) + 0.47              # + VAE weights; T5 assumed disaggregated
    P(f"| {'x'.join(k)} | {e:.0f} | {c:.0f} | {c/e:.1f}× | {dp:.1f} | {vp:.1f} | "
      f"{'yes' if need <= 30.5 else 'NO'} | {'yes' if need <= 66.5 else 'NO'} |")
P(f"\n  14B DiT weights {W14:.1f} GiB (bf16).  'fits' = max(DiT peak, VAE peak) + VAE weights ≤ slice memory, T5 elsewhere.")

open("results/chain/summary.md", "w").write("\n".join(OUT) + "\n")
print("\nwrote results/chain/summary.md")
