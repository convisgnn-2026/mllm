#!/usr/bin/env python
"""
Score a serving run against SLO configurations, after the fact.

Runs record only timestamps; this applies deadlines to them.  One run can be
scored against any number of SLO configs, which is what makes an SLO-tightness
sweep affordable -- the same idea as TetriServe's analyze_slo_from_config.py.

An SLO config maps a template to an absolute deadline in seconds (the deadline
is measured from *arrival*, so queueing counts against it), or to a "N.Nx"
multiplier of that template's full-GPU solo latency.

  python score_slo.py results/run_*.json --slo-configs video_1.0x video_2.0x
  python score_slo.py results/run_*.json --sweep          # all configs found
"""

import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != "/" and not _os.path.isdir(_os.path.join(_d, "results")):
    _d = _os.path.dirname(_d)
_sys.path[:0] = [_os.path.join(_d, "solver"), _os.path.join(_d, "bench")]
_os.chdir(_d)            # results/ and slo_configs/ are addressed from the repo root

import argparse
import glob
import json
import os
import statistics

SLO_DIR = "slo_configs"
SOLO_REF = "results/solo_ref.json"


def load_slo(name, steps, solo_ref):
    path = name if os.path.exists(name) else os.path.join(SLO_DIR, f"{name}.json")
    raw = json.load(open(path))
    out = {}
    for tpl, v in raw.items():
        if isinstance(v, str) and v.endswith("x"):
            base = solo_ref.get(f"{tpl}@{steps}")
            if base is None:
                raise SystemExit(f"{name}: '{v}' for {tpl} needs a solo ref")
            out[tpl] = base * float(v[:-1])
        else:
            out[tpl] = float(v)
    return out


def score(run, slo):
    """Deadline is measured from arrival, so queueing delay counts."""
    rows = run["requests"]
    per_tpl, per_sm = {}, {}
    hit = 0
    for r in rows:
        d = slo.get(r["tpl"])
        if d is None:
            raise SystemExit(f"SLO config has no entry for {r['tpl']}")
        ok = (r["end_s"] - r["arrival_s"]) <= d
        hit += ok
        per_tpl.setdefault(r["tpl"], []).append(ok)
        per_sm.setdefault(r["sm"], []).append(ok)
    # Cost efficiency, in LLM-Pilot's sense: useful work per unit of provisioned
    # resource-time.  Only requests that met their SLO count as useful, so
    # over-provisioning past the SLO buys nothing -- the same reason LLM-Pilot
    # clamps effective throughput at TPS_SLO.
    gpu_h = run.get("gpu_seconds", 0) / 3600.0
    return {
        "sar": hit / len(rows),
        "ce_req_per_gpu_h": round(hit / gpu_h, 1) if gpu_h else None,
        "goodput_rps": round(hit / run["makespan_s"], 4),
        "per_template": {k: round(sum(v) / len(v), 4) for k, v in per_tpl.items()},
        "worst_template": round(min(sum(v) / len(v) for v in per_tpl.values()), 4),
        "per_slice_sm": {k: round(sum(v) / len(v), 4) for k, v in sorted(per_sm.items())},
        "n": len(rows),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run json files (globs ok)")
    ap.add_argument("--slo-configs", nargs="*", default=None)
    ap.add_argument("--sweep", action="store_true", help="use every config in slo_configs/")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--per-template", action="store_true")
    args = ap.parse_args()

    files = sorted({f for pat in args.runs for f in glob.glob(pat)})
    if not files:
        raise SystemExit("no run files matched")

    names = args.slo_configs
    if args.sweep or not names:
        names = sorted(
            (os.path.basename(p)[:-5] for p in glob.glob(f"{SLO_DIR}/*.json")),
            key=lambda s: float(s.rsplit("_", 1)[-1].rstrip("x")))

    solo_ref = json.load(open(SOLO_REF)) if os.path.exists(SOLO_REF) else {}
    out = {}

    print(f"{'policy':>14} {'workload':>26} " +
          " ".join(f"{n.rsplit('_',1)[-1]:>7}" for n in names))
    for f in files:
        run = json.load(open(f))
        wc = run["workload_config"]
        tag = f"{wc['shape_dist'][:12]}/r{wc['rate']}/cv{wc['cv']}"
        row, cells = {}, []
        for n in names:
            slo = load_slo(n, run["steps"], solo_ref)
            s = score(run, slo)
            row[n] = s
            cells.append(f"{s['sar']:>7.3f}")
        out[f] = {"policy": run["policy"], "workload": wc,
                  "n_gpus": run.get("n_gpus"), "sm_provisioned": run.get("sm_provisioned"),
                  "gpu_seconds": run.get("gpu_seconds"),
                  "utilization": run.get("utilization"),
                  "makespan_s": run["makespan_s"], "scores": row}
        print(f"{run['policy']:>14} {tag:>26} " + " ".join(cells))

        if args.per_template:
            for n in names:
                pt = row[n]["per_template"]
                print(f"    {n:>18}: " +
                      "  ".join(f"{k}={v:.2f}" for k, v in sorted(pt.items())))

    if args.out:
        json.dump(out, open(args.out, "w"), indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
