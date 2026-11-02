#!/usr/bin/env python
"""
Serving benchmark: replay a workload against a fixed-granularity policy.

Following TetriServe's benchmark design, this records only *timestamps*.
SLO attainment is computed afterwards by score_slo.py, so one (expensive) run
can be scored against any number of SLO configurations.

Architecture
    main process   = scheduler.  Holds the arrival schedule, the pending queue,
                     and worker idle/busy state.  All policy lives here.
    worker process = executor.   Pinned to one device (a full GPU or one MIG
                     slice), loads the model once, then serves tasks.

Policies (all are FIXED-granularity baselines, in TetriServe's sense):
    full_fcfs        one full GPU, first-come-first-served
    full_srpt        one full GPU, shortest-predicted-remaining first
    mig_uniform      N identical slices, FCFS onto any free slice
    mig_mixed        heterogeneous slices; each template has a preferred slice
                     class, falling back to any free slice that fits

Usage:
  python serve_bench.py --workload wl.json --policy full_fcfs --devices 1
  python serve_bench.py --workload wl.json --policy mig_uniform --devices MIG-a,MIG-b,...
"""

import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != "/" and not _os.path.isdir(_os.path.join(_d, "results")):
    _d = _os.path.dirname(_d)
_sys.path[:0] = [_os.path.join(_d, "solver"), _os.path.join(_d, "bench")]
_os.chdir(_d)            # results/ and slo_configs/ are addressed from the repo root

import argparse
import json
import multiprocessing as mp
import os
import queue
import time

import policies as POL

MODEL = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
STEPS = 20


def parse_tpl(s):
    return tuple(int(x) for x in s.lower().split("x"))


def latent_shape(w, h, f, zc=16):
    return zc, (f - 1) // 4 + 1, h // 8, w // 8


# --------------------------------------------------------------------------
def worker(wid, device, task_q, res_q, ready_q, embed_path, steps):
    if device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    import torch
    from diffusers import AutoencoderKLWan, WanTransformer3DModel

    ctx = torch.load(embed_path, map_location="cpu")["embeds"].to(
        "cuda", torch.bfloat16)
    tr = WanTransformer3DModel.from_pretrained(
        MODEL, subfolder="transformer", dtype=torch.bfloat16).to("cuda").eval()
    vae = AutoencoderKLWan.from_pretrained(
        MODEL, subfolder="vae", dtype=torch.float32).to("cuda").eval()

    def run(tpl):
        zc, lf, lh, lw = latent_shape(*tpl)
        lat = torch.randn(1, zc, lf, lh, lw, device="cuda", dtype=torch.bfloat16)
        ts = torch.tensor([500.0], device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            for _ in range(steps):
                tr(hidden_states=lat, timestep=ts, encoder_hidden_states=ctx,
                   return_dict=False)
            vae.decode(lat.float(), return_dict=False)
        torch.cuda.synchronize()
        del lat
        torch.cuda.empty_cache()

    run((480, 272, 33))                       # warm the allocator and kernels
    p = torch.cuda.get_device_properties(0)
    # Report the device we actually landed on.  A numeric CUDA_VISIBLE_DEVICES
    # silently shifts when some GPU on the box is in MIG mode, so the caller
    # must be able to check that it got the GPU it asked for.
    try:
        import pynvml  # noqa
        got = None
    except Exception:
        got = None
    got = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    ready_q.put({"wid": wid, "sm": p.multi_processor_count,
                 "mem_gib": round(p.total_memory / 2**30, 1),
                 "requested": got, "name": p.name})

    while True:
        task = task_q.get()
        if task is None:
            break
        start = time.time()
        torch.cuda.reset_peak_memory_stats()
        run(parse_tpl(task["tpl"]))
        res_q.put({**task, "wid": wid, "start_abs": start, "end_abs": time.time(),
                   "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)})


# --------------------------------------------------------------------------
def pick_pending(pending, queue_disc, cost):
    """Index into `pending`, by queue discipline."""
    if queue_disc == "srpt":
        return min(range(len(pending)), key=lambda i: cost[pending[i]["tpl"]])
    return 0                                   # fcfs


def pick_worker(req, idle, wmeta, select, cost, big_tpls):
    """Worker id from `idle`.

    any    -- first free worker (correct when the pool is homogeneous)
    match  -- size matching: expensive requests prefer high-SM workers, cheap
              ones prefer low-SM workers, so a big request is not stranded on a
              small slice and a small request does not occupy a full GPU.
              Falls back to whatever is free rather than idling the pool.
    """
    if select == "any":
        return min(idle)
    want_big = req["tpl"] in big_tpls
    return sorted(idle, key=lambda w: wmeta[w]["sm"], reverse=want_big)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", required=True)
    ap.add_argument("--policy", required=True, help="label for this configuration")
    ap.add_argument("--queue", default="sjf", choices=sorted(POL.REGISTRY))
    ap.add_argument("--slo-config", default=None,
                    help="slo_configs name; required by --queue edf")
    ap.add_argument("--select", default="match", choices=["any", "match"])
    ap.add_argument("--devices", required=True,
                    help="comma-separated CUDA_VISIBLE_DEVICES (indices or MIG uuids)")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--embed", default="results/prompt_embed.pt")
    ap.add_argument("--cost-ref", default="results/solo_ref.json",
                    help="template -> full-GPU solo latency, used by SRPT and "
                         "reported for reference; never used for scoring")
    ap.add_argument("--max-requests", type=int, default=None)
    ap.add_argument("--n-gpus", type=int, default=None,
                    help="physical GPUs provisioned (for cost); inferred if omitted")
    ap.add_argument("--big-templates", default="1280x720x49",
                    help="templates that prefer the large slice under mig_mixed")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    wl = json.load(open(args.workload))
    reqs = wl["requests"][: args.max_requests]
    devices = args.devices.split(",")
    big_tpls = set(args.big_templates.split(","))

    ref = json.load(open(args.cost_ref)) if os.path.exists(args.cost_ref) else {}
    cost = {r["tpl"]: ref.get(f"{r['tpl']}@{args.steps}", r["seq"] / 1e4)
            for r in reqs}

    slo = {}
    if args.slo_config:
        import json as _j
        slo = {k: float(v) for k, v in
               _j.load(open(f"slo_configs/{args.slo_config}.json")).items()}
    pol = POL.make(args.queue, cost, slo)

    print(f"policy={args.policy} queue={args.queue} select={args.select} "
          f"workers={len(devices)} requests={len(reqs)} "
          f"span={reqs[-1]['arrival_s']:.0f}s")

    mp.set_start_method("spawn", force=True)
    res_q, ready_q = mp.Queue(), mp.Queue()
    task_qs = [mp.Queue() for _ in devices]
    procs = []
    for i, d in enumerate(devices):
        p = mp.Process(target=worker,
                       args=(i, d, task_qs[i], res_q, ready_q, args.embed,
                             args.steps))
        p.start()
        procs.append(p)

    wmeta = {}
    for _ in devices:
        m = ready_q.get()
        wmeta[m["wid"]] = m
    print("workers ready: " +
          ", ".join(f"w{w}({wmeta[w]['sm']}SM/{wmeta[w]['mem_gib']}G)"
                    for w in sorted(wmeta)))
    for w in sorted(wmeta):
        req = wmeta[w]["requested"]
        if req.startswith("GPU-") or req.startswith("MIG-"):
            continue                            # uuid form: unambiguous
        print(f"  [warn] worker {w} was given a numeric device id ({req}); "
              f"numeric ids shift when any GPU is in MIG mode")

    # ---- scheduling loop -------------------------------------------------
    t0 = time.time()
    idle = set(range(len(devices)))
    pending, done, nxt = [], [], 0

    while len(done) < len(reqs):
        now = time.time() - t0
        while nxt < len(reqs) and reqs[nxt]["arrival_s"] <= now:
            r_ = {**reqs[nxt], "arrival_abs": t0 + reqs[nxt]["arrival_s"]}
            pol.on_arrival(r_)
            pending.append(r_)
            nxt += 1
        if hasattr(pol, "note_pending"):
            pol.note_pending(pending)
        while idle and pending:
            i = pol.pick(pending, now)
            req = pending.pop(i)
            pol.on_dispatch(req)
            w = pick_worker(req, idle, wmeta, args.select, cost, big_tpls)
            idle.discard(w)
            task_qs[w].put({**req, "dispatch_abs": time.time()})
        try:
            r = res_q.get(timeout=0.05)
            idle.add(r["wid"])
            done.append(r)
            if len(done) % 20 == 0:
                print(f"  {len(done)}/{len(reqs)} done, "
                      f"t={time.time()-t0:.0f}s, queued={len(pending)}")
        except queue.Empty:
            pass

    makespan = max(r["end_abs"] for r in done) - t0
    for q_ in task_qs:
        q_.put(None)
    for p in procs:
        p.join(timeout=30)

    rows = [{"id": r["id"], "tpl": r["tpl"], "wid": r["wid"],
             "sm": wmeta[r["wid"]]["sm"],
             "arrival_s": round(r["arrival_abs"] - t0, 3),
             "dispatch_s": round(r["dispatch_abs"] - t0, 3),
             "start_s": round(r["start_abs"] - t0, 3),
             "end_s": round(r["end_abs"] - t0, 3),
             "service_s": round(r["end_abs"] - r["start_abs"], 3),
             "queue_s": round(r["start_abs"] - r["arrival_abs"], 3),
             "peak_gib": r["peak_gib"]}
            for r in sorted(done, key=lambda x: x["id"])]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    # Cost is *provisioned* resource-time: you hold the GPU whether it is busy
    # or not, exactly as an hourly VM is billed.  n_gpus counts distinct
    # physical GPUs, so slicing one GPU into 7 workers still costs one GPU.
    n_gpus = args.n_gpus if args.n_gpus else len({d.split("/")[0] for d in devices})
    sm_prov = sum(wmeta[w]["sm"] for w in wmeta)
    busy = sum(r["service_s"] for r in rows)
    json.dump({"policy": args.policy, "queue": args.queue, "select": args.select,
               "slo_config": args.slo_config,
               "devices": devices, "workers": wmeta, "steps": args.steps,
               "n_gpus": n_gpus, "sm_provisioned": sm_prov,
               "gpu_seconds": round(n_gpus * makespan, 1),
               "sm_seconds": round(sm_prov * makespan, 1),
               "busy_seconds": round(busy, 1),
               "utilization": round(busy / (len(devices) * makespan), 3),
               "workload_config": wl["config"], "makespan_s": round(makespan, 2),
               "requests": rows}, open(args.out, "w"), indent=1)

    lat = sorted(r["end_s"] - r["arrival_s"] for r in rows)
    print(f"cost: {n_gpus} GPU x {makespan:.0f}s = {n_gpus*makespan:.0f} GPU-s | "
          f"util {busy/(len(devices)*makespan)*100:.0f}%")
    print(f"makespan {makespan:.1f}s | latency mean {sum(lat)/len(lat):.1f}s "
          f"p50 {lat[len(lat)//2]:.1f}s p95 {lat[int(len(lat)*.95)-1]:.1f}s "
          f"max {lat[-1]:.1f}s")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
