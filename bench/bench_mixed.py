#!/usr/bin/env python
"""
Mixed workload: does hard partitioning pay for itself when requests differ?

The identical-request experiment (bench_concurrency.py) shows serialization
beating space-sharing on every axis, because sharing makes all K requests
finish together while a queue staggers them.  But that is the easy case.

Real video traffic is heterogeneous by ~100x.  Under FCFS on one full GPU a
single 1280x720x81 request head-of-line blocks every 480x272 request behind
it.  Slices cannot be blocked by a neighbour.  So this script scores a MIXED
request set under *per-request relative deadlines*:

    D_i = mult x solo_i        (solo_i = that template's latency alone on a
                                full GPU, from results/solo_ref.json)

which is the standard deadline model in the DiT-serving literature: a cheap
request is promised a proportionally cheap deadline.  That is what makes
head-of-line blocking visible -- a small request stuck behind a large one
blows a deadline that is itself small.

Usage:
  python bench_mixed.py --mix 1280x720x49,480x272x33x6 --mode serial
  python bench_mixed.py --mix 1280x720x49,480x272x33x6 --mode procs
  python bench_mixed.py --mix 1280x720x49,480x272x33x6 --mode procs --devices MIG-a,MIG-b,...
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
import time

from bench_concurrency import (SOLO_REF, GpuSampler, latent_shape, load_models,
                               parse_tpl, run_request)


def expand_mix(spec):
    """'1280x720x49,480x272x33x6' -> list of (w,h,f) with the x6 repeat suffix."""
    out = []
    for item in spec.split(","):
        parts = item.lower().split("x")
        if len(parts) == 4:                    # WxHxFxREPEAT
            tpl = tuple(int(p) for p in parts[:3])
            out.extend([tpl] * int(parts[3]))
        else:
            out.append(tuple(int(p) for p in parts))
    return out


def worker(idx, tpl, steps, embed_path, t_spawn, q, device, barrier, shared):
    if device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = device
    import torch
    ctx_cpu = torch.load(embed_path, map_location="cpu")["embeds"]
    tr, vae = load_models()
    ctx = ctx_cpu.to("cuda", torch.bfloat16)
    run_request(tr, vae, tpl, 2, ctx, time.time(), -1)   # warmup
    torch.cuda.reset_peak_memory_stats()
    load_done = time.time()

    barrier.wait()
    with shared.get_lock():
        if shared.value == 0.0:
            shared.value = time.time()
    t0 = shared.value

    r = run_request(tr, vae, tpl, steps, ctx, t0, idx)
    r["tpl"] = "x".join(str(v) for v in tpl)
    r["load_s"] = round(load_done - t_spawn, 2)
    q.put(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", default="1280x720x49,480x272x33x6")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--mode", choices=["serial", "procs"], default="serial")
    ap.add_argument("--order", choices=["given", "sjf"], default="given",
                    help="serial dispatch order; 'sjf' = shortest job first")
    ap.add_argument("--devices", default=None)
    ap.add_argument("--sample-gpu", type=int, default=0)
    ap.add_argument("--embed", default="results/prompt_embed.pt")
    ap.add_argument("--out", default="results/mixed.jsonl")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    mix = expand_mix(args.mix)
    devices = args.devices.split(",") if args.devices else None
    ref = json.load(open(SOLO_REF)) if os.path.exists(SOLO_REF) else {}

    def solo_of(tpl):
        k = f"{'x'.join(str(v) for v in tpl)}@{args.steps}"
        if k not in ref:
            raise SystemExit(f"missing solo ref for {k}; run bench_concurrency.py "
                             f"--template {'x'.join(str(v) for v in tpl)} --k 1 "
                             f"--mode serial first")
        return ref[k]

    if args.mode == "serial" and args.order == "sjf":
        mix = sorted(mix, key=solo_of)

    print(f"mix ({len(mix)} requests): "
          f"{[f'{w}x{h}x{f}' for w, h, f in mix]}")
    print(f"mode={args.mode} order={args.order} "
          f"devices={'shared' if not devices else len(devices)}")

    sampler = GpuSampler(args.sample_gpu)
    sampler.start()
    results = []

    if args.mode == "procs":
        mp.set_start_method("spawn", force=True)
        q = mp.Queue()
        barrier = mp.Barrier(len(mix))
        shared = mp.Value("d", 0.0)
        t_spawn = time.time()
        ps = []
        for i, tpl in enumerate(mix):
            dev = devices[i % len(devices)] if devices else None
            p = mp.Process(target=worker,
                           args=(i, tpl, args.steps, args.embed, t_spawn, q, dev,
                                 barrier, shared))
            p.start()
            ps.append(p)
        for _ in ps:
            results.append(q.get())
        for p in ps:
            p.join()
    else:
        import torch
        ctx = torch.load(args.embed, map_location="cpu")["embeds"].to(
            "cuda", torch.bfloat16)
        tr, vae = load_models()
        run_request(tr, vae, mix[0], 2, ctx, time.time(), -1)   # warmup
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for i, tpl in enumerate(mix):
            r = run_request(tr, vae, tpl, args.steps, ctx, t0, i)
            r["tpl"] = "x".join(str(v) for v in tpl)
            results.append(r)

    sampler.stop.set()
    sampler.join(timeout=2)
    results.sort(key=lambda r: r["idx"])
    makespan = max(r["end_rel"] for r in results)
    gpu = sampler.summary(since=(sampler.ts[-1] - makespan) if sampler.ts else None)

    print(f"\n{'req':>4} {'template':>13} {'solo':>7} {'end':>8} {'slow':>7} "
          f"{'step_ms':>9} {'peak_GiB':>9}")
    for r in results:
        s = solo_of(parse_tpl(r["tpl"]))
        r["solo_s"] = s
        print(f"{r['idx']:>4} {r['tpl']:>13} {s:>7.2f} {r['end_rel']:>8.2f} "
              f"{r['end_rel']/s:>6.2f}x {r['step_ms_mean']:>9.1f} "
              f"{r['peak_gib']:>9.2f}")

    print(f"\nmakespan  {makespan:8.2f} s")
    if gpu.get("mem_max_mib"):
        print(f"mem_max   {gpu['mem_max_mib']/1024:8.1f} GiB")

    # SAR under PER-REQUEST relative deadlines D_i = mult * solo_i
    print(f"\n{'mult':>6} {'SAR_all':>8} {'SAR_small':>10} {'SAR_large':>10}")
    med = sorted(r["solo_s"] for r in results)[len(results) // 2]
    sar = {}
    for mult in [1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0]:
        ok = [r["end_rel"] <= mult * r["solo_s"] for r in results]
        sm = [o for o, r in zip(ok, results) if r["solo_s"] <= med]
        lg = [o for o, r in zip(ok, results) if r["solo_s"] > med]
        sar[mult] = sum(ok) / len(ok)
        print(f"{mult:>6.1f} {sum(ok)/len(ok):>8.2f} "
              f"{(sum(sm)/len(sm) if sm else float('nan')):>10.2f} "
              f"{(sum(lg)/len(lg) if lg else float('nan')):>10.2f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "a") as fh:
        fh.write(json.dumps({
            "tag": args.tag, "mix": args.mix, "mode": args.mode,
            "order": args.order, "steps": args.steps, "devices": devices,
            "makespan_s": round(makespan, 2), "gpu": gpu, "sar": sar,
            "requests": results,
        }) + "\n")
    print(f"\nappended -> {args.out}")


if __name__ == "__main__":
    main()
