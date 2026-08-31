#!/usr/bin/env python
"""
Serialization vs space-sharing on one GPU (or one MIG slice set).

Question:
  Does running K requests side by side raise GPU utilization but LOWER
  deadline attainment (SAR) compared with running them one at a time?

Modes
  serial : K requests back to back in ONE process on ONE device
           -> a queue on a full GPU
  procs  : K requests concurrently, each in its own process
           -> on a full GPU  : condition (c), shared memory pool, time-sliced SMs
           -> with --devices  : condition (b), one MIG slice per request
              (each process gets its own weights and its own context, which is
               exactly what a MIG slice forces)

SAR is computed against a *solo* reference latency (one request alone on a full
GPU), cached in results/solo_ref.json, so all modes are scored on the same
absolute deadlines.

Usage:
  python bench_concurrency.py --template 832x480x49 --k 1 --mode serial   # sets solo ref
  python bench_concurrency.py --template 832x480x49 --k 7 --mode serial
  python bench_concurrency.py --template 832x480x49 --k 7 --mode procs
  python bench_concurrency.py --template 832x480x49 --k 7 --mode procs \
         --devices MIG-uuid1,MIG-uuid2,...
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
import statistics
import subprocess
import threading
import time

MODEL = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
SOLO_REF = "results/solo_ref.json"


def parse_tpl(s):
    return tuple(int(x) for x in s.lower().split("x"))


def latent_shape(w, h, f, zc=16):
    return zc, (f - 1) // 4 + 1, h // 8, w // 8


# --------------------------------------------------------------------------
class GpuSampler(threading.Thread):
    """Poll nvidia-smi so we can report utilization, not just throughput."""

    def __init__(self, gpu_index=0, period=0.25):
        super().__init__(daemon=True)
        self.idx, self.period, self.stop = gpu_index, period, threading.Event()
        self.util, self.mem, self.ts = [], [], []

    def run(self):
        q = ("--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits", f"--id={self.idx}")
        while not self.stop.is_set():
            try:
                o = subprocess.run(("nvidia-smi",) + q, capture_output=True,
                                   text=True, timeout=5).stdout.strip()
                u, m = (x.strip() for x in o.split(","))
                # a MIG-enabled GPU reports utilization.gpu as "[N/A]"
                uv = float(u) if u not in ("[N/A]", "N/A", "") else -1.0
                mv = float(m) if m not in ("[N/A]", "N/A", "") else -1.0
                self.ts.append(time.time())      # append together, or the
                self.util.append(uv)             # three lists desynchronise
                self.mem.append(mv)
            except Exception:
                pass
            self.stop.wait(self.period)

    def summary(self, since=None):
        """Restrict to samples after `since` so load time does not dilute util."""
        idx = [i for i, t in enumerate(self.ts) if since is None or t >= since]
        if not idx:
            return {}
        u = [self.util[i] for i in idx if self.util[i] >= 0]
        m = [self.mem[i] for i in idx if self.mem[i] >= 0]
        out = {"samples": len(idx)}
        if u:                      # absent under MIG
            out.update(util_mean=round(statistics.mean(u), 1),
                       util_max=max(u), util_min=min(u))
        if m:
            out.update(mem_max_mib=max(m))
        return out


# --------------------------------------------------------------------------
def run_request(tr, vae, tpl, steps, ctx, t0, idx):
    import torch
    dev = torch.device("cuda")
    zc, lf, lh, lw = latent_shape(*tpl)
    lat = torch.randn(1, zc, lf, lh, lw, device=dev, dtype=torch.bfloat16)
    ts = torch.tensor([500.0], device=dev, dtype=torch.bfloat16)

    start = time.time()
    st = []
    with torch.no_grad():
        for _ in range(steps):
            s = time.perf_counter()
            tr(hidden_states=lat, timestep=ts, encoder_hidden_states=ctx,
               return_dict=False)
            torch.cuda.synchronize()
            st.append((time.perf_counter() - s) * 1e3)
        vs = time.perf_counter()
        vae.decode(lat.float(), return_dict=False)
        torch.cuda.synchronize()
        vae_ms = (time.perf_counter() - vs) * 1e3
    end = time.time()
    del lat
    torch.cuda.empty_cache()

    st_sorted = sorted(st)
    return {"idx": idx, "start_rel": start - t0, "end_rel": end - t0,
            "dur_s": end - start,
            "step_ms_mean": round(statistics.mean(st), 2),
            "step_ms_p95": round(st_sorted[max(0, int(len(st) * .95) - 1)], 2),
            "vae_ms": round(vae_ms, 1),
            "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}


def load_models():
    import torch
    from diffusers import AutoencoderKLWan, WanTransformer3DModel
    tr = WanTransformer3DModel.from_pretrained(
        MODEL, subfolder="transformer", dtype=torch.bfloat16).to("cuda").eval()
    vae = AutoencoderKLWan.from_pretrained(
        MODEL, subfolder="vae", dtype=torch.float32).to("cuda").eval()
    return tr, vae


def worker(idx, tpl, steps, embed_path, t_spawn, q, device, barrier, shared):
    """Own process, own weights, own CUDA context -- the MIG stand-in.

    All workers rendezvous at `barrier` after loading weights, so the reported
    times isolate the *scheduling* effect from the (real, but separate) cost of
    duplicating weights into every slice.  `load_s` keeps the latter visible.
    """
    if device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = device
    import torch
    ctx_cpu = torch.load(embed_path, map_location="cpu")["embeds"]
    tr, vae = load_models()
    ctx = ctx_cpu.to("cuda", torch.bfloat16)
    run_request(tr, vae, tpl, 2, ctx, time.time(), -1)  # warmup, pre-barrier
    torch.cuda.reset_peak_memory_stats()
    load_done = time.time()

    barrier.wait()
    with shared.get_lock():
        if shared.value == 0.0:
            shared.value = time.time()
    t0 = shared.value

    r = run_request(tr, vae, tpl, steps, ctx, t0, idx)
    r["load_s"] = round(load_done - t_spawn, 2)
    q.put(r)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", default="832x480x49")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--mode", choices=["serial", "procs"], default="serial")
    ap.add_argument("--devices", default=None,
                    help="comma-separated CUDA_VISIBLE_DEVICES per worker "
                         "(e.g. MIG uuids). Default: all workers share device 0")
    ap.add_argument("--sample-gpu", type=int, default=0,
                    help="physical GPU index to poll for utilization")
    ap.add_argument("--embed", default="results/prompt_embed.pt")
    ap.add_argument("--out", default="results/concurrency.jsonl")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    tpl = parse_tpl(args.template)
    zc, lf, lh, lw = latent_shape(*tpl)
    devices = args.devices.split(",") if args.devices else None
    print(f"template {args.template} latent {zc}x{lf}x{lh}x{lw} | K={args.k} "
          f"steps={args.steps} mode={args.mode} "
          f"devices={'shared' if not devices else len(devices)}")

    sampler = GpuSampler(args.sample_gpu)
    sampler.start()
    results = []

    if args.mode == "procs":
        mp.set_start_method("spawn", force=True)
        q = mp.Queue()
        barrier = mp.Barrier(args.k)
        shared = mp.Value("d", 0.0)
        t_spawn = time.time()
        ps = []
        for i in range(args.k):
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
        torch.cuda.reset_peak_memory_stats()
        run_request(tr, vae, tpl, 2, ctx, time.time(), -1)  # warmup
        t0 = time.time()
        for i in range(args.k):
            results.append(run_request(tr, vae, tpl, args.steps, ctx, t0, i))

    sampler.stop.set()
    sampler.join(timeout=2)
    results.sort(key=lambda r: r["idx"])
    makespan = max(r["end_rel"] for r in results)
    # wall-clock instant at which the measured window opened
    win_start = min(r["end_rel"] - r["dur_s"] for r in results)
    gpu = sampler.summary(since=(sampler.ts[-1] - makespan) if sampler.ts else None)

    # --- solo reference (one request alone on a full GPU) ---
    ref = {}
    if os.path.exists(SOLO_REF):
        ref = json.load(open(SOLO_REF))
    key = f"{args.template}@{args.steps}"
    if args.mode == "serial" and args.k == 1 and not devices:
        ref[key] = results[0]["dur_s"]
        os.makedirs(os.path.dirname(SOLO_REF) or ".", exist_ok=True)
        json.dump(ref, open(SOLO_REF, "w"), indent=1)
        print(f"[solo ref] {key} = {ref[key]:.2f} s  (written)")
    solo = ref.get(key)
    if solo is None:
        solo = min(r["dur_s"] for r in results)
        print(f"[warn] no solo ref for {key}; using min(dur)={solo:.2f}s")

    print(f"\n{'req':>4} {'load':>6} {'start':>7} {'end':>8} {'dur':>7} {'slow':>6} "
          f"{'step_ms':>9} {'vae_ms':>8} {'peak_GiB':>9}")
    for r in results:
        print(f"{r['idx']:>4} {r.get('load_s',0):>6.1f} {r['start_rel']:>7.2f} "
              f"{r['end_rel']:>8.2f} {r['dur_s']:>7.2f} {r['dur_s']/solo:>6.2f}x "
              f"{r['step_ms_mean']:>9.1f} {r['vae_ms']:>8.1f} {r['peak_gib']:>9.2f}")

    print(f"\nsolo ref       {solo:8.2f} s")
    print(f"makespan       {makespan:8.2f} s")
    print(f"throughput     {args.k/makespan:8.3f} req/s   "
          f"(serial ideal {1/solo:.3f})")
    if gpu:
        if "util_mean" in gpu:
            print(f"GPU util       mean {gpu['util_mean']:.1f}%  "
                  f"[{gpu['util_min']:.0f}-{gpu['util_max']:.0f}%]  n={gpu['samples']}")
        if "mem_max_mib" in gpu:
            print(f"mem_max        {gpu['mem_max_mib']/1024:8.1f} GiB")

    print(f"\n{'D/solo':>7} {'D(s)':>8} {'SAR':>6}")
    sar = {}
    for mult in [1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]:
        D = solo * mult
        hit = sum(1 for r in results if r["end_rel"] <= D)
        sar[mult] = hit / len(results)
        print(f"{mult:>7.2f} {D:>8.2f} {hit/len(results):>6.2f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "a") as fh:
        fh.write(json.dumps({
            "tag": args.tag, "template": args.template, "k": args.k,
            "steps": args.steps, "mode": args.mode,
            "devices": devices, "makespan_s": round(makespan, 2),
            "solo_s": round(solo, 2), "throughput_rps": args.k / makespan,
            "gpu": gpu, "sar": sar, "requests": results,
        }) + "\n")
    print(f"\nappended -> {args.out}")


if __name__ == "__main__":
    main()
