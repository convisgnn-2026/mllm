#!/usr/bin/env python
"""
Cost of moving a latent off the GPU and back -- the feasibility gate for
dispatching a request's denoising steps across different workers.

Why it has to go through host memory: every MIG profile reports P2P = No, and
NCCL does not support MIG, so two slices cannot hand a tensor to each other
directly.  The path is GPU -> host -> GPU, and if that costs a large share of a
step then step-level dispatch is not worth doing.

Measured separately, because the numbers differ by an order of magnitude:
  pageable host memory   what you get from a plain .cpu()
  pinned host memory     page-locked, allows DMA; what an implementation would use
  two-process handoff    the realistic path -- one worker writes, another reads,
                         through a shared pinned buffer

Timed with CUDA events around a synchronised region, warmed up first, and
reported against the measured per-step latency of each template so the overhead
is expressed as the share of a step it would actually consume.
"""
import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != "/" and not _os.path.isdir(_os.path.join(_d, "results")):
    _d = _os.path.dirname(_d)
_sys.path[:0] = [_os.path.join(_d, "solver"), _os.path.join(_d, "bench")]
_os.chdir(_d)

import argparse
import statistics as st
import time

import torch
import torch.multiprocessing as mp

# (template, latent shape, measured per-step latency on a full GPU, on a 70-SM slice)
CASES = [
    ("480x272x33", (1, 16, 9, 34, 60), 0.0304, 0.0481),
    ("832x480x33", (1, 16, 9, 60, 104), 0.0994, 0.1769),
    ("832x480x49", (1, 16, 13, 60, 104), 0.1554, 0.2747),
    ("1280x720x49", (1, 16, 13, 90, 160), 0.5020, 0.8573),
    ("1920x1088x81", (1, 16, 21, 136, 240), 4.4523, 7.6000),
]
DTYPE = torch.bfloat16


def timed(fn, warmup=5, iters=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(True)
        e = torch.cuda.Event(True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) / 1e3)
    return st.median(ts)


def bench_single(shape):
    """D2H and H2D on one device, pageable vs pinned."""
    g = torch.randn(*shape, device="cuda", dtype=DTYPE)
    nbytes = g.numel() * g.element_size()
    out = {"MiB": nbytes / 2**20}

    pageable = torch.empty(shape, dtype=DTYPE)                 # plain host tensor
    pinned = torch.empty(shape, dtype=DTYPE).pin_memory()

    out["d2h_pageable"] = timed(lambda: pageable.copy_(g, non_blocking=False))
    out["d2h_pinned"] = timed(lambda: pinned.copy_(g, non_blocking=False))
    out["h2d_pageable"] = timed(lambda: g.copy_(pageable, non_blocking=False))
    out["h2d_pinned"] = timed(lambda: g.copy_(pinned, non_blocking=False))
    out["rt_pinned"] = out["d2h_pinned"] + out["h2d_pinned"]
    for k in ("d2h_pinned", "h2d_pinned"):
        out[k.replace("_pinned", "_GBps")] = nbytes / out[k] / 1e9
    del g, pageable, pinned
    torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------- two processes
def _writer(dev, shape, shared, ready, done, iters):
    if dev:
        os_env(dev)
    import torch
    g = torch.randn(*shape, device="cuda", dtype=DTYPE)
    for _ in range(iters):
        ready.wait()
        ready.clear()
        shared.copy_(g, non_blocking=False)
        torch.cuda.synchronize()
        done.set()


def _reader(dev, shape, shared, ready, done, iters, out):
    if dev:
        os_env(dev)
    import torch
    g = torch.empty(shape, device="cuda", dtype=DTYPE)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        ready.set()
        done.wait()
        done.clear()
        g.copy_(shared, non_blocking=False)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    out.put(st.median(ts[len(ts) // 3:]))          # drop warm-up third


def os_env(dev):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = dev


def bench_handoff(shape, dev_a, dev_b, iters=40):
    """One process writes to a shared pinned buffer, another reads it."""
    mp.set_start_method("spawn", force=True)
    shared = torch.empty(shape, dtype=DTYPE).pin_memory().share_memory_()
    ready, done = mp.Event(), mp.Event()
    q = mp.Queue()
    w = mp.Process(target=_writer, args=(dev_a, shape, shared, ready, done, iters))
    r = mp.Process(target=_reader, args=(dev_b, shape, shared, ready, done, iters, q))
    w.start(); r.start()
    v = q.get()
    r.join(timeout=30); w.terminate(); w.join(timeout=10)
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--handoff", action="store_true",
                    help="also measure the two-process path (needs --dev-a/--dev-b)")
    ap.add_argument("--dev-a", default=None)
    ap.add_argument("--dev-b", default=None)
    args = ap.parse_args()

    p = torch.cuda.get_device_properties(0)
    print(f"device: {p.name} | {p.multi_processor_count} SM | "
          f"{p.total_memory/2**30:.1f} GiB\n")

    print(f"{'템플릿':>14} {'크기':>9} | {'D2H 고정':>9} {'H2D 고정':>9} {'왕복':>8} "
          f"{'대역폭':>10} | {'step 대비':>10} {'(70SM)':>9}")
    rows = []
    for name, shape, step_full, step_70 in CASES:
        r = bench_single(shape)
        rt = r["rt_pinned"]
        rows.append((name, r, rt / step_full, rt / step_70))
        print(f"{name:>14} {r['MiB']:>7.1f}MiB | {r['d2h_pinned']*1e3:>8.2f}ms "
              f"{r['h2d_pinned']*1e3:>8.2f}ms {rt*1e3:>7.2f}ms "
              f"{r['d2h_GBps']:>7.1f}GB/s | {rt/step_full*100:>9.2f}% "
              f"{rt/step_70*100:>8.2f}%")

    print(f"\n{'템플릿':>14} | {'pageable 왕복':>14} {'pinned 왕복':>13}  개선")
    for name, r, _, _ in rows:
        pa = (r["d2h_pageable"] + r["h2d_pageable"]) * 1e3
        pi = r["rt_pinned"] * 1e3
        print(f"{name:>14} | {pa:>12.2f}ms {pi:>11.2f}ms  {pa/pi:>5.1f}배")

    if args.handoff and args.dev_a and args.dev_b:
        print(f"\n두 프로세스 핸드오프 (공유 pinned 버퍼)")
        print(f"{'템플릿':>14} {'크기':>9} | {'왕복':>9} {'step 대비(70SM)':>16}")
        for name, shape, _, step_70 in CASES[:4]:
            v = bench_handoff(shape, args.dev_a, args.dev_b)
            print(f"{name:>14} {torch.empty(shape,dtype=DTYPE).numel()*2/2**20:>7.1f}MiB | "
                  f"{v*1e3:>8.2f}ms {v/step_70*100:>15.2f}%")


if __name__ == "__main__":
    main()
