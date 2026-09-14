#!/usr/bin/env python
"""
Colocation interference: what happens when a compute-bound and a memory-bound
tenant share one GPU?

This is the coefficient the co-solver (solver/cosolve.py) needs to add "put both
on the same GPU" as a choice.  Without it the solver can only give each tenant
whole GPUs, which makes its answer "the sum of each tenant's requirement" -- too
thin to need a solver.

Two tenants:
  DiT     the real Wan2.1 transformer step.  Compute-bound (§12 P2: scales with
          SM count at 1.71x when SMs are halved).
  decode  a proxy for LLM token generation: x @ W with a tiny batch against a
          multi-GB weight tensor, i.e. GEMV at low arithmetic intensity, which is
          what makes decode memory-bandwidth-bound.  Not a real LLM -- the real
          one goes on the desktop with vLLM.

Concurrency is two CUDA streams **in one process**.  Separate processes would
time-slice (measured in §12 K1: 7 processes on one GPU are 4-13% *slower* than
running them one after another), so they can never show complementarity.  Two
streams is the honest upper bound on what MPS would give, without touching the
MPS daemon on a shared machine.

Reported per pairing:
    slowdown_A = time_A(together) / time_A(alone)
    combined   = 1/slowdown_A + 1/slowdown_B
`combined > 1` means colocation is worth it; `= 1` means it is pure time-slicing;
`< 1` means the pair actively interferes.
"""
import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != "/" and not _os.path.isdir(_os.path.join(_d, "results")):
    _d = _os.path.dirname(_d)
_os.chdir(_d)

import argparse
import csv
import json
import time

import torch

MODEL = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
DTYPE = torch.bfloat16


def build_dit(dev, tpl):
    from diffusers import WanTransformer3DModel
    tr = WanTransformer3DModel.from_pretrained(MODEL, subfolder="transformer", dtype=DTYPE).to(dev).eval()
    w, h, f = tpl
    lf, lh, lw = (f - 1) // 4 + 1, h // 8, w // 8
    lat = torch.randn(1, 16, lf, lh, lw, device=dev, dtype=DTYPE)
    ts = torch.tensor([500.0], device=dev, dtype=DTYPE)
    ctx = torch.randn(1, 512, 4096, device=dev, dtype=DTYPE)

    def step():
        with torch.no_grad():
            tr(hidden_states=lat, timestep=ts, encoder_hidden_states=ctx, return_dict=False)
    return step


def build_decode(dev, weight_gib, batch, hidden=8192):
    """GEMV against a big weight stack: the memory-bound half of LLM decode."""
    bytes_per = hidden * hidden * 2
    n_mat = max(1, int(weight_gib * (1 << 30) // bytes_per))
    W = [torch.randn(hidden, hidden, device=dev, dtype=DTYPE) for _ in range(n_mat)]
    x = torch.randn(batch, hidden, device=dev, dtype=DTYPE)

    def step():
        with torch.no_grad():
            h = x
            for m in W:
                h = h @ m
    return step, n_mat * bytes_per / (1 << 30)


def timeit(fn, iters, warmup=3, stream=None):
    ctx = torch.cuda.stream(stream) if stream is not None else torch.cuda.stream(torch.cuda.current_stream())
    with ctx:
        for _ in range(warmup):
            fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with ctx:
        for _ in range(iters):
            fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def together(fa, fb, sa, sb, secs):
    """Drive both streams from their own CPU thread for `secs` seconds and count
    iterations.  Threads matter: enqueueing one stream's whole loop before the
    other's makes the second stream sit idle while the first is being queued,
    which shows up as a huge fake slowdown.
    """
    import threading
    out = {}

    def drive(tag, fn, st):
        with torch.cuda.stream(st):
            for _ in range(2):
                fn()
        st.synchronize()
        n = 0
        t0 = time.perf_counter()
        with torch.cuda.stream(st):
            while time.perf_counter() - t0 < secs:
                fn(); n += 1
        st.synchronize()
        out[tag] = (time.perf_counter() - t0) / max(1, n)

    ta = threading.Thread(target=drive, args=("a", fa, sa))
    tb = threading.Thread(target=drive, args=("b", fb, sb))
    ta.start(); tb.start(); ta.join(); tb.join()
    return out["a"], out["b"]


def alone(fn, secs, st):
    with torch.cuda.stream(st):
        for _ in range(2):
            fn()
    st.synchronize()
    n = 0
    t0 = time.perf_counter()
    with torch.cuda.stream(st):
        while time.perf_counter() - t0 < secs:
            fn(); n += 1
    st.synchronize()
    return (time.perf_counter() - t0) / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", default="832x480x33")
    ap.add_argument("--weight-gib", type=float, default=16.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--secs", type=float, default=6.0, help="measurement window per case")
    ap.add_argument("-o", "--out", default="results/colocate.csv")
    a = ap.parse_args()
    dev = torch.device("cuda")
    p = torch.cuda.get_device_properties(0)
    tpl = tuple(int(x) for x in a.template.split("x"))
    print(f"device: {p.name} | SM {p.multi_processor_count} | {p.total_memory/2**30:.0f} GiB", flush=True)

    dit = build_dit(dev, tpl)
    dec, wg = build_decode(dev, a.weight_gib, a.batch)
    print(f"decode proxy: {wg:.1f} GiB of weights, batch {a.batch}", flush=True)

    sa, sb = torch.cuda.Stream(), torch.cuda.Stream()
    rows = []

    t_dit = alone(dit, a.secs, sa)
    t_dec = alone(dec, a.secs, sa)
    print(f"\nalone   DiT {t_dit*1e3:8.1f} ms   decode {t_dec*1e3:8.1f} ms", flush=True)
    rows.append(dict(pair="alone", a="dit", b="decode", ta=t_dit, tb=t_dec, sa=1.0, sb=1.0, combined=2.0))

    for name, fa, fb, base_a, base_b in [
            ("dit+decode", dit, dec, t_dit, t_dec),
            ("dit+dit", dit, dit, t_dit, t_dit),
            ("decode+decode", dec, dec, t_dec, t_dec)]:
        ta, tb = together(fa, fb, sa, sb, a.secs)
        ra, rb = ta / base_a, tb / base_b
        comb = 1 / ra + 1 / rb
        print(f"{name:>14}  A {ta*1e3:8.1f} ms ({ra:.2f}x)   B {tb*1e3:8.1f} ms ({rb:.2f}x)   "
              f"합산 처리량 {comb:.2f}x", flush=True)
        rows.append(dict(pair=name, a="A", b="B", ta=ta, tb=tb, sa=ra, sb=rb, combined=comb))

    with open(a.out, "w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {a.out}")
    print("합산 > 1.0 이면 동거가 이득, = 1.0 이면 순수 시분할, < 1.0 이면 서로 방해")


if __name__ == "__main__":
    main()
