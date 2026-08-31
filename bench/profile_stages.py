#!/usr/bin/env python
"""
Stage-level profiler for video DiT.

Measures, per request template (W x H x frames):
  - text encode  : latency, peak mem, embedding size
  - DiT loop     : per-step latency (mean/std/CV), peak mem, latent size
  - VAE decode   : latency, peak mem

Rationale: TetriServe reports step-time CV < 0.7% for image DiT. We check
whether video DiT is equally predictable, and we break the pipeline into its
three stages because their resource profiles differ (this is what motivates
per-stage / per-granularity placement).

Usage:
  python profile_stages.py --sweep --out results/stages.csv
  python profile_stages.py --templates 832x480x81 1280x720x81 --steps 8
"""

import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != "/" and not _os.path.isdir(_os.path.join(_d, "results")):
    _d = _os.path.dirname(_d)
_sys.path[:0] = [_os.path.join(_d, "solver"), _os.path.join(_d, "bench")]
_os.chdir(_d)            # results/ and slo_configs/ are addressed from the repo root

import argparse
import csv
import gc
import json
import os
import statistics
import sys
import time

import torch

MODEL = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
DTYPE = torch.bfloat16
PROMPT = "A curious raccoon peers through a field of sunflowers at sunset"

# (width, height, frames) -- ordered roughly by cost
SWEEP = [
    (480, 272, 17), (480, 272, 33), (480, 272, 49),
    (832, 480, 17), (832, 480, 33), (832, 480, 49), (832, 480, 81),
    (1280, 720, 17), (1280, 720, 33), (1280, 720, 49), (1280, 720, 81),
    (1920, 1088, 33), (1920, 1088, 49), (1920, 1088, 81),
]


def gib(x):
    return x / 2**30


def reset_peak():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def peak():
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated()


def timed(fn, warmup, iters):
    """Return list of per-call wall times (seconds)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        out.append(time.perf_counter() - t0)
    return out


def stats(xs):
    m = statistics.mean(xs)
    s = statistics.pstdev(xs) if len(xs) > 1 else 0.0
    return m, s, (s / m * 100 if m else 0.0)


# --------------------------------------------------------------------------
# Stage 1: text encoder (run once, cache the embedding, then free 21 GB of T5)
# --------------------------------------------------------------------------
def encode_prompt_once(cache_path, device):
    if os.path.exists(cache_path):
        d = torch.load(cache_path, map_location="cpu")
        print(f"[text] cached embed {tuple(d['embeds'].shape)} "
              f"({d['latency_s']*1e3:.0f} ms, peak {gib(d['peak_bytes']):.2f} GiB)")
        return d

    from transformers import AutoTokenizer, UMT5EncoderModel

    print("[text] loading UMT5 encoder (~21 GB on disk) ...")
    tok = AutoTokenizer.from_pretrained(MODEL, subfolder="tokenizer")
    enc = UMT5EncoderModel.from_pretrained(MODEL, subfolder="text_encoder",
                                           dtype=DTYPE).to(device).eval()
    ids = tok([PROMPT], padding="max_length", max_length=512,
              truncation=True, return_tensors="pt")
    ids = {k: v.to(device) for k, v in ids.items()}

    reset_peak()
    with torch.no_grad():
        ts = timed(lambda: enc(**ids), warmup=1, iters=3)
    pk = peak()
    with torch.no_grad():
        embeds = enc(**ids).last_hidden_state

    d = {"embeds": embeds.detach().cpu(),
         "latency_s": statistics.mean(ts),
         "peak_bytes": pk,
         "weight_bytes": sum(p.numel() * p.element_size() for p in enc.parameters())}
    torch.save(d, cache_path)
    print(f"[text] {tuple(embeds.shape)}  {d['latency_s']*1e3:.0f} ms  "
          f"peak {gib(pk):.2f} GiB  weights {gib(d['weight_bytes']):.2f} GiB")

    del enc, embeds
    gc.collect()
    torch.cuda.empty_cache()
    return d


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--templates", nargs="*", default=None,
                    help="WxHxF, e.g. 832x480x81")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--steps", type=int, default=8, help="DiT steps to time")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--out", default="results/stages.csv")
    ap.add_argument("--skip-vae", action="store_true")
    ap.add_argument("--tag", default="full")
    args = ap.parse_args()

    if args.sweep or not args.templates:
        templates = SWEEP
    else:
        templates = [tuple(int(x) for x in t.lower().split("x"))
                     for t in args.templates]

    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    print(f"device: {props.name} | SMs {props.multi_processor_count} | "
          f"{gib(props.total_memory):.1f} GiB | tag={args.tag}")

    from diffusers import AutoencoderKLWan, WanTransformer3DModel

    txt = encode_prompt_once("results/prompt_embed.pt", dev)
    ctx = txt["embeds"].to(dev, DTYPE)

    print("[dit] loading transformer ...")
    tr = WanTransformer3DModel.from_pretrained(MODEL, subfolder="transformer",
                                               dtype=DTYPE).to(dev).eval()
    dit_w = sum(p.numel() * p.element_size() for p in tr.parameters())
    print(f"[dit] params {sum(p.numel() for p in tr.parameters())/1e9:.2f} B, "
          f"weights {gib(dit_w):.2f} GiB")

    vae = None
    if not args.skip_vae:
        print("[vae] loading ...")
        vae = AutoencoderKLWan.from_pretrained(MODEL, subfolder="vae",
                                               dtype=torch.float32).to(dev).eval()
        vae_w = sum(p.numel() * p.element_size() for p in vae.parameters())
        print(f"[vae] weights {gib(vae_w):.2f} GiB")
    else:
        vae_w = 0

    # latent geometry (Wan VAE: /8 spatial, /4 temporal, 16 channels)
    zc = tr.config.in_channels
    sp, tp = 8, 4

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    new = not os.path.exists(args.out)
    fh = open(args.out, "a", newline="")
    wr = csv.writer(fh)
    if new:
        wr.writerow([
            "tag", "sm_count", "gpu_mem_gib", "width", "height", "frames",
            "lat_f", "lat_h", "lat_w", "tokens", "latent_mib",
            "text_ms", "text_peak_gib", "text_w_gib",
            "step_ms_mean", "step_ms_std", "step_cv_pct", "dit_peak_gib", "dit_w_gib",
            "vae_ms", "vae_peak_gib", "vae_w_gib", "status",
        ])

    for (W, H, F) in templates:
        lf = (F - 1) // tp + 1
        lh, lw = H // sp, W // sp
        tokens = lf * lh * lw  # before patchify
        lat_bytes = lf * lh * lw * zc * 2
        name = f"{W}x{H}x{F}"
        print(f"\n=== {name}  latent {zc}x{lf}x{lh}x{lw}  "
              f"({tokens/1000:.1f}k pos, {lat_bytes/2**20:.1f} MiB) ===")

        row = dict(tag=args.tag, sm_count=props.multi_processor_count,
                   gpu_mem_gib=round(gib(props.total_memory), 1),
                   width=W, height=H, frames=F, lat_f=lf, lat_h=lh, lat_w=lw,
                   tokens=tokens, latent_mib=round(lat_bytes / 2**20, 2),
                   text_ms=round(txt["latency_s"] * 1e3, 2),
                   text_peak_gib=round(gib(txt["peak_bytes"]), 3),
                   text_w_gib=round(gib(txt["weight_bytes"]), 3),
                   dit_w_gib=round(gib(dit_w), 3),
                   vae_w_gib=round(gib(vae_w), 3), status="ok")

        # ---- DiT loop ----
        try:
            lat = torch.randn(1, zc, lf, lh, lw, device=dev, dtype=DTYPE)
            tstep = torch.tensor([500.0], device=dev, dtype=DTYPE)
            reset_peak()

            def step():
                with torch.no_grad():
                    tr(hidden_states=lat, timestep=tstep,
                       encoder_hidden_states=ctx, return_dict=False)

            ts = timed(step, warmup=args.warmup, iters=args.steps)
            m, s, cv = stats([t * 1e3 for t in ts])
            row.update(step_ms_mean=round(m, 2), step_ms_std=round(s, 3),
                       step_cv_pct=round(cv, 3),
                       dit_peak_gib=round(gib(peak()), 3))
            print(f"  DiT  step {m:8.1f} ms  CV {cv:5.2f}%  peak {gib(peak()):6.2f} GiB")
        except torch.cuda.OutOfMemoryError:
            row["status"] = "dit_oom"
            print("  DiT  OOM")
            gc.collect(); torch.cuda.empty_cache()
            wr.writerow([row.get(k, "") for k in _COLS]); fh.flush(); continue
        except Exception as e:
            row["status"] = f"dit_err:{type(e).__name__}"
            print(f"  DiT  ERROR {type(e).__name__}: {str(e)[:120]}")
            gc.collect(); torch.cuda.empty_cache()
            wr.writerow([row.get(k, "") for k in _COLS]); fh.flush(); continue

        # ---- VAE decode ----
        if vae is not None:
            try:
                reset_peak()
                z = lat.float()
                # Wan VAE expects latents scaled; magnitude only affects numerics,
                # not the cost profile we are measuring here.
                def dec():
                    with torch.no_grad():
                        vae.decode(z, return_dict=False)
                vt = timed(dec, warmup=1, iters=2)
                row.update(vae_ms=round(statistics.mean(vt) * 1e3, 2),
                           vae_peak_gib=round(gib(peak()), 3))
                print(f"  VAE  decode {statistics.mean(vt)*1e3:8.1f} ms  "
                      f"peak {gib(peak()):6.2f} GiB")
                del z
            except torch.cuda.OutOfMemoryError:
                row.update(vae_ms=-1, vae_peak_gib=-1, status="vae_oom")
                print("  VAE  OOM")
            except Exception as e:
                row.update(vae_ms=-1, vae_peak_gib=-1,
                           status=f"vae_err:{type(e).__name__}")
                print(f"  VAE  ERROR {type(e).__name__}: {str(e)[:120]}")

        del lat
        gc.collect()
        torch.cuda.empty_cache()
        wr.writerow([row.get(k, "") for k in _COLS])
        fh.flush()

    fh.close()
    print(f"\nwrote {args.out}")


_COLS = ["tag", "sm_count", "gpu_mem_gib", "width", "height", "frames",
         "lat_f", "lat_h", "lat_w", "tokens", "latent_mib",
         "text_ms", "text_peak_gib", "text_w_gib",
         "step_ms_mean", "step_ms_std", "step_cv_pct", "dit_peak_gib", "dit_w_gib",
         "vae_ms", "vae_peak_gib", "vae_w_gib", "status"]


if __name__ == "__main__":
    main()
