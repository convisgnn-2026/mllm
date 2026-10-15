#!/usr/bin/env python3
"""DiT 열린 부하 생성기. 포아송 도착률로 n 건을 dit_server 에 보내고 요청별 단계 시간을 남긴다.

  python dit_loadgen.py --url http://127.0.0.1:8300 --n 60 --rate 0.2 --out results/x
출력 <out>/requests.csv, <out>/summary.json (e2e p50/p90/p99, 단계별 평균, 달성 rps).
"""
import argparse, asyncio, csv, json, os, random, time
import httpx, numpy as np

PROMPTS = ["a photo of a red bicycle leaning on a brick wall", "an oil painting of a lighthouse at dusk", "a bowl of ramen on a wooden table, studio light",
           "a snowy mountain village at night, lanterns", "a close-up of a tabby cat wearing glasses", "a futuristic city skyline with flying cars",
           "a watercolor of a sailboat on a calm lake", "a plate of sushi with chopsticks, top view", "an astronaut riding a horse on mars",
           "a cozy cabin interior with a fireplace", "a vintage car on a desert highway", "a bouquet of sunflowers in a glass vase"]

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8300"); ap.add_argument("--n", type=int, default=60); ap.add_argument("--rate", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True); ap.add_argument("--tag", default="")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.seed); np.random.seed(a.seed)
    gaps = np.random.exponential(1.0 / a.rate, a.n) if a.rate != float("inf") else np.zeros(a.n)
    launch = np.cumsum(gaps) - gaps[0]
    rec = [None] * a.n
    async with httpx.AsyncClient(timeout=None) as cl:
        t0 = time.perf_counter()
        async def one(i):
            await asyncio.sleep(max(0.0, launch[i] - (time.perf_counter() - t0)))
            s = time.perf_counter(); p = rng.choice(PROMPTS)
            try:
                r = await cl.post(a.url + "/generate", json=dict(prompt=p, seed=i)); j = r.json()
                e = time.perf_counter()
                rec[i] = dict(idx=i, launch=s - t0, end=e - t0, e2e=e - s, queue=j.get("queue"), encode=j.get("encode"), denoise=j.get("denoise"), vae=j.get("vae"), total=j.get("total"), batch=j.get("batch"), err=j.get("error", ""))
            except Exception as ex:
                rec[i] = dict(idx=i, launch=s - t0, end=time.perf_counter() - t0, e2e=None, queue=None, encode=None, denoise=None, vae=None, total=None, batch=None, err=str(ex))
        await asyncio.gather(*[one(i) for i in range(a.n)])
        mk = time.perf_counter() - t0
    with open(os.path.join(a.out, "requests.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rec[0].keys())); w.writeheader(); w.writerows(rec)
    ok = [r for r in rec if not r["err"]]; e2 = np.array([r["e2e"] for r in ok])
    pct = lambda x: dict(mean=float(np.mean(x)), p50=float(np.percentile(x, 50)), p90=float(np.percentile(x, 90)), p99=float(np.percentile(x, 99)))
    s = dict(tag=a.tag, n=a.n, rate=a.rate, ok=len(ok), errors=a.n - len(ok), makespan_s=mk, req_per_s=len(ok) / mk, e2e_s=pct(e2),
             queue_s=pct([r["queue"] for r in ok]), encode_s=pct([r["encode"] for r in ok]), denoise_s=pct([r["denoise"] for r in ok]), vae_s=pct([r["vae"] for r in ok]))
    json.dump(s, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print(f"[{a.tag}] rate {a.rate} n {a.n} ok {len(ok)} | {s['req_per_s']:.3f} rps | e2e p50/p90 {s['e2e_s']['p50']:.2f}/{s['e2e_s']['p90']:.2f} s | "
          f"queue p50 {s['queue_s']['p50']:.2f} s | denoise {s['denoise_s']['mean']:.2f} s vae {s['vae_s']['mean']:.3f} s")
asyncio.run(main())
