#!/usr/bin/env python3
"""DiT(SD3.5-medium) 단계 분리 가능한 최소 서빙 서버 (diffusers 직접 사용, T5 제외).

  python dit_server.py --port 8300 --enc-dev cuda:0 --dit-dev cuda:0 --vae-dev cuda:0 --size 512 --steps 28

단계: encode(CLIP×2) → denoise(MMDiT, steps) → vae(decode). 각 단계 장치를 따로 둘 수 있다(DDiT 의 immutable 분리).
요청은 FIFO 직렬 처리(배치 1). POST /generate {"prompt": str, "seed": int} → 단계별 시간(초).
"""
import argparse, asyncio, json, time
import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="stabilityai/stable-diffusion-3.5-medium")
ap.add_argument("--port", type=int, default=8300)
ap.add_argument("--enc-dev", default="cuda:0"); ap.add_argument("--dit-dev", default="cuda:0"); ap.add_argument("--vae-dev", default="cuda:0")
ap.add_argument("--size", type=int, default=512); ap.add_argument("--steps", type=int, default=28); ap.add_argument("--cfg", type=float, default=4.5)
ap.add_argument("--max-batch", type=int, default=1, help="대기 중 요청을 최대 이만큼 묶어 한 번에 denoise")
a = ap.parse_args()

from diffusers import StableDiffusion3Pipeline
pipe = StableDiffusion3Pipeline.from_pretrained(a.model, text_encoder_3=None, tokenizer_3=None, torch_dtype=torch.bfloat16)
pipe.set_progress_bar_config(disable=True)
pipe.text_encoder.to(a.enc_dev); pipe.text_encoder_2.to(a.enc_dev)
pipe.transformer.to(a.dit_dev); pipe.vae.to(a.vae_dev)
pipe._execution_device_override = None

def sync(dev): torch.cuda.synchronize(torch.device(dev))

@torch.inference_mode()
def run(prompts, seeds):
    t = {}
    t0 = time.perf_counter()
    pe, npe, ppe, nppe = pipe.encode_prompt(prompt=prompts, prompt_2=None, prompt_3=None, do_classifier_free_guidance=True, device=torch.device(a.enc_dev), max_sequence_length=77)
    sync(a.enc_dev); t["encode"] = time.perf_counter() - t0
    t1 = time.perf_counter()
    pe, npe, ppe, nppe = [x.to(a.dit_dev) for x in (pe, npe, ppe, nppe)]
    gens = [torch.Generator(a.dit_dev).manual_seed(s) for s in seeds]
    pipe.transformer.to(a.dit_dev)
    # pipeline 호출은 output_type=latent 로 denoise 만 수행 (vae 미사용)
    orig_dev = pipe._execution_device
    lat = pipe(prompt_embeds=pe, negative_prompt_embeds=npe, pooled_prompt_embeds=ppe, negative_pooled_prompt_embeds=nppe,
               num_inference_steps=a.steps, guidance_scale=a.cfg, height=a.size, width=a.size, generator=gens, output_type="latent").images
    sync(a.dit_dev); t["denoise"] = time.perf_counter() - t1
    t2 = time.perf_counter()
    lat = lat.to(a.vae_dev)
    lat = (lat / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    img = pipe.vae.decode(lat, return_dict=False)[0]
    img = pipe.image_processor.postprocess(img, output_type="pil")
    sync(a.vae_dev); t["vae"] = time.perf_counter() - t2
    t["total"] = time.perf_counter() - t0
    return t, img

# diffusers 의 _execution_device 는 첫 모듈 기준이라 단계별 장치가 다르면 transformer 장치로 고정
import types
StableDiffusion3Pipeline._execution_device = property(lambda self: torch.device(a.dit_dev))

app = FastAPI()
q: asyncio.Queue = None
async def worker():
    loop = asyncio.get_event_loop()
    while True:
        first = await q.get(); batch = [first]
        while len(batch) < a.max_batch and not q.empty(): batch.append(q.get_nowait())
        ts = time.perf_counter()
        try:
            t, imgs = await loop.run_in_executor(None, run, [b["prompt"] for b in batch], [b["seed"] for b in batch])
            for b in batch: b["fut"].set_result(dict(t, start=ts, batch=len(batch)))
        except Exception as e:
            for b in batch: b["fut"].set_exception(e)

@app.on_event("startup")
async def _s():
    global q; q = asyncio.Queue(); asyncio.create_task(worker())
    run(["warmup"], [0]); run(["warmup"], [1])
    print("READY", flush=True)

@app.post("/generate")
async def gen(body: dict):
    fut = asyncio.get_event_loop().create_future(); arr = time.perf_counter()
    await q.put(dict(prompt=body.get("prompt", ""), seed=int(body.get("seed", 0)), fut=fut))
    try:
        r = await fut
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    r["queue"] = r["start"] - arr; r.pop("start"); return r

@app.get("/health")
async def health(): return {"ok": True}

uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")
