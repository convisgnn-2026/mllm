#!/usr/bin/env python3
"""라운드로빈 프록시 (구성 A/B 용). 요청을 백엔드에 번갈아 보내고 스트리밍 응답을 그대로 흘린다.

  python rr_proxy_server.py --port 8192 --backends 127.0.0.1:8100 127.0.0.1:8200
  --weights 2 1   → 가중 라운드로빈 (구성 B: 4090 에 2건, 3090 에 1건 비율)

toy_proxy_server.py(vLLM, Apache-2.0) 의 클라이언트 패턴을 따른다.
"""
import argparse
import itertools
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

ARGS = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    limits = httpx.Limits(max_connections=2048, max_keepalive_connections=2048)
    clients = [httpx.AsyncClient(timeout=None, base_url=f"http://{b}", limits=limits) for b in ARGS.backends]
    seq = [c for c, w in zip(clients, ARGS.weights) for _ in range(w)]   # 가중치만큼 반복 → 순환
    app.state.rr = itertools.cycle(seq)
    app.state.clients = clients
    yield
    for c in clients:
        await c.aclose()


app = FastAPI(lifespan=lifespan)


async def _forward(api: str, request: Request):
    body = await request.json()
    client = next(app.state.rr)
    headers = {"Authorization": request.headers.get("Authorization", "")}
    if body.get("stream", False):
        async def gen():
            async with client.stream("POST", api, json=body, headers=headers) as r:
                async for chunk in r.aiter_bytes():
                    yield chunk
        return StreamingResponse(gen(), media_type="text/event-stream")
    r = await client.post(api, json=body, headers=headers)
    return StreamingResponse(iter([r.content]), media_type="application/json")


@app.post("/v1/completions")
async def completions(request: Request):
    return await _forward("/v1/completions", request)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await _forward("/v1/chat/completions", request)


@app.get("/health")
async def health():
    return {"status": "ok", "backends": ARGS.backends, "weights": ARGS.weights}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8192)
    p.add_argument("--backends", nargs="+", required=True, help="host:port ...")
    p.add_argument("--weights", nargs="+", type=int, default=None)
    ARGS = p.parse_args()
    if ARGS.weights is None:
        ARGS.weights = [1] * len(ARGS.backends)
    assert len(ARGS.weights) == len(ARGS.backends)
    import uvicorn
    uvicorn.run(app, host=ARGS.host, port=ARGS.port, log_level="warning")
