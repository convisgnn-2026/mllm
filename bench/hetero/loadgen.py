#!/usr/bin/env python3
"""닫힌/열린 부하 생성기. 정확한 토큰 길이의 요청 n개를 프록시에 보내고 요청별 시각을 CSV 로 남긴다.

  python loadgen.py --url http://127.0.0.1:8192 --model M --n 200 \
      --mix 2048:128=0.5,128:1024=0.5 --rate inf --out results/x

--mix   "in:out=비율,..."  요청 클래스와 비율. 순서는 시드로 섞는다.
--rate  inf = 전부 t=0 에 투입(닫힌 방식, 총 완료 시간 = makespan). 숫자 = 포아송 도착률(req/s).
출력   <out>/requests.csv (요청별), <out>/summary.json (집계).  ignore_eos 로 출력 길이를 정확히 고정.
"""
import argparse
import asyncio
import csv
import json
import os
import random
import time

import httpx
import numpy as np
from transformers import AutoTokenizer


def load_trace(path, n, seed, max_in=8000, offset=None):
    """Azure LLM 트레이스 형식(TIMESTAMP,ContextTokens,GeneratedTokens)에서 연속 n 건. 반환 [(in_len, out_len)], 상대 도착 시각(초)."""
    from datetime import datetime
    rows = list(csv.DictReader(open(path)))
    if offset is None:                                     # 전체 중앙값(입력·출력)에 가장 가까운 연속 구간을 고른다 (대표 구간)
        import statistics as st
        ci = [int(r["ContextTokens"]) for r in rows]; go = [int(r["GeneratedTokens"]) for r in rows]
        gi, gg = st.median(ci), st.median(go); best = (1e18, 0)
        for off in range(0, max(1, len(rows) - n), max(1, n // 4)):
            d = abs(st.median(ci[off:off + n]) - gi) / gi + abs(st.median(go[off:off + n]) - gg) / max(gg, 1)
            best = min(best, (d, off))
        offset = best[1]
    rows = rows[offset:offset + n]
    t = [datetime.strptime(r["TIMESTAMP"][:26], "%Y-%m-%d %H:%M:%S.%f") for r in rows]
    rel = [(x - t[0]).total_seconds() for x in t]
    lens = [(max(8, min(int(r["ContextTokens"]), max_in)), max(1, int(r["GeneratedTokens"]))) for r in rows]
    return lens, rel


def make_prompts_lens(tok, lens, seed):
    """요청마다 다른 (in_len, out_len). 프롬프트는 무작위 토큰을 디코드해 재토큰화 길이를 맞춘다."""
    rng = random.Random(seed); vocab = tok.vocab_size; items = []
    for sin, sout in lens:
        ids = [rng.randrange(1000, vocab - 1000) for _ in range(sin + 64)]
        text = tok.decode(ids, skip_special_tokens=True); enc = tok(text, add_special_tokens=False)["input_ids"]; target = sin - 1
        while len(enc) > target:
            enc = enc[:target]; text = tok.decode(enc, skip_special_tokens=True); enc = tok(text, add_special_tokens=False)["input_ids"]
        items.append((text, len(enc) + 1, sout))
    return items


def make_prompts(tok, classes, n, seed):
    """클래스 비율대로 n 개의 (prompt, in_len, out_len). 프롬프트는 무작위 토큰 id 를 디코드해 만들고
    재토큰화 길이가 정확히 in_len 이 되도록 잘라 맞춘다(BOS 포함)."""
    rng = random.Random(seed)
    vocab = tok.vocab_size
    items = []
    for (sin, sout, frac) in classes:
        k = round(n * frac)
        for _ in range(k):
            ids = [rng.randrange(1000, vocab - 1000) for _ in range(sin + 64)]
            text = tok.decode(ids, skip_special_tokens=True)
            enc = tok(text, add_special_tokens=False)["input_ids"]
            target = sin - 1                                   # 서버가 BOS 를 하나 붙인다
            while len(enc) > target:
                enc = enc[:target]; text = tok.decode(enc, skip_special_tokens=True)
                enc = tok(text, add_special_tokens=False)["input_ids"]
            items.append((text, len(enc) + 1, sout))
    while len(items) < n:                                       # 반올림으로 모자라면 첫 클래스로 채움
        items.append(items[0])
    rng.shuffle(items)
    return items[:n]


async def one(client, url, model, prompt, sout, idx, t_launch, rec, sem):
    async with sem:
        payload = {"model": model, "prompt": prompt, "max_tokens": sout, "temperature": 0.0,
                   "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}
        t0 = time.perf_counter(); first = None; ntok = 0; usage = None; err = ""
        try:
            async with client.stream("POST", url + "/v1/completions", json=payload) as r:
                if r.status_code != 200:
                    err = f"http {r.status_code}"
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    d = json.loads(data)
                    if d.get("usage"):
                        usage = d["usage"]
                    if d.get("choices") and d["choices"][0].get("text"):
                        if first is None:
                            first = time.perf_counter()
                        ntok += 1
        except Exception as e:  # 기록만 하고 계속
            err = repr(e)[:120]
        t1 = time.perf_counter()
        # TPOT 는 실제 생성 토큰 수(usage.completion_tokens)로 나눈다. 스트리밍 청크 하나에 토큰이 여러 개 실릴 수 있어
        # 청크 수(ntok)로 나누면 1.3~1.9배 부풀려진다 (2026-09-29 발견). 청크 수는 chunks 로 따로 남긴다.
        gen = (usage or {}).get("completion_tokens") or ntok
        rec.append(dict(idx=idx, launch=t_launch, start=t0, first=first or float("nan"), end=t1,
                        ttft=(first - t0) if first else float("nan"),
                        tpot=((t1 - first) / max(gen - 1, 1)) if first else float("nan"),
                        e2e=t1 - t0, out_tokens=gen, chunks=ntok,
                        usage_prompt=(usage or {}).get("prompt_tokens"), usage_completion=(usage or {}).get("completion_tokens"),
                        err=err))


async def run(a):
    tok = AutoTokenizer.from_pretrained(a.model)
    rng = np.random.default_rng(a.seed)
    if a.trace:
        lens, rel = load_trace(a.trace, a.n, a.seed)
        items = make_prompts_lens(tok, lens, a.seed)
        if a.trace_arrivals == "real":                       # 실제 도착 간격을 목표 평균 도착률로 배율 조정
            mean_rate = (len(rel) - 1) / max(rel[-1], 1e-9); arr = np.array(rel) * (mean_rate / float(a.rate))
        else:
            arr = np.cumsum(rng.exponential(1.0 / float(a.rate), a.n))
        a.mix = f"trace:{os.path.basename(a.trace)}:{a.trace_arrivals}"
    else:
        classes = []
        for part in a.mix.split(","):
            io, frac = part.split("="); sin, sout = io.split(":")
            classes.append((int(sin), int(sout), float(frac)))
        items = make_prompts(tok, classes, a.n, a.seed)
        gaps = np.zeros(a.n) if a.rate == "inf" else rng.exponential(1.0 / float(a.rate), a.n)
        arr = np.cumsum(gaps)
    rec = []; sem = asyncio.Semaphore(a.max_concurrency)
    limits = httpx.Limits(max_connections=4096, max_keepalive_connections=4096)
    async with httpx.AsyncClient(timeout=None, limits=limits) as client:
        T0 = time.perf_counter(); tasks = []
        for i, (prompt, sin, sout) in enumerate(items):
            delay = T0 + arr[i] - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(one(client, a.url, a.model, prompt, sout, i, time.perf_counter() - T0, rec, sem)))
        await asyncio.gather(*tasks)
        T1 = time.perf_counter()
    rec.sort(key=lambda r: r["idx"])
    for r, (_, sin, sout) in zip(rec, items):
        r["in_len"], r["out_len"] = sin, sout
        for k in ("start", "first", "end"):
            r[k] = r[k] - T0 if r[k] == r[k] else r[k]
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "requests.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rec[0].keys())); w.writeheader(); w.writerows(rec)
    ok = [r for r in rec if not r["err"] and r["out_tokens"] > 0]
    tt = np.array([r["ttft"] for r in ok]); tp = np.array([r["tpot"] for r in ok]); e2 = np.array([r["e2e"] for r in ok])
    out_tok = sum(r["out_tokens"] for r in ok); in_tok = sum(r["in_len"] for r in ok)
    mk = T1 - T0
    s = dict(config=a.tag, mix=a.mix, n=a.n, rate=a.rate, seed=a.seed, ok=len(ok), errors=len(rec) - len(ok),
             makespan_s=mk, req_per_s=len(ok) / mk, out_tok_per_s=out_tok / mk, total_tok_per_s=(in_tok + out_tok) / mk,
             ttft_ms=dict(mean=float(tt.mean() * 1e3), p50=float(np.percentile(tt, 50) * 1e3), p90=float(np.percentile(tt, 90) * 1e3), p99=float(np.percentile(tt, 99) * 1e3)),
             tpot_ms=dict(mean=float(tp.mean() * 1e3), p50=float(np.percentile(tp, 50) * 1e3), p90=float(np.percentile(tp, 90) * 1e3), p99=float(np.percentile(tp, 99) * 1e3)),
             e2e_s=dict(mean=float(e2.mean()), p50=float(np.percentile(e2, 50)), p90=float(np.percentile(e2, 90)), p99=float(np.percentile(e2, 99))))
    json.dump(s, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print(f"[{a.tag}] mix={a.mix} n={a.n} rate={a.rate}: makespan {mk:.1f}s | {s['req_per_s']:.2f} req/s | "
          f"{s['out_tok_per_s']:.0f} out tok/s | TTFT p50/p90 {s['ttft_ms']['p50']:.0f}/{s['ttft_ms']['p90']:.0f} ms | "
          f"TPOT p50/p90 {s['tpot_ms']['p50']:.1f}/{s['tpot_ms']['p90']:.1f} ms | errors {s['errors']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8192")
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--mix", default="2048:128=0.5,128:1024=0.5")
    p.add_argument("--rate", default="inf")
    p.add_argument("--trace", default="", help="Azure LLM 트레이스 CSV (TIMESTAMP,ContextTokens,GeneratedTokens)")
    p.add_argument("--trace-arrivals", default="real", choices=["real", "poisson"], help="real = 실제 도착 간격을 --rate 에 맞춰 배율 조정")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-concurrency", type=int, default=4096)
    p.add_argument("--tag", default="")
    p.add_argument("--out", required=True)
    asyncio.run(run(p.parse_args()))
