#!/usr/bin/env python3
"""Prepare and replay a fixed-duration LLM/DiT plan on one common clock.
No server management. All candidates must replay the same plan. Errors and
unfinished requests remain in the CSV. Warmup arrivals are excluded from scoring.
"""
import argparse
import asyncio
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import time

ROOT = Path(__file__).resolve().parents[2]
PROMPTS = ('a photo of a red bicycle', 'a snowy mountain village', 'a bowl of ramen', 'a lighthouse at dusk')


def prepare(a):
    if not all(math.isfinite(v) for v in (a.duration,a.warmup,a.llm_rate,a.dit_rate)) or a.duration <= 0 or a.warmup < 0 or min(a.llm_rate, a.dit_rate) < 0:
        raise ValueError('Invalid duration, warmup or rate')
    with a.trace.open() as f:
        trace = list(csv.DictReader(f))
    if not trace:
        raise ValueError('Empty trace')
    rng = random.Random(a.seed + 200000)
    offset = rng.randrange(len(trace))
    requests = []
    for kind, rate, seed in [('llm', a.llm_rate, a.seed), ('dit', a.dit_rate, a.seed + 100000)]:
        arrivals = random.Random(seed); t = 0.; idx = 0
        if rate == 0:
            continue
        while True:
            t += arrivals.expovariate(rate)
            if t >= a.warmup + a.duration:
                break
            req = dict(kind=kind, idx=idx, at_s=t, measured=t >= a.warmup)
            if kind == 'llm':
                row = trace[(offset + idx) % len(trace)]
                req.update(in_len=max(8, min(int(row['ContextTokens']), 8000)),
                           out_len=max(1, int(row['GeneratedTokens'])))
            else:
                req.update(prompt=PROMPTS[idx % len(PROMPTS)], image_seed=a.seed * 1000000 + idx)
            requests.append(req); idx += 1
    plan = dict(version=1, seed=a.seed, model=a.model, warmup_s=a.warmup, duration_s=a.duration,
                trace_sha256=hashlib.sha256(a.trace.read_bytes()).hexdigest(), trace_offset=offset,
                rates=dict(llm=a.llm_rate, dit=a.dit_rate), requests=sorted(requests, key=lambda r: (r['at_s'], r['kind'])))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    # Refuse to overwrite a plan used by earlier runs.
    with a.out.open('x') as f:
        json.dump(plan, f, indent=2)
    print(json.dumps(dict(plan=str(a.out), requests=len(requests), measured={k: sum(r['kind']==k and r['measured'] for r in requests) for k in ('llm','dit')})))


async def replay(a, transport=None):
    import httpx
    if a.drain_timeout <= 0 or not math.isfinite(a.drain_timeout):
        raise ValueError('drain timeout must be finite and positive')
    raw = a.plan.read_bytes(); plan = json.loads(raw)
    if a.out.exists():
        raise ValueError('Output already exists; choose a new directory')
    requests = [dict(r) for r in plan['requests'] if a.only == 'both' or r['kind'] == a.only]
    llm = [r for r in requests if r['kind'] == 'llm']
    if llm and not all('prompt' in r for r in llm):
        from loadgen import make_prompts_lens
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(plan['model'], local_files_only=True)
        prompts = make_prompts_lens(tok, [(r['in_len'], r['out_len']) for r in llm], plan['seed'])
        for r, (prompt, sin, _) in zip(llm, prompts):
            r['prompt'] = prompt; r['in_len'] = sin
    limits = httpx.Limits(max_connections=4096, max_keepalive_connections=4096)
    records = []
    async with httpx.AsyncClient(timeout=None, limits=limits, transport=transport) as client:
        for kind in {r['kind'] for r in requests}:
            url = a.llm_url if kind == 'llm' else a.dit_url
            response = await client.get(url + '/health', timeout=10)
            if kind == 'llm' and response.status_code == 404:
                response = await client.get(url + '/healthcheck', timeout=10)
            response.raise_for_status()
        # Both services are prepared and healthy before the single clock starts.
        epoch = time.time(); start = time.perf_counter()
        stop = plan['warmup_s'] + plan['duration_s']
        a.out.mkdir(parents=True)
        (a.out/'runtime.json').write_text(json.dumps(dict(start_unix_s=epoch,plan_sha256=hashlib.sha256(raw).hexdigest(),warmup_s=plan['warmup_s'],duration_s=plan['duration_s'],status='running'),indent=2))
        async def one(req):
            planned = req['at_s']; send = time.perf_counter() - start
            rec = {k: req[k] for k in ('kind','idx','measured')}
            rec.update(planned_s=planned, send_s=send, send_lag_s=send-planned,
                       first_s=None, end_s=None, ttft=None, tpot=None, e2e=None,
                       queue=None, encode=None, denoise=None, vae=None, out_tokens=None,
                       err='', in_len=req.get('in_len'), out_len=req.get('out_len'))
            try:
                if req['kind'] == 'llm':
                    usage = None; first = None
                    payload = dict(model=plan['model'], prompt=req['prompt'], max_tokens=req['out_len'],
                                   temperature=0., ignore_eos=True, stream=True, stream_options={'include_usage':True})
                    async with client.stream('POST', a.llm_url+'/v1/completions', json=payload) as response:
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            if not line.startswith('data:'): continue
                            value=line[5:].strip()
                            if value=='[DONE]': continue
                            item=json.loads(value)
                            if item.get('error'): raise ValueError(str(item['error']))
                            if item.get('usage'): usage=item['usage']
                            if item.get('choices') and item['choices'][0].get('text') and first is None:
                                first=time.perf_counter()-start
                    end=time.perf_counter()-start
                    gen=(usage or {}).get('completion_tokens',0)
                    if first is None or gen <= 0: raise ValueError('Missing first token or completion usage')
                    rec.update(first_s=first, ttft=first-planned, tpot=(end-first)/max(gen-1,1), out_tokens=gen)
                else:
                    response=await client.post(a.dit_url+'/generate',json=dict(prompt=req['prompt'],seed=req['image_seed']))
                    response.raise_for_status(); item=response.json()
                    if item.get('error'): raise ValueError(str(item['error']))
                    for k in ('queue','encode','denoise','vae'): rec[k]=item.get(k)
            except asyncio.CancelledError:
                rec['err']='drain_timeout'
            except Exception as ex:
                rec['err']=str(ex)[:300]
            finally:
                rec['end_s']=time.perf_counter()-start
                rec['e2e']=rec['end_s']-planned
                records.append(rec)
                with (a.out/'requests.live.jsonl').open('a') as checkpoint:
                    checkpoint.write(json.dumps(rec)+'\n')
                if len(records) % 100 == 0:
                    print(f'progress: completed={len(records)}/{len(requests)} elapsed={rec["end_s"]:.1f}s errors={sum(bool(r["err"]) for r in records)}',flush=True)
        tasks=[]
        try:
            for req in requests:
                await asyncio.sleep(max(0., start+req['at_s']-time.perf_counter()))
                tasks.append(asyncio.create_task(one(req)))
            await asyncio.sleep(max(0.,start+stop-time.perf_counter()))
            pending={task for task in tasks if not task.done()}
            if pending:
                _, pending=await asyncio.wait(pending, timeout=a.drain_timeout)
                for task in pending: task.cancel()
            await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks: task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
    records.sort(key=lambda r:(r['kind'],r['idx']))
    assert len(records)==len(requests)
    a.out.mkdir(parents=True,exist_ok=True)
    for kind in ('llm','dit'):
        rows=[r for r in records if r['kind']==kind]
        if rows:
            d=a.out/kind;d.mkdir()
            with (d/'requests.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report=dict(plan_sha256=hashlib.sha256(raw).hexdigest(), model=plan['model'],only=a.only,
                start_unix_s=epoch, warmup_s=plan['warmup_s'],duration_s=plan['duration_s'],
                drain_timeout_s=a.drain_timeout,requests=len(records),
                prompt_sha256=hashlib.sha256(json.dumps([(r['kind'],r['idx'],r['prompt']) for r in requests]).encode()).hexdigest(),
                errors=sum(bool(r['err']) for r in records), measured={})
    for kind in ('llm','dit'):
        rows=[r for r in records if r['kind']==kind and r['measured']]
        report['measured'][kind]=dict(n=len(rows), errors=sum(bool(r['err']) for r in rows),attainment={})
        for name,limits in [('A',dict(ttft=.5,tpot=.02) if kind=='llm' else dict(e2e=10.)),
                            ('B',dict(ttft=1.,tpot=.012) if kind=='llm' else dict(e2e=3.))]:
            passed=sum(not r['err'] and all(r.get(k) is not None and math.isfinite(r[k]) and r[k]<=v for k,v in limits.items()) for r in rows)
            report['measured'][kind]['attainment'][name]=passed/len(rows) if rows else None
    (a.out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    (a.out/'runtime.json').write_text(json.dumps(dict(report,status='finished'),indent=2)+'\n')
    # Time-binned backlog is derived from actual sends and completions, including warmup.
    with (a.out/'backlog.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['time_s','kind','sent','completed','inflight'])
        for second in range(math.ceil(stop)+1):
            for kind in ('llm','dit'):
                rows=[r for r in records if r['kind']==kind]
                sent=sum(r['send_s']<=second for r in rows);completed=sum(r['end_s']<=second for r in rows)
                writer.writerow([second,kind,sent,completed,sent-completed])
    print(json.dumps(report,indent=2))
    return report,records


def main():
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--trace',type=Path,default=ROOT/'bench/hetero/traces/AzureLLMInferenceTrace_code.csv')
    p.add_argument('--model',default='meta-llama/Llama-3.2-3B-Instruct');p.add_argument('--seed',type=int,default=0)
    p.add_argument('--duration',type=float,default=600);p.add_argument('--warmup',type=float,default=60)
    p.add_argument('--llm-rate',type=float,default=2);p.add_argument('--dit-rate',type=float,default=.3);p.add_argument('--out',type=Path,required=True)
    p=sub.add_parser('replay');p.add_argument('--plan',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--only',choices=['both','llm','dit'],default='both');p.add_argument('--drain-timeout',type=float,default=300)
    p.add_argument('--llm-url',default='http://127.0.0.1:8192');p.add_argument('--dit-url',default='http://127.0.0.1:8300')
    args=ap.parse_args()
    if args.command=='prepare':prepare(args)
    else:asyncio.run(replay(args))

if __name__=='__main__':main()
