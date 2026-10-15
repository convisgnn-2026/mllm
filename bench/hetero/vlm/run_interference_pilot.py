#!/usr/bin/env python3
"""Shared LLM/vision encoder pilot. Idle-resident baselines; not solver proof."""
import argparse,asyncio,base64,csv,hashlib,io,json,math,os,random,signal,subprocess,sys,time
from pathlib import Path
import httpx
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[3]
PY=sys.executable
GPU4='GPU-d5bd27b1-7998-9ae2-b764-3c528e12a47b';GPU3='GPU-208c8076-e9ae-fde6-aacf-b0c3fae16a9e'
VLM='Qwen/Qwen2-VL-2B-Instruct';LLM='meta-llama/Llama-3.2-3B-Instruct'


def plan(seed,horizon,llm_rate,vlm_rate):
    out=[]
    for kind,rate in [('llm',llm_rate),('vlm',vlm_rate)]:
        rng=random.Random(seed+(10000 if kind=='vlm' else 0));t=0;idx=0
        while rate>0:
            t+=rng.expovariate(rate)
            if t>=horizon:break
            out.append(dict(kind=kind,idx=idx,planned_s=t));idx+=1
    return sorted(out,key=lambda r:(r['planned_s'],r['kind']))


def percentile(rows,key):
    values=[r[key] for r in rows if not r['error'] and r.get(key) is not None]
    return float(np.percentile(values,90)) if values else None


async def replay(case,requests,payloads,out,warmup,duration,gate,profile=False):
    out.mkdir();rows=[]
    async with httpx.AsyncClient(timeout=None,limits=httpx.Limits(max_connections=512)) as client:
        r=await client.post(f'http://127.0.0.1:8530/experiment/gate/{gate}');r.raise_for_status()
        t0=time.perf_counter();epoch=time.time();recent_gaps=[];prefill_pending=set();feedback_stop=asyncio.Event()
        async def report_feedback():
            while not feedback_stop.is_set():
                await asyncio.sleep(.100)
                gaps=list(recent_gaps);recent_gaps.clear()
                try:
                    response=await client.post('http://127.0.0.1:8530/experiment/feedback',json={'max_gap_s':max(gaps,default=0),'prefill_active':len(prefill_pending),'source_unix_s':time.time()},timeout=2)
                    response.raise_for_status()
                except Exception as ex:
                    with (out/'feedback_errors.jsonl').open('a') as f:f.write(json.dumps({'error':str(ex)})+'\n')
        reporter=asyncio.create_task(report_feedback())
        async def capture_profile():
            await asyncio.sleep(max(0,t0+warmup+5-time.perf_counter()))
            controls=[];started=[]
            try:
                for port in (8510,8540):
                    before=time.time()
                    response=await client.post(f'http://127.0.0.1:{port}/start_profile',timeout=120)
                    response.raise_for_status();started.append(port)
                    controls.append({'port':port,'command':'start','before_unix_s':before,'after_unix_s':time.time()})
                await asyncio.sleep(8)
            finally:
                for port in reversed(started):
                    before=time.time()
                    response=await client.post(f'http://127.0.0.1:{port}/stop_profile',timeout=120)
                    response.raise_for_status()
                    controls.append({'port':port,'command':'stop','before_unix_s':before,'after_unix_s':time.time()})
                (out/'profile_control.json').write_text(json.dumps(controls,indent=2)+'\n')
        profile_task=asyncio.create_task(capture_profile()) if profile else None
        async def one(req):
            body=payloads[(req['kind'],req['idx'])];first=None;gaps=[];last=None;usage=None;chunks=0;chunk_times=[]
            row=dict(req,measured=req['planned_s']>=warmup,send_s=time.perf_counter()-t0,error='',ttft_s=None,tpot_s=None,e2e_s=None,chunk_gap_p90_s=None)
            if req['kind']=='llm':prefill_pending.add(req['idx'])
            try:
                port=8540 if req['kind']=='llm' else 8530
                async with client.stream('POST',f'http://127.0.0.1:{port}/v1/chat/completions',json=body) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.startswith('data:'):continue
                        text=line[5:].strip()
                        if text=='[DONE]':continue
                        item=json.loads(text)
                        if item.get('error'):raise ValueError(str(item['error']))
                        if item.get('usage'):usage=item['usage']
                        choices=item.get('choices') or []
                        if choices and choices[0].get('delta',{}).get('content'):
                            now=time.perf_counter()-t0
                            if first is None:
                                first=now
                                if req['kind']=='llm':prefill_pending.discard(req['idx'])
                            chunk_times.append(now)
                            if last is not None:
                                gaps.append(now-last)
                                if req['kind']=='llm':recent_gaps.append(now-last)
                            last=now;chunks+=1
                end=time.perf_counter()-t0;nt=(usage or {}).get('completion_tokens',0)
                if first is None or nt<=0:raise ValueError('No content or completion usage')
                row.update(ttft_s=first-req['planned_s'],tpot_s=(end-first)/max(nt-1,1),output_tokens=nt,chunk_gap_p90_s=float(np.percentile(gaps,90)) if gaps else None)
            except asyncio.CancelledError:row['error']='drain_timeout'
            except Exception as ex:row['error']=str(ex)[:250]
            finally:
                if req['kind']=='llm':prefill_pending.discard(req['idx'])
                row['chunk_times_s']=chunk_times;row['start_unix_s']=epoch;row['end_s']=time.perf_counter()-t0;row['e2e_s']=row['end_s']-req['planned_s'];rows.append(row)
                with (out/'requests.live.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        tasks=[]
        for req in requests:
            await asyncio.sleep(max(0,t0+req['planned_s']-time.perf_counter()));tasks.append(asyncio.create_task(one(req)))
        await asyncio.sleep(max(0,t0+warmup+duration-time.perf_counter()))
        pending={t for t in tasks if not t.done()}
        if pending:
            _,pending=await asyncio.wait(pending,timeout=120)
            for t in pending:t.cancel()
        await asyncio.gather(*tasks)
        feedback_stop.set();await reporter
        if profile_task is not None:await profile_task
    result=dict(case=case,start_unix_s=epoch,warmup_s=warmup,duration_s=duration,gate_limit=gate,plan_sha256=hashlib.sha256(json.dumps(requests).encode()).hexdigest(),metrics={})
    for kind in ('llm','vlm'):
        selected=[r for r in rows if r['kind']==kind and r['measured']]
        result['metrics'][kind]=dict(n=len(selected),errors=sum(bool(r['error']) for r in selected),**{k:percentile(selected,k) for k in ('ttft_s','tpot_s','e2e_s','chunk_gap_p90_s')})
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    if any(r['error']=='drain_timeout' for r in rows):
        raise RuntimeError('Drain timeout; do not start another case with server backlog')
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--duration',type=float,default=90);ap.add_argument('--warmup',type=float,default=15);ap.add_argument('--seed',type=int,default=0);ap.add_argument('--out',type=Path,default=ROOT/'results/hetero/vlm_interference_pilot');ap.add_argument('--control-study',action='store_true');ap.add_argument('--timing-study',action='store_true');ap.add_argument('--profile-study',action='store_true');ap.add_argument('--placement-study',action='store_true');ap.add_argument('--load-sweep',action='store_true');ap.add_argument('--encoder-placement',choices=['with_llm','with_vlm'],default='with_llm');a=ap.parse_args()
    if a.out.exists():raise ValueError('Output exists; preserve before retry')
    a.out.mkdir(parents=True);procs=[];files=[]
    def state(status,**kw):(a.out/'status.json').write_text(json.dumps(dict(status=status,unix_s=time.time(),**kw),indent=2))
    def start(name,cmd,gpu=None):
        env=dict(os.environ,HF_HUB_OFFLINE='1',VLLM_USE_V2_MODEL_RUNNER='0',VLLM_USE_FLASHINFER_SAMPLER='0')
        if gpu:env['CUDA_VISIBLE_DEVICES']=gpu
        f=(a.out/f'{name}.log').open('w');files.append(f)
        p=subprocess.Popen(cmd,env=env,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);procs.append(p);return p
    def ready(port,p):
        for _ in range(210):
            if p.poll() is not None:raise RuntimeError(f'{port} exited {p.returncode}')
            try:
                if httpx.get(f'http://127.0.0.1:{port}/health',timeout=2).status_code==200:return
            except httpx.HTTPError:pass
            time.sleep(2)
        raise RuntimeError(f'{port} readiness timeout')
    try:
        # Pre-materialize identical per-kind payloads for all cases, before the clock.
        state('preparing_requests')
        cases=[('llm_only',1.,0.,0),('vlm_only_low',0.,.5,0),('vlm_only_high',0.,2.,0),('mixed_low',1.,.5,0),('gated_low',1.,.5,1),('mixed_high',1.,2.,0),('gated_high',1.,2.,1)]
        if a.control_study:
            # Match encoder concurrency across fixed/adaptive arms; retain serialized reference.
            cases=[('free',1.,2.,0),('serialized',1.,2.,1),('fixed_pause',1.,2.,2),('adaptive_pause',1.,2.,3)]
        if a.timing_study:
            cases=[('serialized',1.,2.,1),('reactive_gap',1.,2.,3),('prefill_guard',1.,2.,4)]
        if a.profile_study or a.placement_study:
            cases=[('llm_only',1.,0.,0),('vlm_only_high',0.,2.,0),('mixed_high',1.,2.,0)]
        if a.load_sweep:
            cases=[('reference_mix',1.,2.,0),('llm_heavy',4.,.5,0),('vlm_heavy',.25,4.,0),('both_heavy',4.,4.,0)]
        random.Random(a.seed+1013).shuffle(cases)
        plans={name:plan(a.seed,a.warmup+a.duration,lr,vr) for name,lr,vr,g in cases}
        requests=plan(a.seed,a.warmup+a.duration,max(c[1] for c in cases),max(c[2] for c in cases))
        from transformers import AutoTokenizer
        sys.path.insert(0,str(ROOT/'bench/hetero'));from loadgen import make_prompts_lens
        tok=AutoTokenizer.from_pretrained(LLM,local_files_only=True)
        llms=[r for r in requests if r['kind']=='llm'];prompts=make_prompts_lens(tok,[(1024,256)]*len(llms),a.seed)
        payloads={}
        for req,(text,_,_) in zip(llms,prompts):
            payloads[('llm',req['idx'])]=dict(model=LLM,messages=[dict(role='user',content=text)],max_tokens=256,temperature=0,ignore_eos=True,stream=True,stream_options={'include_usage':True})
        for req in requests:
            if req['kind']!='vlm':continue
            rng=np.random.default_rng(a.seed*100000+req['idx']);im=Image.fromarray(rng.integers(0,256,(672,672,3),dtype=np.uint8));buf=io.BytesIO();im.save(buf,format='PNG')
            uri='data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()
            payloads[('vlm',req['idx'])]=dict(model=VLM,messages=[dict(role='user',content=[dict(type='image_url',image_url={'url':uri}),dict(type='text',text='Describe this image.')])],max_tokens=64,temperature=0,ignore_eos=True,stream=True,stream_options={'include_usage':True})
        (a.out/'plans.json').write_text(json.dumps(plans,indent=2));(a.out/'config.json').write_text(json.dumps(dict(seed=a.seed,encoder_placement=a.encoder_placement,gpu_assignment={'encoder':GPU3 if a.encoder_placement=='with_llm' else GPU4,'llm':GPU3,'vlm_language':GPU4},profile_enabled=a.profile_study,cases=cases,models=[LLM,VLM],image_size=672,max_pixels=451584,baseline='all models resident; only selected service receives requests',payload_sha256=hashlib.sha256(json.dumps([(str(k),v) for k,v in payloads.items()]).encode()).hexdigest()),indent=2))
        cache=a.out/'ec_cache';cache.mkdir()
        common=[PY,'-m','vllm.entrypoints.cli.main','serve',VLM,'--dtype','bfloat16','--enforce-eager','--no-enable-prefix-caching','--max-model-len','4096','--max-num-seqs','16','--enable-request-id-headers','--limit-mm-per-prompt','{"image":1}','--mm-processor-cache-gb','0','--mm-processor-kwargs','{"max_pixels":451584}']
        def ec(role):return json.dumps(dict(ec_connector='ECExampleConnector',ec_role=role,ec_connector_extra_config={'shared_storage_path':str(cache)}))
        def profile_args(service):
            if not a.profile_study:return []
            return ['--profiler-config',json.dumps({'profiler':'torch','torch_profiler_dir':str((a.out/'traces'/service).resolve()),'torch_profiler_with_stack':False,'torch_profiler_record_shapes':False,'ignore_frontend':True,'torch_profiler_dump_cuda_time_total':False})]
        state('starting_encoder');e=start('encoder',common+['--port','8510','--mm-encoder-only','--gpu-memory-utilization','.1','--max-num-batched-tokens','8192','--ec-transfer-config',ec('ec_producer')]+profile_args('encoder'),GPU3 if a.encoder_placement=='with_llm' else GPU4);ready(8510,e)
        state('starting_vlm_language');d=start('vlm_language',common+['--port','8520','--enable-mm-embeds','--gpu-memory-utilization','.4','--ec-transfer-config',ec('ec_consumer')],GPU4);ready(8520,d)
        state('starting_llm');l=start('llm',[PY,'-m','vllm.entrypoints.cli.main','serve',LLM,'--port','8540','--dtype','bfloat16','--enforce-eager','--no-enable-prefix-caching','--max-model-len','4096','--gpu-memory-utilization','.4','--max-num-seqs','64']+profile_args('llm'),GPU3);ready(8540,l)
        p=start('proxy',[PY,str(ROOT/'bench/hetero/vlm/gated_proxy.py'),'--out',str(a.out)]);ready(8530,p)
        monitor=start('gpu_monitor',['nvidia-smi','--query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw','--format=csv,noheader','-l','1'])
        results=[]
        for name,lr,vr,g in cases:
            state('running',case=name,completed=len(results),total=len(cases));print('RUN',name,flush=True)
            result=asyncio.run(replay(name,plans[name],payloads,a.out/name,a.warmup,a.duration,g,profile=a.profile_study));results.append(result)
            (a.out/'results.json').write_text(json.dumps(results,indent=2));print(json.dumps(result),flush=True)
            # Clear per-request disk outputs after fully drained case to avoid reuse.
            for f in cache.rglob('*.safetensors'):f.unlink()
            time.sleep(3)
        state('complete',completed=len(results),total=len(cases))
    except BaseException as ex:state('failed',error=str(ex));raise
    finally:
        for p in reversed(procs):
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        for p in procs:
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        for f in files:f.close()

if __name__=='__main__':main()
