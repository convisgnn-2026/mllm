#!/usr/bin/env python3
"""Encoder admission gate around the unmodified official vLLM proxy."""
import argparse,asyncio,json,time
from pathlib import Path
import uvicorn
import disagg_epd_proxy as proxy

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    gate=asyncio.Semaphore(1);limit=0;original=proxy.fanout_encoder_primer
    last_finish=0.;last_slow=0.;prefill_active=0;prefill_signal_time=0.
    async def primer(request,urls,req_id):
        global last_finish
        t=time.perf_counter();started=None;error='';policy_wait=0.
        try:
            if limit:
                async with gate:
                    if limit == 2:
                        delay=max(0.,last_finish+.020-time.perf_counter())
                        await asyncio.sleep(delay);policy_wait=delay
                    elif limit == 3:
                        # Yield while recent observed LLM chunk gaps are slow, capped per admission.
                        wait_start=time.perf_counter()
                        while time.perf_counter()-last_slow < .100 and time.perf_counter()-wait_start < .100:
                            await asyncio.sleep(.005)
                        policy_wait=time.perf_counter()-wait_start
                    elif limit == 4:
                        wait_start=time.perf_counter()
                        while prefill_active > 0 and time.perf_counter()-prefill_signal_time < .300 and time.perf_counter()-wait_start < .100:
                            await asyncio.sleep(.005)
                        policy_wait=time.perf_counter()-wait_start
                    started=time.perf_counter()
                    try:return await original(request,urls,req_id)
                    finally:last_finish=time.perf_counter()
            started=time.perf_counter();return await original(request,urls,req_id)
        except BaseException as ex:error=str(ex);raise
        finally:
            end=time.perf_counter()
            with (a.out/'encoder_admission.jsonl').open('a') as f:
                f.write(json.dumps(dict(request_id=req_id,arrival_unix_s=time.time()-(end-t),gate_limit=limit,gate_wait_s=(started-t) if started else None,primer_s=(end-started) if started else None,policy_wait_s=policy_wait,prefill_active=prefill_active,error=error))+'\n')
    proxy.fanout_encoder_primer=primer
    @proxy.app.post('/experiment/gate/{value}')
    async def set_gate(value:int):
        global limit
        if value not in (0,1,2,3,4):return {'error':'supported limits: 0, 1, 2, 3, 4'}
        global last_slow,last_finish,prefill_active,prefill_signal_time
        last_slow=0.;last_finish=0.;prefill_active=0;prefill_signal_time=0.;limit=value;return {'limit':limit}
    @proxy.app.post('/experiment/feedback')
    async def feedback(request:proxy.Request):
        global last_slow,prefill_active,prefill_signal_time
        body=await request.json()
        prefill_active=body.get('prefill_active',0);prefill_signal_time=time.perf_counter()
        # Client arrival feedback, not GPU kernel telemetry.
        if body.get('max_gap_s',0) > .025:last_slow=time.perf_counter()
        with (a.out/'feedback.jsonl').open('a') as f:
            f.write(json.dumps(dict(unix_s=time.time(),**body))+'\n')
        return {'ok':True}
    proxy.app.state.e_urls=['http://127.0.0.1:8510'];proxy.app.state.p_urls=[];proxy.app.state.d_urls=['http://127.0.0.1:8520']
    uvicorn.run(proxy.app,host='127.0.0.1',port=8530,log_level='info')
