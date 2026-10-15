#!/usr/bin/env python3
"""Two-GPU encoder separation smoke test; not a performance benchmark."""
import base64,json,os,signal,subprocess,time
from pathlib import Path
import httpx
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/hetero/vlm_encoder_smoke'
PY='/home/jinwoo.kim27/venv/jinwoo/bin/python'
MODEL='Qwen/Qwen2-VL-2B-Instruct'
GPUS=['GPU-d5bd27b1-7998-9ae2-b764-3c528e12a47b','GPU-208c8076-e9ae-fde6-aacf-b0c3fae16a9e']

def main():
    if OUT.exists():raise RuntimeError('Output exists; preserve and rename before retry')
    OUT.mkdir(parents=True);cache=OUT/'ec_cache';cache.mkdir()
    procs=[];logs=[]
    def state(status,**kw):(OUT/'status.json').write_text(json.dumps(dict(status=status,unix_s=time.time(),**kw),indent=2))
    def start(name,cmd,gpu=None):
        env=dict(os.environ,HF_HUB_OFFLINE='1',VLLM_USE_FLASHINFER_SAMPLER='0',VLLM_USE_V2_MODEL_RUNNER='0',VLLM_LOGGING_LEVEL='DEBUG')
        if gpu:env['CUDA_VISIBLE_DEVICES']=gpu
        log=(OUT/f'{name}.log').open('w');logs.append(log)
        p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);procs.append(p)
        return p
    def ready(port,p):
        until=time.monotonic()+420
        while time.monotonic()<until:
            if p.poll() is not None:raise RuntimeError(f'process for {port} exited {p.returncode}')
            try:
                if httpx.get(f'http://127.0.0.1:{port}/health',timeout=2).status_code==200:return
            except httpx.HTTPError:pass
            time.sleep(2)
        raise RuntimeError(f'health timeout {port}')
    try:
        common=[PY,'-m','vllm.entrypoints.cli.main','serve',MODEL,'--dtype','bfloat16','--enforce-eager','--no-enable-prefix-caching','--max-model-len','4096','--max-num-seqs','8','--enable-request-id-headers','--limit-mm-per-prompt','{"image":1}','--mm-processor-kwargs','{"max_pixels":262144}']
        def ec(role):return json.dumps(dict(ec_connector='ECExampleConnector',ec_role=role,ec_connector_extra_config=dict(shared_storage_path=str(cache))))
        state('starting_encoder')
        e=start('encoder',common+['--port','8510','--mm-encoder-only','--gpu-memory-utilization','.1','--max-num-batched-tokens','8192','--ec-transfer-config',ec('ec_producer')],GPUS[1])
        ready(8510,e)
        state('starting_language')
        d=start('language',common+['--enable-mm-embeds','--port','8520','--gpu-memory-utilization','.4','--ec-transfer-config',ec('ec_consumer')],GPUS[0]);ready(8520,d)
        proxy=start('proxy',[PY,str(ROOT/'bench/hetero/vlm/disagg_epd_proxy.py'),'--host','127.0.0.1','--port','8530','--encode-servers-urls','http://127.0.0.1:8510','--prefill-servers-urls','disable','--decode-servers-urls','http://127.0.0.1:8520']);ready(8530,proxy)
        state('testing');results=[]
        for color in ('red','blue'):
            image=Image.new('RGB',(336,336),'white');draw=ImageDraw.Draw(image);draw.rectangle((60,60,276,276),fill=color)
            path=OUT/f'{color}.png';image.save(path)
            uri='data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()
            body=dict(model=MODEL,max_tokens=32,temperature=0,messages=[dict(role='user',content=[dict(type='image_url',image_url=dict(url=uri)),dict(type='text',text='Describe the shape and its color in this image.')])])
            t=time.perf_counter();response=httpx.post('http://127.0.0.1:8530/v1/chat/completions',json=body,timeout=120);response.raise_for_status()
            item=response.json();results.append(dict(image=color,e2e_s=time.perf_counter()-t,response=item))
            (OUT/f'{color}_response.json').write_text(json.dumps(results[-1],indent=2))
        mem=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv'],text=True)
        (OUT/'gpu_processes.csv').write_text(mem)
        assert list(cache.rglob('*.safetensors')), 'No encoded cache file: separation unproven'
        state('complete',model=MODEL,encoder_gpu=GPUS[1],language_gpu=GPUS[0],connector='ECExampleConnector',cache_files=len(list(cache.rglob('*.safetensors'))),results=results)
        print('SMOKE COMPLETE',flush=True)
    except BaseException as ex:state('failed',error=str(ex));raise
    finally:
        for p in reversed(procs):
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        for p in procs:
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        for log in logs:log.close()

if __name__=='__main__':main()
