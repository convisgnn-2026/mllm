#!/usr/bin/env python3
"""Download the public model and perform a GPU encoder separation smoke test."""
import json,subprocess,sys,time,os
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from pathlib import Path
from huggingface_hub import snapshot_download
ROOT=Path(__file__).resolve().parents[3]
STATUS=ROOT/'results/hetero/vlm_encoder_check_status.json'
def status(state,**kw):
    STATUS.parent.mkdir(parents=True,exist_ok=True)
    STATUS.write_text(json.dumps(dict(status=state,unix_s=time.time(),**kw),indent=2))
try:
    status('downloading',model='Qwen/Qwen2-VL-2B-Instruct')
    model=snapshot_download('Qwen/Qwen2-VL-2B-Instruct')
    status('smoke_running',model_path=model)
    subprocess.run([sys.executable,str(ROOT/'bench/hetero/vlm/smoke_encoder_split.py')],check=True,cwd=ROOT)
    status('complete',result=str(ROOT/'results/hetero/vlm_encoder_smoke/status.json'))
except BaseException as ex:
    status('failed',error=str(ex));raise
