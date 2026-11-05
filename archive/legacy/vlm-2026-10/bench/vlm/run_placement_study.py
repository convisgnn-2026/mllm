#!/usr/bin/env python3
"""Two fixed-resource placements, each with resident solo references; one seed."""
import argparse,fcntl,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--load-sweep',action='store_true');a=ap.parse_args()
    out=ROOT/'results/hetero'/('vlm_placement_load_sweep_seed0' if a.load_sweep else 'vlm_placement_seed0')
    if out.exists():raise RuntimeError('Preserve existing output; no overwrite')
    lock=(ROOT/'results/hetero/vlm_placement.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out.mkdir()
    def state(status,**kw):
        temporary=out/'status.tmp';temporary.write_text(json.dumps(dict(status=status,unix_s=time.time(),**kw),indent=2));temporary.replace(out/'status.json')
    try:
        state('preflight')
        check=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        if check:raise RuntimeError('GPU compute processes already present: '+check)
        placements=['with_llm','with_vlm'] if a.load_sweep else ['with_vlm','with_llm']
        for i,placement in enumerate(placements):
            state('running',placement=placement,completed_placements=i,total_placements=2,child_status=str(out/placement/'status.json'))
            with (out/f'{placement}.runner.log').open('w') as f:
                subprocess.run([sys.executable,'-u',str(ROOT/'bench/hetero/vlm/run_interference_pilot.py'),'--load-sweep' if a.load_sweep else '--placement-study','--encoder-placement',placement,'--seed','0','--duration','60' if a.load_sweep else '90','--out',str(out/placement)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
        configs=[json.loads((out/p/'config.json').read_text()) for p in placements]
        assert configs[0]['payload_sha256']==configs[1]['payload_sha256'],'Payload mismatch'
        assert json.loads((out/placements[0]/'plans.json').read_text())==json.loads((out/placements[1]/'plans.json').read_text()),'Plan mismatch'
        summary={}
        for placement in placements:
            rows={x['case']:x for x in json.loads((out/placement/'results.json').read_text())}
            summary[placement]={'results':rows,'mixed_over_solo':{}}
            for service,solo in ([] if a.load_sweep else [('llm','llm_only'),('vlm','vlm_only_high')]):
                a=rows[solo]['metrics'][service];b=rows['mixed_high']['metrics'][service]
                summary[placement]['mixed_over_solo'][service]={k:b[k]/a[k] if a.get(k) and b.get(k) is not None else None for k in ['ttft_s','e2e_s','tpot_s','chunk_gap_p90_s']}
        (out/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
        state('complete',completed_placements=2,total_placements=2,total_cases=8 if a.load_sweep else 6)
    except BaseException as ex:state('failed',error=str(ex));raise
if __name__=='__main__':main()
