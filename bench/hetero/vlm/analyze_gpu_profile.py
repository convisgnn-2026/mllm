#!/usr/bin/env python3
"""Compare bounded, common profiler windows; timings include profiler overhead."""
import argparse,bisect,collections,gzip,json
from pathlib import Path
import numpy as np

def stats(values):
    a=np.asarray(values,dtype=float)
    return {'n':len(a),'p50':float(np.percentile(a,50)),'p90':float(np.percentile(a,90)),'mean':float(a.mean())} if len(a) else {'n':0}
def merge(intervals):
    out=[]
    for a,b in sorted(intervals):
        if out and a<=out[-1][1]:out[-1]=(out[-1][0],max(b,out[-1][1]))
        else:out.append((a,b))
    return out

def main(root):
    cases=sorted(json.loads((root/'results.json').read_text()),key=lambda c:c['start_unix_s'])
    paths={r:sorted((root/'traces'/r).glob('*.json.gz')) for r in ['llm','encoder']}
    report={};data={}
    for i,case in enumerate(cases):
        name=case['case'];control=json.loads((root/name/'profile_control.json').read_text())
        # Exclude startup and all stop/export activity, especially serial 30s LLM export.
        lo=max(x['after_unix_s'] for x in control if x['command']=='start')+1
        hi=min(x['before_unix_s'] for x in control if x['command']=='stop')
        report[name]={'window_unix_s':[lo,hi],'window_s':hi-lo,'services':{}};data[name]={}
        for role in ['llm','encoder']:
            with gzip.open(paths[role][i],'rt') as h:trace=json.load(h)
            base=trace['baseTimeNanoseconds']/1e9;start=(lo-base)*1e6;end=(hi-base)*1e6
            events=trace['traceEvents'];launch={}
            for e in events:
                if e.get('cat') in ('cuda_runtime','cuda_driver') and 'LaunchKernel' in e.get('name',''):
                    launch[e.get('args',{}).get('correlation')]=(e['ts'],e['ts']+e.get('dur',0))
            kernels=[];groups=collections.defaultdict(list);annotations=collections.defaultdict(list)
            for e in events:
                if e.get('ph')!='X' or not(start<=e.get('ts',-1) and e['ts']+e.get('dur',0)<=end):continue
                if e.get('cat')=='kernel':
                    args=e.get('args',{});l=launch.get(args.get('correlation'))
                    key=(e['name'],str(args.get('grid')),str(args.get('block')),args.get('shared memory'))
                    k={'a':e['ts']-start,'b':e['ts']+e['dur']-start,'dur':e['dur'],'key':key,'launch_end':l[1]-start if l else None,'launch_start':l[0]-start if l else None}
                    kernels.append(k);groups[key].append(e['dur'])
                elif e.get('cat')=='user_annotation' and e['name'].startswith('execute_'):
                    annotations[e['name']].append(e['dur']/1000)
            kernels.sort(key=lambda k:k['a']);merged=merge([(k['a'],k['b']) for k in kernels])
            waits=[max(0,k['a']-k['launch_end']) for k in kernels if k['launch_end'] is not None]
            gaps=[b[0]-a[1] for a,b in zip(merged,merged[1:])]
            report[name]['services'][role]={'trace_file':str(paths[role][i].relative_to(root)),'kernel_us':stats([k['dur'] for k in kernels]),'launch_end_to_kernel_start_us':stats(waits),'between_kernel_busy_intervals_us':stats(gaps),'kernel_busy_fraction':sum(b-a for a,b in merged)/((hi-lo)*1e6),'cpu_execute_ms':{k:stats(v) for k,v in annotations.items()}}
            data[name][role]={'kernels':kernels,'groups':groups,'merged':merged}
            del trace,events,launch
        print('processed',name,flush=True)
    comparisons={}
    for role,solo in [('llm','llm_only'),('encoder','vlm_only_high')]:
        a=data[solo][role]['groups'];b=data['mixed_high'][role]['groups'];matched=[]
        for key in a.keys() & b.keys():
            if min(len(a[key]),len(b[key]))<30:continue
            matched.append({'kernel':key[0],'grid':key[1],'block':key[2],'shared_memory':key[3],'solo_us':stats(a[key]),'mixed_us':stats(b[key]),'median_ratio':float(np.median(b[key])/np.median(a[key]))})
        matched.sort(key=lambda x:x['mixed_us']['n']*x['mixed_us']['mean'],reverse=True)
        comparisons[role]=matched
    enc=data['mixed_high']['encoder']['merged'];starts=[a for a,b in enc];ends=[b for a,b in enc]
    def intersects(a,b):
        i=bisect.bisect_right(ends,a)
        return i<len(enc) and starts[i]<b
    buckets={k:[] for k in ['queue_with_encoder_us','queue_without_encoder_us','kernel_with_encoder_us','kernel_without_encoder_us']}
    llm=data['mixed_high']['llm']['kernels']
    for k in llm:
        tag='with' if intersects(k['a'],k['b']) else 'without';buckets[f'kernel_{tag}_encoder_us'].append(k['dur'])
        if k['launch_end'] is not None:
            a=min(k['a'],k['launch_end']);tag='with' if intersects(a,k['a']) else 'without'
            buckets[f'queue_{tag}_encoder_us'].append(k['a']-a)
    result={'limitations':['Single seed; profiler perturbs CPU launch timing and GPU execution.','Window excludes profiler start (1s guard) and all stop/export; full encoder trace is much longer due to sequential stop.','Matched kernel name/grid/block/shared memory is not exact tensor/batch equivalence.','Launch-end to GPU-start includes own stream queue and cross-process GPU scheduling, not uniquely contention.','No explicit preprocessing or file-transfer spans; cannot quantify their causal share.'], 'cases':report,'matched_kernels':comparisons,'mixed_llm_overlap':{k:stats(v) for k,v in buckets.items()}}
    (root/'gpu_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    for name,c in report.items():
        for role,m in c['services'].items():print(name,role,'kernel',m['kernel_us'],'queue',m['launch_end_to_kernel_start_us'],'busy',m['kernel_busy_fraction'])
    print('overlap',result['mixed_llm_overlap'])
    for role,rows in comparisons.items():
        print('matched',role,len(rows));print([(r['kernel'][:65],round(r['median_ratio'],3),r['solo_us']['p50'],r['mixed_us']['p50']) for r in rows[:8]])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root)
