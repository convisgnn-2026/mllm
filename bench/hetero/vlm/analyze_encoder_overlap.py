#!/usr/bin/env python3
"""Request-level primer overlap association; does not identify GPU kernel causality."""
import argparse,json
from pathlib import Path
import numpy as np

def analyze(root):
    events=[json.loads(line) for line in (root/'encoder_admission.jsonl').read_text().splitlines()]
    report={}
    for folder in sorted(root.iterdir()):
        source=folder/'requests.live.jsonl'
        summary=folder/'summary.json'
        if not source.exists() or not summary.exists():continue
        meta=json.loads(summary.read_text());epoch=meta['start_unix_s'];pairs=[];inside=[];outside=[]
        for line in source.read_text().splitlines():
            row=json.loads(line)
            if row['kind']!='llm' or not row['measured'] or row['error']:continue
            start=epoch+row['send_s'];end=epoch+row['end_s'];intervals=[]
            for event in events:
                if event.get('gate_wait_s') is None or event.get('primer_s') is None:continue
                a=event['arrival_unix_s']+event['gate_wait_s'];b=a+event['primer_s']
                if a<end and b>start:intervals.append((max(a,start),min(b,end)))
            merged=[]
            for a,b in sorted(intervals):
                if merged and a<=merged[-1][1]:merged[-1]=(merged[-1][0],max(b,merged[-1][1]))
                else:merged.append((a,b))
            fraction=sum(b-a for a,b in merged)/max(end-start,1e-9)
            pairs.append((fraction,row['chunk_gap_p90_s']))
            chunks=row.get('chunk_times_s',[])
            for a,b in zip(chunks,chunks[1:]):
                bucket=inside if any(x<epoch+b and y>epoch+a for x,y in merged) else outside
                bucket.append(b-a)
        xs=[x for x,y in pairs if y is not None];ys=[y for x,y in pairs if y is not None]
        corr=float(np.corrcoef(xs,ys)[0,1]) if len(xs)>2 and np.std(xs)>0 and np.std(ys)>0 else None
        report[folder.name]={'llm_requests':len(pairs),'mean_primer_overlap_fraction':float(np.mean(xs)) if xs else None,'request_overlap_vs_chunk_gap_pearson':corr,'gap_overlapping_primer_p90_s':float(np.percentile(inside,90)) if inside else None,'gap_outside_primer_p90_s':float(np.percentile(outside,90)) if outside else None,'gaps_overlapping':len(inside),'gaps_outside':len(outside)}
    (root/'overlap_analysis.json').write_text(json.dumps({'limitation':'Primer includes preprocessing, HTTP and file transfer. Association is not GPU causality. Old pilot lacks chunk timestamps; only request-level association available.', 'cases':report},indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);analyze(p.parse_args().root)
