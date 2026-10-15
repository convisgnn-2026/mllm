#!/usr/bin/env python3
"""Align client feedback reception with primer HTTP intervals, not GPU kernels."""
import argparse,json
from pathlib import Path
import numpy as np

def analyze(root):
    events=[json.loads(x) for x in (root/'encoder_admission.jsonl').read_text().splitlines()]
    feedback=[json.loads(x) for x in (root/'feedback.jsonl').read_text().splitlines()]
    result={}
    for folder in sorted(root.iterdir()):
        if not (folder/'summary.json').exists():continue
        case=json.loads((folder/'summary.json').read_text());start=case['start_unix_s']+case['warmup_s'];end=start+case['duration_s']
        primers=[]
        for event in events:
            if event.get('gate_wait_s') is None or event.get('primer_s') is None:continue
            a=event['arrival_unix_s']+event['gate_wait_s'];b=a+event['primer_s']
            if start<=a<end:primers.append((a,b,event))
        slow=[f for f in feedback if start<=f['unix_s']<end and f['max_gap_s']>.025]
        during=[f for f in slow if any(a<=f['unix_s']<=b for a,b,e in primers)]
        covered=sum(any(start<=f['unix_s']<=a and a-f['unix_s']<.1 for f in slow) for a,b,e in primers)
        pending=sum(any(start<=f['unix_s']<=a and a-f['unix_s']<.3 and f.get('prefill_active',0)>0 for f in feedback) for a,b,e in primers)
        waits=[e.get('policy_wait_s',0) for a,b,e in primers]
        latencies=[f['unix_s']-f['source_unix_s'] for f in feedback if start<=f['unix_s']<end and 'source_unix_s' in f]
        result[folder.name]={'primers_started_in_measurement':len(primers),'slow_feedback_events':len(slow),'slow_events_received_during_primer':len(during),'primers_with_recent_slow_feedback':covered,'primers_with_recent_pending_first_response_feedback':pending,'policy_wait_over_1ms_count':sum(x>.001 for x in waits),'policy_wait_p90_s':float(np.percentile(waits,90)) if waits else None,'feedback_transit_p90_s':float(np.percentile(latencies,90)) if latencies else None}
    (root/'signal_timing_analysis.json').write_text(json.dumps({'limitation':'Signals are client chunk arrivals. Pending first response includes queue, network and prefill. Primer includes CPU/file/HTTP. These timings do not isolate GPU kernels or prove causality. Feedback buckets overlap in time.', 'cases':result},indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);analyze(p.parse_args().root)
