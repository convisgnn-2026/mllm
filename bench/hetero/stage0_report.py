#!/usr/bin/env python3
"""Stage 0 functional smoke report: did each candidate load, how much memory, did requests
complete without errors, and are the records instrumented identically. No SLO or cost scoring.

  python bench/hetero/stage0_report.py results/hetero/stage0_smoke
"""
import csv
import datetime
import json
import re
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
# Known-benign vLLM startup noise (optional module import), identical in every run.
BENIGN=('import_utils.py',)


def read_mem(p):
    rows=[]
    for line in p.read_text().strip().splitlines():
        idx,mem=[x.strip() for x in line.split(',')]
        rows.append((int(idx),int(mem.split()[0])))
    return dict(rows)


def scan_log(p,done_hms=None,patterns=('CUDA out of memory','OutOfMemoryError','Traceback','Error','TIMEOUT','failed')):
    """Flag suspicious log lines. Lines stamped at/after the run's `done` time are shutdown noise
    (the runner kills EngineCore first, so APIServer logs EngineDeadError) and are reported separately."""
    if not p.exists():return [],0
    hits={};shutdown=0
    for line in p.read_text(errors='replace').splitlines():
        if not any(pat in line for pat in patterns) or 'kv_load_failure_policy' in line or any(b in line for b in BENIGN):continue
        m=re.search(r'\b(\d\d:\d\d:\d\d)\b',line)
        if done_hms and m and m.group(1)>=done_hms:shutdown+=1;continue
        for pat in patterns:
            if pat in line:hits.setdefault(pat,line.strip()[:160])
    return [f'{k}: {v}' for k,v in hits.items()],shutdown


def main():
    base=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'results/hetero/stage0_smoke'
    runs=[]
    for manifest in sorted(base.glob('*/*/replay/manifest.json')):
        case=manifest.parents[1];cand=case.parent.name
        rep=json.loads(manifest.read_text())
        mem=read_mem(case/'mem_after_load.txt') if (case/'mem_after_load.txt').exists() else {}
        dep=json.loads((case/'deployment.json').read_text()) if (case/'deployment.json').exists() else {}
        cols={}
        for kind in ('llm','dit'):
            f=case/'replay'/kind/'requests.csv'
            if f.exists():
                with f.open() as fh:
                    r=csv.DictReader(fh);cols[kind]=tuple(r.fieldnames);rows=list(r)
                errs={}
                for row in rows:
                    if row['err']:errs[row['err'][:80]]=errs.get(row['err'][:80],0)+1
                rep.setdefault('error_kinds',{})[kind]=errs
                rep.setdefault('send_lag_p99_s',{})[kind]=sorted(float(x['send_lag_s']) for x in rows)[int(.99*(len(rows)-1))] if rows else None
        logs=[];shutdown_noise=0
        done_hms=datetime.datetime.fromtimestamp((case/'done').stat().st_mtime).strftime('%H:%M:%S') if (case/'done').exists() else None
        for name in ('serve_8100.log','serve_8200.log','dit_server.log','proxy.log'):
            h,sd=scan_log(case/name,done_hms);logs+=[f'{name} {x}' for x in h];shutdown_noise+=sd
        peak={};peak_w={}
        if (case/'gpu.csv').exists():
            for row in csv.reader((case/'gpu.csv').open()):
                try:i=int(row[1]);peak[i]=max(peak.get(i,0),int(row[3].split()[0]));peak_w[i]=max(peak_w.get(i,0.),float(row[4].split()[0]))
                except (ValueError,IndexError):continue
        console=case.parent/f'{case.name}.console.log'
        kv={}
        for log in sorted(case.glob('serve_*.log')):
            mm=re.search(r'GPU KV cache size: ([\d,]+) tokens',log.read_text(errors='replace'));kv[log.stem]=int(mm.group(1).replace(',','')) if mm else None
        runs.append(dict(peak_mem_mib=peak,peak_power_w=peak_w,kv_cache_tokens=kv,candidate=cand,plan=case.name,done=(case/'done').exists(),only=rep['only'],
                         requests=rep['requests'],errors=rep['errors'],measured=rep['measured'],
                         error_kinds=rep.get('error_kinds',{}),send_lag_p99_s=rep.get('send_lag_p99_s',{}),
                         mem_after_load_mib=mem,gpu_mem_total_mib=None,deployment=dep,columns={k:list(v) for k,v in cols.items()},
                         plan_sha256=rep['plan_sha256'],prompt_sha256=rep['prompt_sha256'],log_flags=logs,shutdown_noise_lines=shutdown_noise,
                         console_tail=console.read_text(errors='replace').splitlines()[-3:] if console.exists() else []))
    # incomplete / failed cases (no manifest)
    failed=[]
    for case in sorted(base.glob('*/*')):
        if case.is_dir() and not (case/'replay/manifest.json').exists():
            console=case.parent/f'{case.name}.console.log'
            failed.append(dict(candidate=case.parent.name,plan=case.name,
                               console_tail=console.read_text(errors='replace').splitlines()[-8:] if console.exists() else [],
                               log_flags=[f'{n} {h}' for n in ('serve_8100.log','serve_8200.log','dit_server.log') for h in scan_log(case/n)[0]]))
    # instrumentation parity: same columns and same plan/prompt hash per `only`
    parity={}
    for only in ('both','llm','dit'):
        group=[r for r in runs if r['only']==only]
        if not group:continue
        parity[only]=dict(n=len(group),
            same_plan_sha=len({r['plan_sha256'] for r in group})==1,
            same_prompt_sha=len({r['prompt_sha256'] for r in group})==1,
            same_columns=len({json.dumps(r['columns'],sort_keys=True) for r in group})==1,
            same_denominators=len({json.dumps({k:v['n'] for k,v in r['measured'].items()},sort_keys=True) for r in group})==1)
    kv_values=sorted({v for r in runs for v in r['kv_cache_tokens'].values() if v is not None})
    parity['kv_cache_tokens']=dict(distinct_values=kv_values,identical=len(kv_values)<=1)
    out=dict(base=str(base),completed=len(runs),failed=len(failed),runs=runs,failed_runs=failed,parity=parity)
    (base/'stage0_summary.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
    L=['# Stage 0 기능 smoke 결과','',
       '기능 확인용이다. 요청 수가 적고 warmup이 짧아 지연·만족률 값은 결론에 쓰지 않는다. SLO·비용 판정 없음.','',
       '| 후보 | replay | 완료 | 요청 | 오류 | 측정 LLM/DiT n | 로드 후 메모리 MiB (4090 / 3090) | replay 중 최대 MiB (4090 / 3090) | send_lag p99 s | 로그 경고 (종료 잡음 제외) |','|---|---|---|---|---|---|---|---|---|---|']
    for r in runs:
        m=r['measured'];mem=r['mem_after_load_mib']
        lag=r['send_lag_p99_s'];lag_s='/'.join(f"{lag[k]:.3f}" for k in ('llm','dit') if lag.get(k) is not None)
        pk=r['peak_mem_mib']
        L.append(f"| {r['candidate']} | {r['only']} | {'yes' if r['done'] else 'no'} | {r['requests']} | {r['errors']} | {m.get('llm',{}).get('n','-')}/{m.get('dit',{}).get('n','-')} | {mem.get(0,'-')} / {mem.get(1,'-')} | {pk.get(0,'-')} / {pk.get(1,'-')} | {lag_s} | {len(r['log_flags'])} |")
    if failed:
        L+=['','## 실패·미완료 실행','']
        for f in failed:
            L.append(f"- **{f['candidate']} / {f['plan']}**");L+=[f'  - `{x}`' for x in f['log_flags'][:5]];L+=[f'  - console: `{x}`' for x in f['console_tail'][-4:]]
    L+=['','## 계측 일치 검사','','| replay | 실행 수 | plan hash 동일 | prompt hash 동일 | CSV 열 동일 | 분모 동일 |','|---|---|---|---|---|---|']
    for only,p in parity.items():
        if only!='kv_cache_tokens':L.append(f"| {only} | {p['n']} | {p['same_plan_sha']} | {p['same_prompt_sha']} | {p['same_columns']} | {p['same_denominators']} |")
    kvp=parity['kv_cache_tokens']
    L+=['',f"LLM KV cache 용량(토큰) 전체 실행에서 동일: **{kvp['identical']}** (관측값 {kvp['distinct_values']}). 후보·replay 종류가 달라도 같아야 단독·혼합 비교가 성립한다.",'']
    L+=['| 후보 | replay | KV 토큰 (서버별) |','|---|---|---|']+[f"| {r['candidate']} | {r['only']} | {r['kv_cache_tokens']} |" for r in runs]
    noise=sum(r['shutdown_noise_lines'] for r in runs)
    L+=['',f'종료 시점(done 이후) 로그 줄 {noise}개는 runner가 EngineCore를 먼저 종료해 APIServer가 남기는 EngineDeadError 잡음으로 분류했다. 요청 오류와 무관하다.']
    flagged=[r for r in runs if r['log_flags'] or r['errors']]
    if flagged:
        L+=['','## 오류·로그 경고 상세','']
        for r in flagged:
            L.append(f"- **{r['candidate']} ({r['only']})** 오류 {r['errors']} {json.dumps(r['error_kinds'],ensure_ascii=False)}")
            L+=[f'  - `{x}`' for x in r['log_flags']]
    (base/'stage0_summary.md').write_text('\n'.join(L)+'\n')
    print('\n'.join(L))

if __name__=='__main__':main()
