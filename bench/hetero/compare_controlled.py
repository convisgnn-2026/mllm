#!/usr/bin/env python3
"""Summarize completed paired controlled runs; reject incomparable plans."""
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'results/hetero/controlled_runs'


def summarize():
    lines=['# 통제된 GPU 비교 실험', '',
           'LLM 2 rps + DiT 0.3 rps. seed별 공통 계획, warmup 60초 + 측정 600초. A/B는 탐색 deadline이며 실제 서비스 요구로 확정한 값이 아니다.', '',
           '- A: TTFT≤0.5s, 요청 평균 TPOT≤20ms, DiT E2E≤10s.',
           '- B: TTFT≤1s, 요청 평균 TPOT≤12ms, DiT E2E≤3s.',
           '- 각 서비스 만족률 90% 이상일 때 해당 구성이 통과한다. 오류·미완료 포함.',
           '- TTFT/E2E는 계획 도착 기준이며 client 송신 지연을 포함한다.', '',
           '| seed | 구성 | LLM/DiT 표본 | LLM/DiT 오류 | A 만족률 LLM/DiT | B 만족률 LLM/DiT | A/B 통과 |',
           '|---|---|---|---|---|---|---|']
    paired=[];count=0
    for seed in range(5):
        pair={}
        for cfg in ('C_4090','S4_3090'):
            p=BASE/cfg/f'seed{seed}'/'replay/manifest.json'
            if not p.exists():continue
            report=json.loads(p.read_text());pair[cfg]=report;count+=1
            l,d=report['measured']['llm'],report['measured']['dit']
            passes=[all(report['measured'][k]['attainment'][name] is not None and report['measured'][k]['attainment'][name]>=.9 for k in ('llm','dit')) for name in ('A','B')]
            lines.append(f"| {seed} | {cfg} | {l['n']}/{d['n']} | {l['errors']}/{d['errors']} | {l['attainment']['A']:.1%}/{d['attainment']['A']:.1%} | {l['attainment']['B']:.1%}/{d['attainment']['B']:.1%} | {passes[0]}/{passes[1]} |")
        if len(pair)==2:
            a,b=pair.values()
            for key in ('plan_sha256','prompt_sha256','model','warmup_s','duration_s','only'):
                if a[key]!=b[key]:raise ValueError(f'seed {seed}: mismatched {key}')
            for kind in ('llm','dit'):
                if a['measured'][kind]['n']!=b['measured'][kind]['n']:raise ValueError('mismatched denominator')
            paired.append(dict(seed=seed,results=pair))
    lines+=['',f'완료 실행 {count}/10, 비교 가능한 seed pair {len(paired)}/5.', '',
            '## 해석', '',
            '수요·길이 분포가 pilot과 달라 기존 만족률과 직접 비교하지 않는다. 모든 seed에서 재현되는지 먼저 확인한다. deadline 하나의 경계 통과만으로 우위를 확정하지 않는다. 이 비교는 정적 구성 선택을 조사하며 온라인 solver·전환 비용·구매비 절감을 입증하지 않는다.']
    BASE.mkdir(parents=True,exist_ok=True)
    (BASE/'comparison.md').write_text('\n'.join(lines)+'\n')
    (BASE/'comparison.json').write_text(json.dumps(dict(completed_runs=count,paired_seeds=len(paired),pairs=paired),indent=2)+'\n')
    return count

if __name__=='__main__':print('Completed runs:',summarize())
