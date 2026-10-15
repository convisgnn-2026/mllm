#!/usr/bin/env python3
"""Measured latency trade-off; dots are independent seed runs, not error bars."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'results/hetero/controlled_runs'
OUT=BASE/'figures';OUT.mkdir(exist_ok=True)
plt.rcParams['font.family']=['Noto Sans CJK KR','DejaVu Sans']
configs=[('C_4090','PD 분리 · DiT 4090','#2378bc'),('S4_3090','서비스별 GPU 분리','#e47d27')]
data={}
for cfg,_,_ in configs:
 data[cfg]=[]
 for seed in range(5):
  vals={}
  for kind in ('llm','dit'):
   with (BASE/cfg/f'seed{seed}/replay'/kind/'requests.csv').open() as f:
    rows=[r for r in csv.DictReader(f) if r['measured']=='True']
   ok=[r for r in rows if not r['err']]
   for key in (('ttft','tpot') if kind=='llm' else ('e2e',)):
    vals[key]=float(np.percentile([float(r[key]) for r in ok],90))
   vals[kind+'_errors']=len(rows)-len(ok)
  vals['seed']=seed;data[cfg].append(vals)
fig,axes=plt.subplots(1,2,figsize=(13,6.7))
for ax,key,xlabel in zip(axes,('ttft','tpot'),('LLM 첫 응답 지연 p90 (ms)','LLM 요청 평균 TPOT p90 (ms)')):
 for cfg,label,color in configs:
  d=data[cfg];x=np.array([r[key]*1000 for r in d]);y=np.array([r['e2e'] for r in d])
  ax.scatter(x,y,color=color,s=100,alpha=.85,label=label,zorder=3)
  for xx,yy,r in zip(x,y,d):ax.annotate(f"s{r['seed']}",(xx,yy),xytext=(6,5+12*r['seed'] if cfg=='C_4090' else 5),textcoords='offset points',fontsize=9,color=color)
 ax.set_xlabel(xlabel,fontsize=12);ax.set_ylabel('DiT 이미지 완료 지연 p90 (s)',fontsize=12)
 ax.set_xlim(0,max(r[key]*1000 for vals in data.values() for r in vals)*1.15);ax.set_ylim(bottom=0);ax.grid(alpha=.18)
 ax.text(.02,.98,'← LLM 지연 감소\n↓ DiT 지연 감소',transform=ax.transAxes,va='top',fontsize=10,color='#555')
axes[0].set_title('첫 응답과 이미지 완료 사이의 trade-off',fontsize=13)
axes[1].set_title('토큰 생성 지연도 함께 확인',fontsize=13)
axes[0].legend(loc='upper right',fontsize=10)
fig.suptitle('동일한 두 GPU에서도 배치에 따라 LLM과 DiT가 얻는 이점이 달라진다',fontsize=16,y=.98)
fig.text(.5,.915,'4090 + 3090  |  LLM 2 rps + DiT 0.3 rps  |  점 하나 = seed 한 번의 측정 (총 5쌍)',ha='center',fontsize=11)
fig.text(.5,.06,'PD 분리 · DiT 4090: 4090에 prefill+DiT, 3090에 decode\n서비스별 GPU 분리: 4090에 LLM 전체, 3090에 DiT 전체',ha='center',fontsize=10)
fig.text(.5,.015,'warmup 60초 + 측정 600초. TTFT/E2E는 계획 도착 기준. 지연 p90은 성공 요청만 사용; 각 배치 seed 0의 LLM 오류 1건은 제외.\n점 간 변동은 seed별 차이이며 신뢰구간이 아니다. 현재 두 배치의 관측 비교이며 solver 개선 결과가 아니다.',ha='center',fontsize=8.5,color='#555')
fig.tight_layout(rect=(0,.12,1,.88))
path=OUT/'llm_dit_tradeoff.png';fig.savefig(path,dpi=170)
fig.savefig(OUT/'llm_dit_tradeoff.svg')
(OUT/'llm_dit_tradeoff_data.json').write_text(json.dumps(data,indent=2)+'\n')
print(path)
for cfg,label,_ in configs:
 d=data[cfg];print(label, {k:[round(min(r[k] for r in d),3),round(max(r[k] for r in d),3)] for k in ('ttft','tpot','e2e')})
