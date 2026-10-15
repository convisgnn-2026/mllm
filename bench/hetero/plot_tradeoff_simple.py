#!/usr/bin/env python3
"""Accessible view of five controlled runs; bar=median seed p90, whisker=range."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/hetero/controlled_runs/figures'
data=json.loads((OUT/'llm_dit_tradeoff_data.json').read_text())
plt.rcParams['font.family']=['Noto Sans CJK KR','DejaVu Sans']
colors=['#267db4','#e98b37']
fig=plt.figure(figsize=(12,7.6),facecolor='white')
fig.text(.5,.95,'같은 GPU 두 장, 작업을 나누는 방법에 따라 기다리는 시간이 달라집니다',ha='center',fontsize=17,weight='bold')
fig.text(.5,.902,'① 텍스트 처리 단계를 나누고, 빠른 GPU에 이미지 생성도 배치',ha='center',fontsize=12,color=colors[0])
fig.text(.5,.865,'④ 텍스트와 이미지가 GPU 한 장씩 따로 사용',ha='center',fontsize=12,color=colors[1])
axes=[fig.add_axes([.08,.34,.39,.40]),fig.add_axes([.58,.34,.36,.40])]
for ax,key,title,subtitle in zip(axes,['ttft','e2e'],['텍스트가 나오기 시작할 때까지','이미지가 완성될 때까지'],['④가 더 빠르게 시작합니다','①이 더 빠르게 완성합니다']):
 vals=[np.array([r[key] for r in data[c]]) for c in ['C_4090','S4_3090']]
 med=np.array([np.median(v) for v in vals]);lo=np.array([v.min() for v in vals]);hi=np.array([v.max() for v in vals])
 ax.bar([0,1],med,color=colors,width=.55,zorder=3)
 ax.errorbar([0,1],med,yerr=np.array([med-lo,hi-med]),fmt='none',ecolor='#444',capsize=6,lw=1.5,zorder=4)
 for i,v in enumerate(med):
  ax.text(i,hi[i]+max(hi)*.07,f'{v:.1f}초',ha='center',fontsize=17,weight='bold',color=colors[i])
 ax.set_xticks([0,1],['① 단계 나눔','④ 서비스별 분리'],fontsize=12)
 ax.set_ylabel('기다린 시간 (초)',fontsize=11)
 ax.set_ylim(0,max(hi)*1.33);ax.set_title(title+'\n'+subtitle,fontsize=14,pad=15)
 ax.grid(axis='y',alpha=.18,zorder=0);ax.set_axisbelow(True)
 ax.spines[['top','right']].set_visible(False)
fig.text(.5,.255,'①은 이미지를 빨리 완성하고, ④는 텍스트를 빨리 시작합니다.',ha='center',fontsize=17,weight='bold')
fig.text(.5,.207,'두 그래프 모두 막대가 짧을수록 좋습니다. 세로축 눈금은 각각 다릅니다.',ha='center',fontsize=11,color='#555')
fig.text(.5,.14,'①  4090: 텍스트 입력 처리 + 이미지 생성  /  3090: 텍스트 생성\n④  4090: 텍스트 입력 처리 + 텍스트 생성  /  3090: 이미지 생성',ha='center',fontsize=11,linespacing=1.8)
fig.text(.5,.06,'막대: 반복 5회에서 얻은 p90의 중앙값. 검은 선: 반복 결과의 최솟값–최댓값(신뢰구간 아님).\np90: 성공 요청의 90%가 이 시간 안에 도달한 값이며, 평균 대기시간이 아닙니다. 각 구성의 오류 1건은 지연 계산에서 제외했습니다.\nLLM 초당 2건 + DiT 초당 0.3건, warmup 60초 + 측정 600초. 배치 비교이며 간섭 원인이나 solver 효과를 단독 입증하지 않습니다.',ha='center',fontsize=8.5,color='#555',linespacing=1.5)
OUT.mkdir(exist_ok=True)
for ext in ['png','svg']:fig.savefig(OUT/f'llm_dit_tradeoff_simple.{ext}',dpi=160)
print(OUT/'llm_dit_tradeoff_simple.png')
