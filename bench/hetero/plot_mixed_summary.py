#!/usr/bin/env python3
"""Two experiments, four plain-language metrics; do not conflate split types."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
out=ROOT/'results/hetero/mixed_summary';out.mkdir(exist_ok=True)
dit=json.loads((ROOT/'results/hetero/controlled_runs/figures/llm_dit_tradeoff_data.json').read_text())
vlm=json.loads((ROOT/'results/hetero/vlm_placement_seed0/comparison.json').read_text())
plt.rcParams.update({'font.family':'Noto Sans CJK KR','axes.unicode_minus':False})
fig=plt.figure(figsize=(11,9),facecolor='white')
fig.text(.5,.955,'혼합 실행에서 단계 분리가 항상 유리한 것은 아닙니다',ha='center',fontsize=19,weight='bold')
fig.text(.5,.918,'같은 4090 + 3090 두 장 · 막대가 짧을수록 좋음 · 각 그래프의 눈금은 다름',ha='center',fontsize=11,color='#555')
colors=['#4676B5','#E69943'];records=[]
fig.text(.07,.858,'LLM + DiT',fontsize=17,weight='bold')
fig.text(.30,.858,'텍스트는 늦어지고, 이미지는 빨라짐 → 장단점이 함께 존재',fontsize=12)
fig.text(.07,.464,'LLM + VLM',fontsize=17,weight='bold')
fig.text(.30,.464,'Encoder를 LLM과 같은 GPU에 두면 두 서비스 모두 느려짐',fontsize=12)
for idx,(key,title) in enumerate([('ttft','텍스트 첫 응답'),('e2e','이미지 완성')]):
 vals=[[r[key] for r in dit[c]] for c in ['C_4090','S4_3090']]
 med=[float(np.median(v)) for v in vals]
 ax=fig.add_axes([.10+idx*.48,.615,.35,.175])
 ax.barh([1,0],med,color=colors,height=.5)
 for y,v in zip([1,0],med):ax.text(v+max(med)*.03,y,f'{v:.2f}초',va='center',fontsize=13,weight='bold')
 ax.set_yticks([1,0],['단계 분리','서비스별 GPU']);ax.set_xlim(0,max(med)*1.4);ax.set_title(title,fontsize=14);ax.spines[['top','right']].set_visible(False)
 records.append({'experiment':'LLM+DiT','metric':key,'stage_split':med[0],'service_separate':med[1]})
fig.text(.07,.545,'단계 분리: 4090에 LLM 입력 처리 + DiT / 3090에 LLM 생성\n서비스별 GPU: 4090에 LLM / 3090에 DiT',fontsize=10,color='#444',linespacing=1.6)
for idx,(service,key,title) in enumerate([('llm','e2e_s','LLM 답변 완료'),('vlm','ttft_s','VLM 첫 응답')]):
 vals=[vlm[c]['results']['mixed_high']['metrics'][service][key] for c in ['with_llm','with_vlm']]
 ax=fig.add_axes([.10+idx*.48,.22,.35,.175]);ax.barh([1,0],vals,color=colors,height=.5)
 for y,v in zip([1,0],vals):ax.text(v+max(vals)*.03,y,f'{v:.2f}초',va='center',fontsize=13,weight='bold')
 ax.set_yticks([1,0],['단계 분리','서비스별 GPU']);ax.set_xlim(0,max(vals)*1.4);ax.set_title(title,fontsize=14);ax.spines[['top','right']].set_visible(False)
 records.append({'experiment':'LLM+VLM','metric':key,'service':service,'stage_split':vals[0],'service_separate':vals[1]})
fig.text(.07,.15,'단계 분리: 3090에 독립 LLM + VLM encoder / 4090에 VLM 언어 처리\n서비스별 GPU: 3090에 독립 LLM / 4090에 VLM encoder + 언어 처리',fontsize=10,color='#444',linespacing=1.6)
fig.text(.07,.048,'성공 요청 p90. 위: 5개 seed p90의 중앙값, LLM 2건/s + DiT 0.3건/s. 아래: seed 0, LLM 1건/s + VLM 2건/s.\n위는 LLM P/D 분리 배치 비교이며 DiT 단계 분리 효과의 검증이 아님. 아래는 VLM E/(P+D)의 GPU 배치 비교.\n모델·부하·측정 길이가 다른 두 실험이므로 위아래 수치의 직접 비교 및 분리 기법 전체에 대한 일반화는 하지 않음.',fontsize=8.5,color='#666',linespacing=1.5)
for ext in ['png','svg']:fig.savefig(out/f'mixed_split_summary.{ext}',dpi=180)
(out/'plotted_values.json').write_text(json.dumps(records,indent=2)+'\n')
print(out/'mixed_split_summary.png')
