#!/usr/bin/env python3
"""Readable version of k_timeshare.png: measured k (x) vs k computed from the time-slicing rule (y), linear axes,
one panel per metric, filled = unsaturated (model applies), hollow = saturated (model says 'do not trust'); points
with measured k > axis limit are drawn at the right edge with '>' markers.
  python3 bench/hetero/plot_k_timeshare_explained.py -> results/hetero/k_model/k_timeshare_explained.png
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import font_manager
ROOT = Path(__file__).resolve().parents[2]
for f in ('NanumGothic', 'Noto Sans CJK KR', 'Noto Sans KR', 'Malgun Gothic'):
    if any(f in x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams['font.family'] = f; break
plt.rcParams['axes.unicode_minus'] = False
pts = json.loads((ROOT / 'results/hetero/k_model/k_timeshare.json').read_text())['points']
LAB = dict(ttft='TTFT', tpot='TPOT', dit='DiT E2E'); LIM = dict(ttft=5.0, tpot=4.0, dit=4.0)
COL = {'S4_4090': '#1f77b4', 'S3_3090': '#ff7f0e', 'C_4090': '#9467bd', 'C_3090': '#8c564b', 'C_split': '#e377c2'}
NAME = {'S4_4090': 'S4_4090 (단일 엔진+DiT, 4090)', 'S3_3090': 'S3_3090 (단일 엔진+DiT, 3090)', 'C_4090': 'C_4090 (P/D, DiT가 prefill GPU)',
        'C_3090': 'C_3090 (P/D, DiT가 decode GPU)', 'C_split': 'C_split (P/D, denoise 4090 / enc·VAE 3090)'}
fig, axs = plt.subplots(1, 3, figsize=(12, 4.2))
for ax, m in zip(axs, ('ttft', 'tpot', 'dit')):
    sel = [p for p in pts if p['m'] == m]; L = LIM[m]
    ax.fill_between([1, L], [1 / 1.25, L / 1.25], [1.25, L * 1.25], color='0.9', lw=0, label='±25%')
    ax.plot([1, L], [1, L], 'k-', lw=0.8)
    over = 0
    for p in sel:
        x = min(p['k'], L * 0.98); y = p['TS3']; c = COL[p['c']]
        if p['k'] > L:
            ax.plot(x, y, marker='>', color=c, ms=8, ls=''); over += 1
        elif p['sat']:
            ax.plot(x, y, 'o', color=c, mfc='white', ms=7, mew=1.5)
        else:
            ax.plot(x, y, 'o', color=c, ms=7)
            ax.annotate(f"α{p['al']}", (x, y), textcoords='offset points', xytext=(5, 4), fontsize=7, color=c)
    uns = [p for p in sel if not p['sat']]; e = sorted(abs(p['TS3'] / p['k'] - 1) for p in uns)
    ax.set_xlim(1, L); ax.set_ylim(1, L); ax.set_xlabel('실측 k (혼합 p90 / 단독 p90)'); ax.set_aspect('equal')
    ax.set_title(f"{LAB[m]}\n비포화 {len(uns)}점: 중앙값 오차 {e[len(e)//2]:.0%}, 25% 이내 {sum(x<=0.25 for x in e)}/{len(uns)}" + (f"\n(오른쪽 끝 '>' {over}점은 실측 k > {L:g}, 포화)" if over else ''), fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
axs[0].set_ylabel('계산한 k (시간 분할 규칙, 맞춘 매개변수 없음)')
h = [Line2D([], [], color=COL[c], marker='o', ls='', label=NAME[c]) for c in COL] + [
    Line2D([], [], color='k', marker='o', ls='', label='비포화: 모델 적용 (채움)'), Line2D([], [], color='k', marker='o', mfc='white', ls='', label='포화 판정: 모델 "불신" (빈 원)'),
    Line2D([], [], color='k', marker='>', ls='', label='실측 k가 축 밖 (포화)'), plt.Rectangle((0, 0), 1, 1, color='0.9', label='±25% 구간')]
fig.legend(handles=h, loc='lower center', ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.22))
fig.suptitle('대각선 위 = 계산값이 실측과 같음.  채운 점이 회색 띠 안에 있으면 "측정 없이 ±25% 안에 맞춘 것"', fontsize=10)
fig.tight_layout(rect=(0, 0.0, 1, 0.97)); out = ROOT / 'results/hetero/k_model/k_timeshare_explained.png'
fig.savefig(out, dpi=180, bbox_inches='tight'); print(out)
