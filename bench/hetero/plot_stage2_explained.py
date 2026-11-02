#!/usr/bin/env python3
"""Presentation version of fig_stage2_selectors with self-explanatory Korean labels.

  python3 bench/hetero/plot_stage2_explained.py [--summary results/hetero/stage2_report/stage2_summary.json]
                                                [--out results/hetero/stage2_report/fig_stage2_selectors_explained]
Reads stage2_summary.json only; the numbers are those of stage2_report.py.
"""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NAME = {
    'solo': '단독 결과만\n(간섭 무시)',
    'solo+k (LOAO)': '단독 × 간섭 배율\n(다른 α의 배율)',
    'solo+k (seed 901)': '단독 × 간섭 배율\n(seed 901 측정)',
    'mixed seed 901': 'seed 901 혼합\n결과 재사용',
    'oracle': '정답지 보고\n선택 (상한)',
    'fixed:C_4090': '항상 PD 분할\n(고정)',
    'fixed:S4_3090': '항상 LLM 4090\n/ DiT 3090 (고정)',
    'solo only': '단독 결과만\n(간섭 무시)',
    'solo+k-const (2 anchors)': '단독 × 배율\n(앵커 2개, 상수)',
    'solo+k-linear (2 anchors)': '단독 × 배율\n(앵커 2개, 선형 외삽)\n= solver',
    'solo+k-measured (same alpha)': '단독 × 배율\n(같은 α에서 측정)',
    'fixed:S4_4090': '항상 4090 한 장\n(고정)',
    'fixed:S3_4090': '항상 LLM 3090\n/ DiT 4090 (고정)',
}
GROUP = {'solo': 'pred', 'solo+k (LOAO)': 'pred', 'solo+k (seed 901)': 'pred', 'mixed seed 901': 'pred', 'oracle': 'oracle', 'fixed:C_4090': 'fixed', 'fixed:S4_3090': 'fixed',
         'solo only': 'pred', 'solo+k-const (2 anchors)': 'pred', 'solo+k-linear (2 anchors)': 'solver', 'solo+k-measured (same alpha)': 'pred', 'fixed:S4_4090': 'fixed', 'fixed:S3_4090': 'fixed'}
GCOL = {'pred': '#0072B2', 'solver': '#D55E00', 'oracle': '#000000', 'fixed': '#7f7f7f'}
COL_TITLE = {'seed 901 (profiling)': 'seed 901: 프로파일을 만든 날\n(같은 데이터로 자기 채점)',
             'stage1_seed902': 'seed 902: 처음 보는 날 1\n(요청 순서·길이가 다름)',
             'stage1_seed903': 'seed 903: 처음 보는 날 2\n(실제 부하가 12% 더 무거움)',
             'pooled': 'seed 902 + 903 합산\n(처음 보는 두 날)'}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--summary', type=Path, default=ROOT / 'results/hetero/stage2_report/stage2_summary.json')
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--stage2b', action='store_true', help='summary is stage2b_summary.json: held-out alphas, pooled + per seed')
    a = ap.parse_args()
    if a.out is None:
        a.out = a.summary.parent / ('fig_stage2b_selectors_explained' if a.stage2b else 'fig_stage2_selectors_explained')
    if any('Noto Sans CJK KR' in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams['font.family'] = 'Noto Sans CJK KR'
    plt.rcParams['axes.unicode_minus'] = False
    S = json.loads(a.summary.read_text())
    if a.stage2b:
        keep = ['solo only', 'solo+k-linear (2 anchors)', 'solo+k-measured (same alpha)', 'oracle', 'fixed:S4_4090', 'fixed:C_4090', 'fixed:S4_3090']
        sub = lambda r: {n: r[n] for n in keep if n in r}
        blocks = [('pooled', sub(S['pooled']['held_out']))] + [(d, sub(v['result']['held_out'])) for d, v in S['unseen'].items()]
    else:
        blocks = [('seed 901 (profiling)', S['selectors_profiling_seed'])] + [(d, v['result']) for d, v in S['selectors_unseen'].items()]
    blocks = [b for b in blocks if b[1]]
    fig, axs = plt.subplots(2, len(blocks), figsize=(6.0 * len(blocks), 8.4), squeeze=False)
    for j, (title, res) in enumerate(blocks):
        names = list(res); x = np.arange(len(names))
        cols = [GCOL[GROUP[n]] for n in names]
        # top: max-alpha
        ax = axs[0, j]
        share = [res[n]['exact'] / max(res[n]['cells'], 1) for n in names]
        ax.bar(x, share, color=cols)
        for i, (n, s) in enumerate(zip(names, share)):
            ax.text(i, s + .02, f"{res[n]['exact']}/{res[n]['cells']}", ha='center', fontsize=7)
        ax.set_ylim(0, 1.15); ax.set_ylabel('정답과 같은 배치를 고른 비율', fontsize=8)
        ax.set_title(COL_TITLE.get(title, title) + '\n\n[위] SLO 조건마다 "가장 높은 수요까지 버틸 배치" 고르기', fontsize=9)
        ax.set_xticks(x, [NAME.get(n, n) for n in names], fontsize=7, rotation=20, ha='right')
        for t, n in zip(ax.get_xticklabels(), names):
            t.set_color(GCOL[GROUP[n]])
        ax.spines[['top', 'right']].set_visible(False)
        # bottom: min-cost
        ax = axs[1, j]
        bottom = np.zeros(len(names))
        for k, col, lab in (('mc_ok', '#009E73', '지켰고 가장 쌈'), ('mc_overpay', '#E69F00', '지켰지만 더 싼 배치도 있었음 (과지불)'),
                            ('mc_violation', '#D55E00', '실제로는 SLO 못 지킴 (위반)'), ('mc_missed', '#bbbbbb', '지킬 배치가 있는데 "없다"고 답함 (놓침)')):
            v = np.array([res[n][k] - (res[n]['mc_overpay'] if k == 'mc_ok' else 0) for n in names], float)
            ax.bar(x, v, bottom=bottom, color=col, label=lab if j == 0 else None)
            if k == 'mc_violation':
                for i, val in enumerate(v):
                    if val > 0:
                        ax.text(i, bottom[i] + val / 2, f'{int(val)}', ha='center', va='center', fontsize=7, color='white')
            bottom += v
        n_cases = res[names[0]]['mc_cases']
        ax.set_title(f'[아래] SLO 조건·수요마다 "가장 싼 배치" 고르기, 실제 혼합 실행으로 채점\n채점 대상 {n_cases}쌍 = 지킬 수 있는 배치가 하나라도 있는 (조건, α)', fontsize=9)
        ax.set_ylabel('(SLO 조건, α) 쌍 수', fontsize=8)
        ax.set_xticks(x, [NAME.get(n, n) for n in names], fontsize=7, rotation=20, ha='right')
        for t, n in zip(ax.get_xticklabels(), names):
            t.set_color(GCOL[GROUP[n]])
        ax.spines[['top', 'right']].set_visible(False)
    h, l = axs[1, 0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, frameon=False, loc='lower center', bbox_to_anchor=(0.5, 0.075), ncol=4, title='아래 행 채점 결과 (막대 안 숫자 = 위반 건수)', title_fontsize=8)
    if a.stage2b:
        foot = ('글자색: 주황 = solver(단독 프로파일 + 앵커 α 0.5·1.0에서만 잰 간섭 배율을 선형 외삽), 파랑 = 비교용 예측기, 검정 = 정답지(오라클), 회색 = 정보 없이 배치 하나로 고정.   '
                '비용: 4090 한 장 = 1.0, 두 장 = 1.75.\n'
                '평가 α = 0.6·0.75·1.1·1.25: 배율을 잰 α도 아니고 프로파일을 만든 seed도 아닌, solver가 처음 보는 조건. '
                '서비스를 다른 카드에 두는 배치는 배율 1(측정값 0.99~1.07)로 두고 seed 901 혼합 결과를 프로파일로 씀. 프로파일이 없는 α는 바로 위 α의 것을 씀(보수적).')
        title = 'Solver 있음 vs 없음: 혼합 앵커 2회 + 단독 프로파일로 처음 보는 수요·날에서 배치를 고르면 실제로 SLO를 지키는가'
    else:
        foot = ('글자색: 파랑 = seed 901 정보로 예측하는 선택기, 검정 = 정답지(오라클, 실제로는 불가능), 회색 = 정보 없이 배치 하나로 고정.   '
                '비용: 4090 한 장 = 1.0, 두 장 쓰는 배치 = 1.75 (3090 = 0.75).   α는 단독 프로파일이 있는 0.5·1.0만.\n'
                '"단독"은 두 서버를 다 올린 채 LLM만 또는 DiT만 돌린 결과. "간섭 배율" k = 혼합 p90 / 단독 p90 (seed 901의 같은 α 혼합 실행에서 측정 → 아직 혼합 없이 얻은 값은 아님).')
        title = 'Solver 비교: seed 901 정보만 들고 새 하루(seed 902·903)에서 배치를 고르면 실제로 SLO를 지키는가'
    fig.text(0.5, 0.01, foot, ha='center', fontsize=7.5)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0.11, 1, 0.96))
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(f'{a.out}.{ext}', dpi=200, bbox_inches='tight')
    print(a.out)


if __name__ == '__main__':
    main()
