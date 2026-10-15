#!/usr/bin/env python3
"""Audit existing mixed-service measurements; no GPU or third-party packages.
Deadlines are sensitivity probes, not user-validated service objectives.
Failures/non-finite metrics count as misses. No latency model or online oracle.
"""
import argparse
import csv
import itertools
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ('C_4090', 'C_3090', 'C_split', 'S4_3090')
TTFT = (.3, .5, 1., 2.)
TPOT = (.012, .02, .03, .05)
DIT = (3., 5., 10., 20.)


def number(row, key):
    try:
        return float(row[key])
    except (KeyError, ValueError, TypeError):
        return float('nan')


def satisfies(row, limits):
    return not row.get('err') and all(math.isfinite(number(row, k)) and number(row, k) <= v for k, v in limits.items())


def attainment(rows, limits):
    return sum(bool(satisfies(r, limits)) for r in rows) / len(rows) if rows else 0.


def dominates(a, b):
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def read_cell(path):
    rows = {}
    audit = {}
    for kind in ('llm', 'dit'):
        with (path / kind / 'requests.csv').open() as f:
            rows[kind] = list(csv.DictReader(f))
        audit[kind] = dict(n=len(rows[kind]), errors=sum(bool(r.get('err')) for r in rows[kind]),
                           last_launch_s=max(number(r, 'launch') for r in rows[kind]))
    return rows, audit


def analyze(source, output):
    cells = json.loads((source / 'cells.json').read_text())
    loads = sorted({(c['lr'], c['dr']) for c in cells if c['llm'] != 'none' and c['dit'] != 'none'})
    records, audits, fronts = [], [], []
    for lr, dr in loads:
        measurements = {}
        vectors = {}
        for cfg in CONFIGS:
            candidates = list((source / cfg).glob(f'code_L{lr}_D{dr}'))
            if len(candidates) != 1:
                raise ValueError(f'Missing or ambiguous cell: {cfg}, {lr}, {dr}')
            measurements[cfg], audit = read_cell(candidates[0])
            audits.append(dict(config=cfg, llm_rate=lr, dit_rate=dr, **audit))
            c = next(c for c in cells if f"{c['llm']}_{c['dit']}" == cfg and c['lr'] == lr and c['dr'] == dr)
            vectors[cfg] = (c['L']['ttft_p90'], c['L']['tpot_p90'], c['D']['e2e_p90'])
        front = [cfg for cfg in CONFIGS if not any(dominates(vectors[other], vectors[cfg]) for other in CONFIGS if other != cfg)]
        fronts.append(dict(llm_rate=lr, dit_rate=dr, nondominated=front, latency_vectors=vectors))
        for tt, tp, dd in itertools.product(TTFT, TPOT, DIT):
            scores = {cfg: dict(llm=attainment(measurements[cfg]['llm'], dict(ttft=tt, tpot=tp)),
                                dit=attainment(measurements[cfg]['dit'], dict(e2e=dd))) for cfg in CONFIGS}
            feasible = [cfg for cfg in CONFIGS if min(scores[cfg].values()) >= .9]
            records.append(dict(llm_rate=lr, dit_rate=dr, ttft_s=tt, tpot_s=tp, dit_e2e_s=dd,
                                feasible=feasible, attainment=scores))
    coverage = {cfg: sum(cfg in r['feasible'] for r in records) for cfg in CONFIGS}
    joint = sum(bool(r['feasible']) for r in records)
    best = max(coverage, key=coverage.get)
    unique = {cfg: sum(r['feasible'] == [cfg] for r in records) for cfg in CONFIGS}
    report = dict(schema_version=1, exploratory=True, attainment_target=.9,
                  deadline_grid=dict(ttft_s=TTFT, tpot_s=TPOT, dit_e2e_s=DIT),
                  scenarios=len(records), feasible_scenarios=joint, fixed_coverage=coverage,
                  unique_feasible=unique, best_fixed=best, selection_coverage_gap=joint-coverage[best],
                  pareto=fronts, audit=audits, scenarios_detail=records)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'analysis.json').write_text(json.dumps(report, indent=2) + '\n')
    lines = ['# Solver 필요성: 기존 실측의 탐색적 분석', '',
             '> 서비스 요구를 확정하거나 온라인 solver의 효과를 입증한 결과가 아니다. 기존 측정에서 구성 선택의 여지가 있는지 확인한다.', '',
             '## 방법', '',
             '- 네 혼재 구성, 여섯 부하 조건의 requests.csv를 직접 채점한다.',
             '- TTFT {0.3, 0.5, 1, 2}s × 요청 평균 TPOT {12, 20, 30, 50}ms × DiT E2E {3, 5, 10, 20}s.',
             '- 두 서비스 각각 요청의 90% 이상 만족해야 통과한다. 오류와 비유한 값은 실패로 계산한다.',
             '- 12/20ms는 관찰된 TPOT 범위를 조사하는 탐색 기준이며 실제 사용자 요구가 아니다.',
             '- 지연 Pareto 비교는 TTFT p90·TPOT p90·DiT E2E p90만 사용한다. 전력·통계적 불확실성은 포함하지 않는다.', '',
             '## 결과', '', f'- 평가 조합: {len(records)}개. 하나 이상 통과 구성 존재: {joint}개.',
             f'- 전체 조합에서 통과 영역이 가장 넓은 고정 구성: {best} ({coverage[best]}개).',
             f'- 각 조합에서 통과 구성을 선택할 수 있다고 가정한 영역과의 차이: {joint-coverage[best]}개.',
             '- 조합은 동등하게 열거한 민감도 분석이다. 이 개수는 실제 트래픽 goodput나 성능 향상률이 아니다.', '',
             '| 구성 | 통과 조합 수 | 유일한 통과 구성인 조합 수 |', '|---|---:|---:|']
    lines += [f'| {cfg} | {coverage[cfg]} | {unique[cfg]} |' for cfg in CONFIGS]
    lines += ['', '## 지연 Pareto 후보', '', '| LLM rps | DiT rps | 비지배 구성 |', '|---:|---:|---|']
    lines += [f"| {f['llm_rate']} | {f['dit_rate']} | {', '.join(f['nondominated'])} |" for f in fronts]
    lines += ['', '## 유일한 통과 구성의 예시', '', '| 구성 | LLM/DiT rps | TTFT / TPOT / DiT deadline | LLM / DiT 만족률 |', '|---|---|---|---|']
    for cfg in CONFIGS:
        r = next((r for r in records if r['feasible'] == [cfg]), None)
        if r:
            s = r['attainment'][cfg]
            lines.append(f"| {cfg} | {r['llm_rate']}/{r['dit_rate']} | {r['ttft_s']}s / {r['tpot_s']*1000:g}ms / {r['dit_e2e_s']}s | {s['llm']:.1%} / {s['dit']:.1%} |")
    lines += ['', '## 판정과 한계', '',
              '- 서로 다른 구성만 통과하는 요구 영역이 있으면 구성 선택기의 근거가 된다. 동적 재배치나 복잡한 최적화 알고리즘의 필요성과는 다르다.',
              '- 유일한 통과 사례는 후보 네 개와 이번 측정에만 해당한다. 경계 부근의 차이는 반복 실험으로 확인해야 한다.',
              '- DiT 표본 수는 LLM 부하에 따라 90/45/30건, 단독 기준은 60건이다. 서로 다른 부하 간 지연 추세와 단독 대비 E2E 개선을 인과적으로 해석하지 않는다.',
              '- 시작 barrier, 동일 관측 구간, 반복 seed가 없으므로 통계적 유의성과 정상상태 용량을 주장하지 않는다.',
              '- 이 분석은 과거 정적 측정의 채점이며 수요 전환, 배치 변경 비용, 실제 온라인 제어 성능을 포함하지 않는다.', '',
              '## 측정 감사', '', '| 구성 | LLM/DiT rps | LLM/DiT 표본 | LLM/DiT 오류 | 마지막 도착 LLM/DiT s |', '|---|---|---|---|---|']
    for a in audits:
        l, d = a['llm'], a['dit']
        lines.append(f"| {a['config']} | {a['llm_rate']}/{a['dit_rate']} | {l['n']}/{d['n']} | {l['errors']}/{d['errors']} | {l['last_launch_s']:.1f}/{d['last_launch_s']:.1f} |")
    (output / 'summary.md').write_text('\n'.join(lines) + '\n')
    return report


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, default=ROOT / 'results/hetero/mixed_modal')
    ap.add_argument('--out', type=Path, default=ROOT / 'results/hetero/solver_need')
    args = ap.parse_args()
    r = analyze(args.source, args.out)
    print(json.dumps({k: r[k] for k in ('scenarios', 'feasible_scenarios', 'fixed_coverage', 'unique_feasible', 'best_fixed', 'selection_coverage_gap')}, indent=2))
