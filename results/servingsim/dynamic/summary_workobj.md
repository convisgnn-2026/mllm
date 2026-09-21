# Time-axis: does the optimum move?  (4 GPUs, window 300s x 60, reconfiguration 33s, floor 0.9, objective=work)

per-request work (GPU-s): DiT 18.53 · LLM 0.76 · VLM 1.05  (ratio 24 : 1 : 1.4)

objective=floor: every policy maximises the weakest tenant's SAR (per-tenant SLA). objective=work: every policy maximises work-weighted attainment (aggregate). The point of running both is that the winner is not supposed to depend on it -- cost-awareness should help under whichever objective you declare.

## staggered

per-window optimum changes **13/59** times, **9** distinct splits used

| policy | 최약 테넌트 SAR | 테넌트 평균 SAR | 하한 통과 창 | 요청가중 만족 | 작업가중 만족 | 전환 |
|---|---|---|---|---|---|---|
| equal 2:1:1 | 0.605 | 0.605 | 0.48 | 0.210 | 0.295 | 0 |
| static-best(하한) 0:4:0 | 0.343 | 0.343 | 0.35 | 0.761 | 0.640 | 0 |
| static-best(요청가중) 0:4:0 | 0.343 | 0.343 | 0.35 | 0.761 | 0.640 | 0 |
| static-best(작업가중) 0:4:0 | 0.343 | 0.343 | 0.35 | 0.761 | 0.640 | 0 |
| hysteresis k=2 | 0.888 | 0.888 | 0.88 | 0.924 | 0.931 | 6 |
| hysteresis k=3 | 0.804 | 0.804 | 0.82 | 0.817 | 0.837 | 5 |
| adaptive-naive | 0.986 | 0.986 | 1.00 | 0.973 | 0.977 | 13 |
| adaptive-cost(예측 완벽) | 0.986 | 0.986 | 1.00 | 0.973 | 0.977 | 8 |
| adaptive-cost(지속 가정) | 0.986 | 0.986 | 1.00 | 0.973 | 0.977 | 8 |

## correlated

per-window optimum changes **7/59** times, **5** distinct splits used

| policy | 최약 테넌트 SAR | 테넌트 평균 SAR | 하한 통과 창 | 요청가중 만족 | 작업가중 만족 | 전환 |
|---|---|---|---|---|---|---|
| equal 2:1:1 | 0.236 | 0.604 | 0.08 | 0.209 | 0.295 | 0 |
| static-best(하한) 0:4:0 | 0.017 | 0.337 | 0.02 | 0.761 | 0.640 | 0 |
| static-best(요청가중) 0:4:0 | 0.017 | 0.337 | 0.02 | 0.761 | 0.640 | 0 |
| static-best(작업가중) 0:4:0 | 0.017 | 0.337 | 0.02 | 0.761 | 0.640 | 0 |
| hysteresis k=2 | 0.245 | 0.547 | 0.22 | 0.798 | 0.697 | 7 |
| hysteresis k=3 | 0.194 | 0.495 | 0.17 | 0.779 | 0.680 | 5 |
| adaptive-naive | 0.266 | 0.567 | 0.25 | 0.802 | 0.702 | 7 |
| adaptive-cost(예측 완벽) | 0.266 | 0.567 | 0.25 | 0.802 | 0.702 | 7 |
| adaptive-cost(지속 가정) | 0.266 | 0.567 | 0.25 | 0.802 | 0.702 | 7 |

## random

per-window optimum changes **32/59** times, **10** distinct splits used

| policy | 최약 테넌트 SAR | 테넌트 평균 SAR | 하한 통과 창 | 요청가중 만족 | 작업가중 만족 | 전환 |
|---|---|---|---|---|---|---|
| equal 2:1:1 | 0.215 | 0.644 | 0.02 | 0.236 | 0.303 | 0 |
| static-best(하한) 0:3:1 | 0.087 | 0.564 | 0.05 | 0.782 | 0.733 | 0 |
| static-best(요청가중) 0:4:0 | 0.017 | 0.337 | 0.02 | 0.811 | 0.719 | 0 |
| static-best(작업가중) 0:3:1 | 0.087 | 0.564 | 0.05 | 0.782 | 0.733 | 0 |
| hysteresis k=2 | 0.160 | 0.583 | 0.10 | 0.847 | 0.802 | 11 |
| hysteresis k=3 | 0.087 | 0.513 | 0.05 | 0.571 | 0.565 | 5 |
| adaptive-naive | 0.299 | 0.652 | 0.25 | 0.910 | 0.861 | 32 |
| adaptive-cost(예측 완벽) | 0.232 | 0.645 | 0.18 | 0.905 | 0.857 | 15 |
| adaptive-cost(지속 가정) | 0.301 | 0.652 | 0.25 | 0.910 | 0.862 | 30 |

## burst

per-window optimum changes **4/59** times, **3** distinct splits used

| policy | 최약 테넌트 SAR | 테넌트 평균 SAR | 하한 통과 창 | 요청가중 만족 | 작업가중 만족 | 전환 |
|---|---|---|---|---|---|---|
| equal 2:1:1 | 0.500 | 0.816 | 0.00 | 0.471 | 0.531 | 0 |
| static-best(하한) 0:3:1 | 0.000 | 0.621 | 0.00 | 0.811 | 0.743 | 0 |
| static-best(요청가중) 0:3:1 | 0.000 | 0.621 | 0.00 | 0.811 | 0.743 | 0 |
| static-best(작업가중) 0:3:1 | 0.000 | 0.621 | 0.00 | 0.811 | 0.743 | 0 |
| hysteresis k=2 | 0.698 | 0.846 | 0.65 | 0.918 | 0.897 | 4 |
| hysteresis k=3 | 0.674 | 0.838 | 0.62 | 0.881 | 0.862 | 4 |
| adaptive-naive | 0.722 | 0.854 | 0.68 | 0.956 | 0.932 | 4 |
| adaptive-cost(예측 완벽) | 0.722 | 0.854 | 0.68 | 0.956 | 0.932 | 4 |
| adaptive-cost(지속 가정) | 0.722 | 0.854 | 0.68 | 0.956 | 0.932 | 4 |

## 한 줄

- **staggered**: 최적 13회 이동 · 최약 테넌트 SAR 0.605 → 0.986 (+38.1%p), 하한 통과 창 0.48 → 1.00 · 작업가중 0.295 → 0.977
- **correlated**: 최적 7회 이동 · 최약 테넌트 SAR 0.236 → 0.266 (+2.9%p), 하한 통과 창 0.08 → 0.25 · 작업가중 0.295 → 0.702
- **random**: 최적 32회 이동 · 최약 테넌트 SAR 0.215 → 0.301 (+8.5%p), 하한 통과 창 0.02 → 0.25 · 작업가중 0.303 → 0.862
- **burst**: 최적 4회 이동 · 최약 테넌트 SAR 0.500 → 0.722 (+22.1%p), 하한 통과 창 0.00 → 0.68 · 작업가중 0.531 → 0.932
