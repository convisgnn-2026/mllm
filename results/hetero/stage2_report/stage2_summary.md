# Stage 2: resident-solo profiles vs mixed, selector comparison

Profiling seed 901, alpha ∈ [0.5, 1.0] (tested grid only). Cost r = 0.75 (p_3090 / p_4090, research assumption), occupied cards only.

## Interference: mixed p90 / solo p90

| candidate | alpha | TTFT solo→mixed s | TPOT solo→mixed ms | DiT E2E solo→mixed s | ratios TTFT/TPOT/DiT |
|---|---|---|---|---|---|
| S4_4090 | 0.5 | 0.317 → 0.425 | 10.0 → 23.0 | 2.025 → 4.100 | 1.34 / 2.31 / 2.02 |
| S4_4090 | 1.0 | 0.336 → 0.664 | 14.8 → 53.6 | 2.597 → 8.551 | 1.97 / 3.62 / 3.29 |
| S3_3090 | 0.5 | 0.797 → 1.998 | 26.6 → 173.0 | 6.581 → 23.856 | 2.51 / 6.50 / 3.63 |
| S3_3090 | 1.0 | 0.992 → 41.406 | 66.1 → 422.7 | 24.401 → 359.182 | 41.72 / 6.39 / 14.72 |
| S4_3090 | 0.5 | 0.314 → 0.311 | 10.0 → 10.0 | 6.583 → 6.576 | 0.99 / 0.99 / 1.00 |
| S4_3090 | 1.0 | 0.334 → 0.340 | 14.8 → 14.9 | 29.314 → 31.372 | 1.02 / 1.00 / 1.07 |
| S3_4090 | 0.5 | 0.797 → 0.791 | 26.8 → 26.7 | 2.029 → 2.038 | 0.99 / 1.00 / 1.00 |
| S3_4090 | 1.0 | 0.994 → 0.996 | 66.5 → 67.1 | 2.597 → 2.622 | 1.00 / 1.01 / 1.01 |
| C_4090 | 0.5 | 0.688 → 0.757 | 10.2 → 10.2 | 2.016 → 2.332 | 1.10 / 1.00 / 1.16 |
| C_4090 | 1.0 | 0.744 → 0.939 | 10.3 → 10.3 | 2.596 → 3.775 | 1.26 / 1.00 / 1.45 |
| C_3090 | 0.5 | 0.680 → 2.363 | 10.1 → 26.0 | 6.598 → 11.942 | 3.47 / 2.57 / 1.81 |
| C_3090 | 1.0 | 0.762 → 3.557 | 10.3 → 29.8 | 29.667 → 205.485 | 4.66 / 2.89 / 6.93 |
| C_split | 0.5 | 0.665 → 0.756 | 10.1 → 10.3 | 2.110 → 2.529 | 1.14 / 1.01 / 1.20 |
| C_split | 1.0 | 0.763 → 0.953 | 10.3 → 10.6 | 2.679 → 3.957 | 1.25 / 1.03 / 1.48 |

## Cell prediction vs mixed (profiling seed, 192 cells × 2 alphas per candidate)

fp = predictor says pass, mixed fails (SLO violation risk); fn = predictor says fail, mixed passes (lost capacity).

| predictor | candidate | tp | fp | fn | tn |
|---|---|---|---|---|---|
| solo | S4_4090 | 21 | 138 | 0 | 225 |
| solo | S3_3090 | 0 | 8 | 0 | 376 |
| solo | S4_3090 | 47 | 0 | 0 | 337 |
| solo | S3_4090 | 16 | 0 | 0 | 368 |
| solo | C_4090 | 120 | 36 | 0 | 228 |
| solo | C_3090 | 0 | 40 | 0 | 344 |
| solo | C_split | 107 | 49 | 0 | 228 |
| solo+k (LOAO) | S4_4090 | 4 | 8 | 17 | 355 |
| solo+k (LOAO) | S3_3090 | 0 | 0 | 0 | 384 |
| solo+k (LOAO) | S4_3090 | 47 | 0 | 0 | 337 |
| solo+k (LOAO) | S3_4090 | 16 | 0 | 0 | 368 |
| solo+k (LOAO) | C_4090 | 96 | 16 | 24 | 248 |
| solo+k (LOAO) | C_3090 | 0 | 0 | 0 | 384 |
| solo+k (LOAO) | C_split | 83 | 9 | 24 | 268 |

## Selectors on the profiling seed (candidates: S4_4090, S3_3090, S4_3090, S3_4090, C_4090, C_3090, C_split)

max-alpha: over SLO cells where someone passes or is predicted to pass. exact = realised alpha_max equals oracle; regret = Σ(oracle − realised); overpromise = predicted alpha above realised.
min-cost: per (cell, alpha) where at least one candidate passes in mixed. violation = chosen candidate fails in mixed; overpay = passes but costs more than the cheapest passing one; missed = selector finds none. false-feasible = (cell, alpha) where nothing passes in mixed but the selector still deploys (counted separately).

| selector | max-α cells | exact | regret Σα | overpromise | min-cost cases | ok (incl. overpay) | violation | overpay | missed | cost regret Σ | false-feasible |
|---|---|---|---|---|---|---|---|---|---|---|---|
| solo | 119 | 62 | 34.00 | 99 | 138 | 45 | 93 | 0 | 0 | 0.00 | 70 |
| solo+k (LOAO) | 94 | 80 | 7.00 | 22 | 138 | 114 | 6 | 15 | 18 | 11.25 | 16 |
| oracle | 94 | 94 | 0.00 | 0 | 138 | 138 | 0 | 0 | 0 | 0.00 | 0 |
| fixed:C_4090 | 94 | 76 | 9.00 | 0 | 138 | 120 | 18 | 18 | 0 | 13.50 | 246 |
| fixed:S4_3090 | 94 | 23 | 45.50 | 0 | 138 | 47 | 91 | 14 | 0 | 10.50 | 246 |

## Selectors on unseen stage1_seed902 (candidates: S4_4090, S4_3090, S3_4090, C_4090; profiles and k from seed 901)

max-alpha: over SLO cells where someone passes or is predicted to pass. exact = realised alpha_max equals oracle; regret = Σ(oracle − realised); overpromise = predicted alpha above realised.
min-cost: per (cell, alpha) where at least one candidate passes in mixed. violation = chosen candidate fails in mixed; overpay = passes but costs more than the cheapest passing one; missed = selector finds none. false-feasible = (cell, alpha) where nothing passes in mixed but the selector still deploys (counted separately).

| selector | max-α cells | exact | regret Σα | overpromise | min-cost cases | ok (incl. overpay) | violation | overpay | missed | cost regret Σ | false-feasible |
|---|---|---|---|---|---|---|---|---|---|---|---|
| solo | 123 | 68 | 28.50 | 99 | 156 | 82 | 70 | 0 | 4 | 0.00 | 56 |
| solo+k (seed 901) | 105 | 97 | 4.00 | 0 | 156 | 139 | 0 | 25 | 17 | 18.75 | 0 |
| mixed seed 901 | 105 | 96 | 4.50 | 0 | 156 | 138 | 0 | 25 | 18 | 18.75 | 0 |
| oracle | 105 | 105 | 0.00 | 0 | 156 | 156 | 0 | 0 | 0 | 0.00 | 0 |
| fixed:C_4090 | 105 | 74 | 18.00 | 0 | 156 | 120 | 36 | 39 | 0 | 29.25 | 228 |
| fixed:S4_3090 | 105 | 38 | 42.50 | 0 | 156 | 71 | 85 | 28 | 0 | 21.00 | 228 |

## Selectors on unseen stage1_seed903 (candidates: S4_4090, S4_3090, S3_4090, C_4090; profiles and k from seed 901)

max-alpha: over SLO cells where someone passes or is predicted to pass. exact = realised alpha_max equals oracle; regret = Σ(oracle − realised); overpromise = predicted alpha above realised.
min-cost: per (cell, alpha) where at least one candidate passes in mixed. violation = chosen candidate fails in mixed; overpay = passes but costs more than the cheapest passing one; missed = selector finds none. false-feasible = (cell, alpha) where nothing passes in mixed but the selector still deploys (counted separately).

| selector | max-α cells | exact | regret Σα | overpromise | min-cost cases | ok (incl. overpay) | violation | overpay | missed | cost regret Σ | false-feasible |
|---|---|---|---|---|---|---|---|---|---|---|---|
| solo | 119 | 80 | 21.50 | 111 | 106 | 46 | 60 | 0 | 0 | 0.00 | 102 |
| solo+k (seed 901) | 96 | 93 | 1.50 | 37 | 106 | 100 | 5 | 2 | 1 | 1.50 | 34 |
| mixed seed 901 | 95 | 92 | 1.50 | 36 | 106 | 100 | 5 | 2 | 1 | 1.50 | 33 |
| oracle | 90 | 90 | 0.00 | 0 | 106 | 106 | 0 | 0 | 0 | 0.00 | 0 |
| fixed:C_4090 | 90 | 76 | 7.00 | 0 | 106 | 92 | 14 | 22 | 0 | 16.50 | 278 |
| fixed:S4_3090 | 90 | 26 | 33.00 | 0 | 106 | 40 | 66 | 14 | 0 | 10.50 | 278 |
