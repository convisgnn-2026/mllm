# Stage 2b: solver with / without, unseen seeds

Candidates ['S4_4090', 'S4_3090', 'S3_4090', 'C_4090']; shared-card candidates ['S4_4090', 'C_4090'] use solo profiles + k; the others use k = 1 with the seed-901 mixed run as profile. Anchors [0.5, 1.0]; held-out alphas [0.6, 0.75, 1.1, 1.25]. Cost r = 0.75, occupied cards.

Solo profiles on seed 901: {'S4_4090': [0.5, 0.6, 0.75, 0.85, 1.0, 1.5], 'C_4090': [0.5, 1.0, 1.1, 1.25, 1.4, 1.5]}

## Measured k (mixed p90 / solo p90, seed 901)

| candidate | alpha | k TTFT | k TPOT | k DiT |
|---|---|---|---|---|
| C_4090 | 0.5 | 1.10 | 1.00 | 1.16 |
| C_4090 | 1.0 | 1.26 | 1.00 | 1.45 |
| C_4090 | 1.1 | 1.30 | 1.00 | 1.67 |
| C_4090 | 1.25 | 1.44 | 1.00 | 2.08 |
| C_4090 | 1.4 | 2.16 | 1.00 | 3.89 |
| C_4090 | 1.5 | 2.53 | 1.00 | 6.33 |
| S4_4090 | 0.5 | 1.34 | 2.31 | 2.02 |
| S4_4090 | 0.6 | 1.47 | 2.49 | 2.16 |
| S4_4090 | 0.75 | 1.62 | 2.80 | 2.15 |
| S4_4090 | 0.85 | 1.71 | 2.99 | 2.33 |
| S4_4090 | 1.0 | 1.97 | 3.62 | 3.29 |
| S4_4090 | 1.5 | 10.53 | 7.29 | 20.80 |

## Two-anchor k models vs measured k (relative error, held-out alphas)

| candidate | alpha | linear TTFT/TPOT/DiT | const TTFT/TPOT/DiT |
|---|---|---|---|
| C_4090 | 1.1 | -0% / +0% / -9% | -3% / +0% / -13% |
| C_4090 | 1.25 | -7% / +0% / -23% | -12% / +0% / -30% |
| C_4090 | 1.4 | -36% / +1% / -57% | -42% / +0% / -63% |
| C_4090 | 1.5 | -44% / +1% / -72% | -50% / +1% / -77% |
| S4_4090 | 0.6 | -0% / +3% / +5% | -9% / -7% / -6% |
| S4_4090 | 0.75 | +2% / +6% / +24% | -17% / -18% / -6% |
| S4_4090 | 0.85 | +4% / +8% / +25% | +16% / +21% / +41% |
| S4_4090 | 1.5 | -75% / -32% / -78% | -81% / -50% / -84% |

## Seed 901, held-out alphas, shared candidates: predicted cell vs mixed cell

fp = predicted pass, mixed fails; fn = predicted fail, mixed passes.

| predictor | candidate | tp | fp | fn | tn |
|---|---|---|---|---|---|
| solo only | S4_4090 | 24 | 261 | 0 | 483 |
| solo only | C_4090 | 56 | 156 | 0 | 556 |
| solo+k-linear | S4_4090 | 24 | 2 | 0 | 742 |
| solo+k-linear | C_4090 | 56 | 52 | 0 | 660 |
| solo+k-const | S4_4090 | 24 | 18 | 0 | 726 |
| solo+k-const | C_4090 | 56 | 56 | 0 | 656 |
| solo+k-measured | S4_4090 | 24 | 4 | 0 | 740 |
| solo+k-measured | C_4090 | 56 | 0 | 0 | 712 |

## Pooled over unseen seeds, held-out alphas [0.6, 0.75, 1.1, 1.25] (the fair test)

| selector | max-α cells | exact | regret Σα | overpromise | min-cost cases | ok | violation | overpay | missed | cost regret Σ | false-feasible |
|---|---|---|---|---|---|---|---|---|---|---|---|
| solo only | 224 | 153 | 44.45 | 205 | 383 | 142 | 237 | 0 | 4 | 0.00 | 329 |
| solo+k-const (2 anchors) | 181 | 129 | 32.30 | 71 | 383 | 255 | 56 | 6 | 72 | 4.50 | 77 |
| solo+k-linear (2 anchors) | 180 | 140 | 25.85 | 56 | 383 | 272 | 36 | 10 | 75 | 7.50 | 74 |
| solo+k-measured (same alpha) | 180 | 138 | 27.10 | 51 | 383 | 266 | 40 | 10 | 77 | 7.50 | 62 |
| mixed seed 901 | 180 | 140 | 25.85 | 48 | 383 | 272 | 32 | 10 | 79 | 7.50 | 62 |
| oracle | 168 | 168 | 0.00 | 0 | 383 | 383 | 0 | 0 | 0 | 0.00 | 0 |
| fixed:S4_4090 | 168 | 9 | 120.85 | 0 | 383 | 45 | 338 | 0 | 0 | 0.00 | 1153 |
| fixed:S4_3090 | 168 | 54 | 79.35 | 0 | 383 | 156 | 227 | 30 | 0 | 22.50 | 1153 |
| fixed:S3_4090 | 168 | 1 | 132.85 | 0 | 383 | 16 | 367 | 12 | 0 | 9.00 | 1153 |
| fixed:C_4090 | 168 | 125 | 28.95 | 0 | 383 | 310 | 73 | 39 | 0 | 29.25 | 1153 |

## Pooled over unseen seeds, all alphas [0.5, 0.6, 0.75, 1.0, 1.1, 1.25]

| selector | max-α cells | exact | regret Σα | overpromise | min-cost cases | ok | violation | overpay | missed | cost regret Σ | false-feasible |
|---|---|---|---|---|---|---|---|---|---|---|---|
| solo only | 242 | 157 | 42.25 | 217 | 645 | 270 | 367 | 0 | 8 | 0.00 | 487 |
| solo+k-const (2 anchors) | 203 | 181 | 7.30 | 79 | 645 | 493 | 61 | 33 | 91 | 24.75 | 110 |
| solo+k-linear (2 anchors) | 203 | 187 | 5.40 | 70 | 645 | 510 | 41 | 37 | 94 | 27.75 | 107 |
| solo+k-measured (same alpha) | 203 | 185 | 6.15 | 69 | 645 | 504 | 45 | 37 | 96 | 27.75 | 95 |
| mixed seed 901 | 203 | 187 | 5.40 | 66 | 645 | 510 | 37 | 37 | 98 | 27.75 | 95 |
| oracle | 195 | 195 | 0.00 | 0 | 645 | 645 | 0 | 0 | 0 | 0.00 | 0 |
| fixed:S4_4090 | 195 | 13 | 117.85 | 0 | 645 | 121 | 524 | 0 | 0 | 0.00 | 1659 |
| fixed:S4_3090 | 195 | 57 | 90.60 | 0 | 645 | 267 | 378 | 72 | 0 | 54.00 | 1659 |
| fixed:S3_4090 | 195 | 1 | 145.35 | 0 | 645 | 44 | 601 | 38 | 0 | 28.50 | 1659 |
| fixed:C_4090 | 195 | 139 | 31.95 | 0 | 645 | 522 | 123 | 100 | 0 | 75.00 | 1659 |

## stage1_seed902, held_out alphas (candidates S4_4090, S4_3090, S3_4090, C_4090)

| selector | max-α cells | exact | regret Σα | overpromise | min-cost cases | ok | violation | overpay | missed | cost regret Σ | false-feasible |
|---|---|---|---|---|---|---|---|---|---|---|---|
| solo only | 113 | 69 | 23.80 | 99 | 221 | 92 | 125 | 0 | 4 | 0.00 | 137 |
| solo+k-const (2 anchors) | 98 | 65 | 18.05 | 26 | 221 | 155 | 18 | 6 | 48 | 4.50 | 21 |
| solo+k-linear (2 anchors) | 98 | 71 | 16.10 | 18 | 221 | 162 | 9 | 9 | 50 | 6.75 | 20 |
| solo+k-measured (same alpha) | 98 | 70 | 16.60 | 16 | 221 | 157 | 11 | 9 | 53 | 6.75 | 16 |
| mixed seed 901 | 98 | 71 | 16.10 | 14 | 221 | 160 | 7 | 9 | 54 | 6.75 | 16 |
| oracle | 96 | 96 | 0.00 | 0 | 221 | 221 | 0 | 0 | 0 | 0.00 | 0 |
| fixed:S4_4090 | 96 | 5 | 66.25 | 0 | 221 | 33 | 188 | 0 | 0 | 0.00 | 547 |
| fixed:S4_3090 | 96 | 30 | 45.15 | 0 | 221 | 92 | 129 | 22 | 0 | 16.50 | 547 |
| fixed:S3_4090 | 96 | 1 | 73.75 | 0 | 221 | 16 | 205 | 12 | 0 | 9.00 | 547 |
| fixed:C_4090 | 96 | 69 | 17.25 | 0 | 221 | 178 | 43 | 27 | 0 | 20.25 | 547 |

## stage1_seed902, anchors alphas (candidates S4_4090, S4_3090, S3_4090, C_4090)

| selector | max-α cells | exact | regret Σα | overpromise | min-cost cases | ok | violation | overpay | missed | cost regret Σ | false-feasible |
|---|---|---|---|---|---|---|---|---|---|---|---|
| solo only | 123 | 68 | 28.50 | 99 | 156 | 82 | 70 | 0 | 4 | 0.00 | 56 |
| solo+k-const (2 anchors) | 105 | 96 | 4.50 | 0 | 156 | 138 | 0 | 25 | 18 | 18.75 | 0 |
| solo+k-linear (2 anchors) | 105 | 96 | 4.50 | 0 | 156 | 138 | 0 | 25 | 18 | 18.75 | 0 |
| solo+k-measured (same alpha) | 105 | 96 | 4.50 | 0 | 156 | 138 | 0 | 25 | 18 | 18.75 | 0 |
| mixed seed 901 | 105 | 96 | 4.50 | 0 | 156 | 138 | 0 | 25 | 18 | 18.75 | 0 |
| oracle | 105 | 105 | 0.00 | 0 | 156 | 156 | 0 | 0 | 0 | 0.00 | 0 |
| fixed:S4_4090 | 105 | 19 | 53.00 | 0 | 156 | 50 | 106 | 0 | 0 | 0.00 | 228 |
| fixed:S4_3090 | 105 | 38 | 42.50 | 0 | 156 | 71 | 85 | 28 | 0 | 21.00 | 228 |
| fixed:S3_4090 | 105 | 8 | 66.00 | 0 | 156 | 24 | 132 | 22 | 0 | 16.50 | 228 |
| fixed:C_4090 | 105 | 74 | 18.00 | 0 | 156 | 120 | 36 | 39 | 0 | 29.25 | 228 |

## stage1_seed903, held_out alphas (candidates S4_4090, S4_3090, S3_4090, C_4090)

| selector | max-α cells | exact | regret Σα | overpromise | min-cost cases | ok | violation | overpay | missed | cost regret Σ | false-feasible |
|---|---|---|---|---|---|---|---|---|---|---|---|
| solo only | 111 | 84 | 20.65 | 106 | 162 | 50 | 112 | 0 | 0 | 0.00 | 192 |
| solo+k-const (2 anchors) | 83 | 64 | 14.25 | 45 | 162 | 100 | 38 | 0 | 24 | 0.00 | 56 |
| solo+k-linear (2 anchors) | 82 | 69 | 9.75 | 38 | 162 | 110 | 27 | 1 | 25 | 0.75 | 54 |
| solo+k-measured (same alpha) | 82 | 68 | 10.50 | 35 | 162 | 109 | 29 | 1 | 24 | 0.75 | 46 |
| mixed seed 901 | 82 | 69 | 9.75 | 34 | 162 | 112 | 25 | 1 | 25 | 0.75 | 46 |
| oracle | 72 | 72 | 0.00 | 0 | 162 | 162 | 0 | 0 | 0 | 0.00 | 0 |
| fixed:S4_4090 | 72 | 4 | 54.60 | 0 | 162 | 12 | 150 | 0 | 0 | 0.00 | 606 |
| fixed:S4_3090 | 72 | 24 | 34.20 | 0 | 162 | 64 | 98 | 8 | 0 | 6.00 | 606 |
| fixed:S3_4090 | 72 | 0 | 59.10 | 0 | 162 | 0 | 162 | 0 | 0 | 0.00 | 606 |
| fixed:C_4090 | 72 | 56 | 11.70 | 0 | 162 | 132 | 30 | 12 | 0 | 9.00 | 606 |

## stage1_seed903, anchors alphas (candidates S4_4090, S4_3090, S3_4090, C_4090)

| selector | max-α cells | exact | regret Σα | overpromise | min-cost cases | ok | violation | overpay | missed | cost regret Σ | false-feasible |
|---|---|---|---|---|---|---|---|---|---|---|---|
| solo only | 119 | 80 | 21.50 | 111 | 106 | 46 | 60 | 0 | 0 | 0.00 | 102 |
| solo+k-const (2 anchors) | 95 | 92 | 1.50 | 36 | 106 | 100 | 5 | 2 | 1 | 1.50 | 33 |
| solo+k-linear (2 anchors) | 95 | 92 | 1.50 | 36 | 106 | 100 | 5 | 2 | 1 | 1.50 | 33 |
| solo+k-measured (same alpha) | 95 | 92 | 1.50 | 36 | 106 | 100 | 5 | 2 | 1 | 1.50 | 33 |
| mixed seed 901 | 95 | 92 | 1.50 | 36 | 106 | 100 | 5 | 2 | 1 | 1.50 | 33 |
| oracle | 90 | 90 | 0.00 | 0 | 106 | 106 | 0 | 0 | 0 | 0.00 | 0 |
| fixed:S4_4090 | 90 | 20 | 40.00 | 0 | 106 | 26 | 80 | 0 | 0 | 0.00 | 278 |
| fixed:S4_3090 | 90 | 26 | 33.00 | 0 | 106 | 40 | 66 | 14 | 0 | 10.50 | 278 |
| fixed:S3_4090 | 90 | 2 | 51.00 | 0 | 106 | 4 | 102 | 4 | 0 | 3.00 | 278 |
| fixed:C_4090 | 90 | 76 | 7.00 | 0 | 106 | 92 | 14 | 22 | 0 | 16.50 | 278 |
