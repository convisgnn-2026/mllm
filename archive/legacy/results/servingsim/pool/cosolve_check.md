# Pool-split co-solver  (4 x RTX 4090-class, DiT x10 of B300, mix=short, SLO: DiT 3.0x solo, LLM TTFT<=3s & TPOT<=60ms, floor 0.9)

## DiT tenant alone: SAR by (workers, req/s)

| workers | 0 | 0.02 |
|---|---|---|
| 1 | 1.00 | 0.88 |
| 2 | 1.00 | 0.99 |
| 3 | 1.00 | 1.00 |
| 4 | 1.00 | 1.00 |

## LLM tenant alone: SAR by (GPUs, req/s)  (LLMServingSim, colo)

| GPUs | 0 | 2 | 4 |
|---|---|---|---|
| 1 | 1.00 | 0.16 | 0.12 |
| 2 | 1.00 | 1.00 | 0.43 |
| 3 | 1.00 | 1.00 | 1.00 |
| 4 | 1.00 | 1.00 | 1.00 |

## Co-solver: best split (DiT:LLM:VLM GPUs) per demand triple, floor 0.9

### VLM = 0 req/s

| DiT req/s \ LLM req/s | 0 | 2 | 4 |
|---|---|---|---|
| 0 | 0:0 ✓ +4 idle | 0:2 ✓ +2 idle | 0:3 ✓ +1 idle |
| 0.02 | 2:0 ✓ +2 idle | 2:2 ✓ | 1:3 ✗ (0.88/1.00) |

feasible: 5/6.  ✗ = least-bad split with (DiT/LLM SAR).
fixed splits: 2:2:0 → 4/6, 1:3:0 → 3/6

**all tables: 5/6 demand points feasible with the adaptive split.**
