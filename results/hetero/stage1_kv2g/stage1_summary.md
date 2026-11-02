# Stage 1 pilot: fixed-cost capacity map (exploratory, seed 901)

protocol bb7ecf80e001 · runs done 1/1 (valid 1). Exploratory capacity map on one seed; not a selector validation. alpha_max is the largest *tested* alpha that passed (observed grid values only, never exact capacity bounds; no passing point means none observed). Untested cells are blank, never passes. Prices are research assumptions, not market quotes.

## Runs

| candidate | alpha | state | valid (KV+plan) | LLM n / ok / err / timeout / unfinished | n=1 tokens | DiT n / ok / err / timeout | LLM TTFT p90 (ok) | LLM TPOT p90 (ok) | DiT E2E p90 (ok) | backlog@end LLM/DiT | drain used s | saturation suspect | Wh obs 4090/3090 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C_4090 | 1.0 | done | True | 1152/1151/1/0/0 | 0 | 166/166/0/0 | 0.994 | 0.0103 | 3.777 | 0/3 | 0 | False | 46.14/40.2 |

## Stability windows (200 s × 3, measured arrivals / ok completions / failed ends / in-flight at window end)

| candidate | alpha | service | w0 | w1 | w2 | completed in drain | drain timeouts |
|---|---|---|---|---|---|---|---|
| C_4090 | 1.0 | llm | 399/397/0/2 | 393/393/1/1 | 360/361/0/0 | 0 | 0 |
| C_4090 | 1.0 | dit | 70/69/0/2 | 49/51/0/0 | 47/44/0/3 | 3 | 0 |

## alpha_max per SLO cell (all 192 cells; verdict per alpha, ✓ pass · ✗ fail · · untested)

Attainment denominators include errors/timeouts/unfinished. A 99 % target with ~90 DiT samples is a point estimate (one failure breaks it).

| TTFT s | TPOT ms | DiT E2E s | target | C_4090 |
|---|---|---|---|---|
| 0.3 | 12 | 3.0 | 0.9 | ✗ |
| 0.3 | 12 | 3.0 | 0.95 | ✗ |
| 0.3 | 12 | 3.0 | 0.99 | ✗ |
| 0.3 | 12 | 5.0 | 0.9 | ✗ |
| 0.3 | 12 | 5.0 | 0.95 | ✗ |
| 0.3 | 12 | 5.0 | 0.99 | ✗ |
| 0.3 | 12 | 10.0 | 0.9 | ✗ |
| 0.3 | 12 | 10.0 | 0.95 | ✗ |
| 0.3 | 12 | 10.0 | 0.99 | ✗ |
| 0.3 | 12 | 20.0 | 0.9 | ✗ |
| 0.3 | 12 | 20.0 | 0.95 | ✗ |
| 0.3 | 12 | 20.0 | 0.99 | ✗ |
| 0.3 | 20 | 3.0 | 0.9 | ✗ |
| 0.3 | 20 | 3.0 | 0.95 | ✗ |
| 0.3 | 20 | 3.0 | 0.99 | ✗ |
| 0.3 | 20 | 5.0 | 0.9 | ✗ |
| 0.3 | 20 | 5.0 | 0.95 | ✗ |
| 0.3 | 20 | 5.0 | 0.99 | ✗ |
| 0.3 | 20 | 10.0 | 0.9 | ✗ |
| 0.3 | 20 | 10.0 | 0.95 | ✗ |
| 0.3 | 20 | 10.0 | 0.99 | ✗ |
| 0.3 | 20 | 20.0 | 0.9 | ✗ |
| 0.3 | 20 | 20.0 | 0.95 | ✗ |
| 0.3 | 20 | 20.0 | 0.99 | ✗ |
| 0.3 | 30 | 3.0 | 0.9 | ✗ |
| 0.3 | 30 | 3.0 | 0.95 | ✗ |
| 0.3 | 30 | 3.0 | 0.99 | ✗ |
| 0.3 | 30 | 5.0 | 0.9 | ✗ |
| 0.3 | 30 | 5.0 | 0.95 | ✗ |
| 0.3 | 30 | 5.0 | 0.99 | ✗ |
| 0.3 | 30 | 10.0 | 0.9 | ✗ |
| 0.3 | 30 | 10.0 | 0.95 | ✗ |
| 0.3 | 30 | 10.0 | 0.99 | ✗ |
| 0.3 | 30 | 20.0 | 0.9 | ✗ |
| 0.3 | 30 | 20.0 | 0.95 | ✗ |
| 0.3 | 30 | 20.0 | 0.99 | ✗ |
| 0.3 | 50 | 3.0 | 0.9 | ✗ |
| 0.3 | 50 | 3.0 | 0.95 | ✗ |
| 0.3 | 50 | 3.0 | 0.99 | ✗ |
| 0.3 | 50 | 5.0 | 0.9 | ✗ |
| 0.3 | 50 | 5.0 | 0.95 | ✗ |
| 0.3 | 50 | 5.0 | 0.99 | ✗ |
| 0.3 | 50 | 10.0 | 0.9 | ✗ |
| 0.3 | 50 | 10.0 | 0.95 | ✗ |
| 0.3 | 50 | 10.0 | 0.99 | ✗ |
| 0.3 | 50 | 20.0 | 0.9 | ✗ |
| 0.3 | 50 | 20.0 | 0.95 | ✗ |
| 0.3 | 50 | 20.0 | 0.99 | ✗ |
| 0.5 | 12 | 3.0 | 0.9 | ✗ |
| 0.5 | 12 | 3.0 | 0.95 | ✗ |
| 0.5 | 12 | 3.0 | 0.99 | ✗ |
| 0.5 | 12 | 5.0 | 0.9 | ✗ |
| 0.5 | 12 | 5.0 | 0.95 | ✗ |
| 0.5 | 12 | 5.0 | 0.99 | ✗ |
| 0.5 | 12 | 10.0 | 0.9 | ✗ |
| 0.5 | 12 | 10.0 | 0.95 | ✗ |
| 0.5 | 12 | 10.0 | 0.99 | ✗ |
| 0.5 | 12 | 20.0 | 0.9 | ✗ |
| 0.5 | 12 | 20.0 | 0.95 | ✗ |
| 0.5 | 12 | 20.0 | 0.99 | ✗ |
| 0.5 | 20 | 3.0 | 0.9 | ✗ |
| 0.5 | 20 | 3.0 | 0.95 | ✗ |
| 0.5 | 20 | 3.0 | 0.99 | ✗ |
| 0.5 | 20 | 5.0 | 0.9 | ✗ |
| 0.5 | 20 | 5.0 | 0.95 | ✗ |
| 0.5 | 20 | 5.0 | 0.99 | ✗ |
| 0.5 | 20 | 10.0 | 0.9 | ✗ |
| 0.5 | 20 | 10.0 | 0.95 | ✗ |
| 0.5 | 20 | 10.0 | 0.99 | ✗ |
| 0.5 | 20 | 20.0 | 0.9 | ✗ |
| 0.5 | 20 | 20.0 | 0.95 | ✗ |
| 0.5 | 20 | 20.0 | 0.99 | ✗ |
| 0.5 | 30 | 3.0 | 0.9 | ✗ |
| 0.5 | 30 | 3.0 | 0.95 | ✗ |
| 0.5 | 30 | 3.0 | 0.99 | ✗ |
| 0.5 | 30 | 5.0 | 0.9 | ✗ |
| 0.5 | 30 | 5.0 | 0.95 | ✗ |
| 0.5 | 30 | 5.0 | 0.99 | ✗ |
| 0.5 | 30 | 10.0 | 0.9 | ✗ |
| 0.5 | 30 | 10.0 | 0.95 | ✗ |
| 0.5 | 30 | 10.0 | 0.99 | ✗ |
| 0.5 | 30 | 20.0 | 0.9 | ✗ |
| 0.5 | 30 | 20.0 | 0.95 | ✗ |
| 0.5 | 30 | 20.0 | 0.99 | ✗ |
| 0.5 | 50 | 3.0 | 0.9 | ✗ |
| 0.5 | 50 | 3.0 | 0.95 | ✗ |
| 0.5 | 50 | 3.0 | 0.99 | ✗ |
| 0.5 | 50 | 5.0 | 0.9 | ✗ |
| 0.5 | 50 | 5.0 | 0.95 | ✗ |
| 0.5 | 50 | 5.0 | 0.99 | ✗ |
| 0.5 | 50 | 10.0 | 0.9 | ✗ |
| 0.5 | 50 | 10.0 | 0.95 | ✗ |
| 0.5 | 50 | 10.0 | 0.99 | ✗ |
| 0.5 | 50 | 20.0 | 0.9 | ✗ |
| 0.5 | 50 | 20.0 | 0.95 | ✗ |
| 0.5 | 50 | 20.0 | 0.99 | ✗ |
| 1.0 | 12 | 3.0 | 0.9 | ✗ |
| 1.0 | 12 | 3.0 | 0.95 | ✗ |
| 1.0 | 12 | 3.0 | 0.99 | ✗ |
| 1.0 | 12 | 5.0 | 0.9 | ✓ observed=1.0 upper-unmeasured |
| 1.0 | 12 | 5.0 | 0.95 | ✗ |
| 1.0 | 12 | 5.0 | 0.99 | ✗ |
| 1.0 | 12 | 10.0 | 0.9 | ✓ observed=1.0 upper-unmeasured |
| 1.0 | 12 | 10.0 | 0.95 | ✗ |
| 1.0 | 12 | 10.0 | 0.99 | ✗ |
| 1.0 | 12 | 20.0 | 0.9 | ✓ observed=1.0 upper-unmeasured |
| 1.0 | 12 | 20.0 | 0.95 | ✗ |
| 1.0 | 12 | 20.0 | 0.99 | ✗ |
| 1.0 | 20 | 3.0 | 0.9 | ✗ |
| 1.0 | 20 | 3.0 | 0.95 | ✗ |
| 1.0 | 20 | 3.0 | 0.99 | ✗ |
| 1.0 | 20 | 5.0 | 0.9 | ✓ observed=1.0 upper-unmeasured |
| 1.0 | 20 | 5.0 | 0.95 | ✗ |
| 1.0 | 20 | 5.0 | 0.99 | ✗ |
| 1.0 | 20 | 10.0 | 0.9 | ✓ observed=1.0 upper-unmeasured |
| 1.0 | 20 | 10.0 | 0.95 | ✗ |
| 1.0 | 20 | 10.0 | 0.99 | ✗ |
| 1.0 | 20 | 20.0 | 0.9 | ✓ observed=1.0 upper-unmeasured |
| 1.0 | 20 | 20.0 | 0.95 | ✗ |
| 1.0 | 20 | 20.0 | 0.99 | ✗ |
| 1.0 | 30 | 3.0 | 0.9 | ✗ |
| 1.0 | 30 | 3.0 | 0.95 | ✗ |
| 1.0 | 30 | 3.0 | 0.99 | ✗ |
| 1.0 | 30 | 5.0 | 0.9 | ✓ observed=1.0 upper-unmeasured |
| 1.0 | 30 | 5.0 | 0.95 | ✗ |
| 1.0 | 30 | 5.0 | 0.99 | ✗ |
| 1.0 | 30 | 10.0 | 0.9 | ✓ observed=1.0 upper-unmeasured |
| 1.0 | 30 | 10.0 | 0.95 | ✗ |
| 1.0 | 30 | 10.0 | 0.99 | ✗ |
| 1.0 | 30 | 20.0 | 0.9 | ✓ observed=1.0 upper-unmeasured |
| 1.0 | 30 | 20.0 | 0.95 | ✗ |
| 1.0 | 30 | 20.0 | 0.99 | ✗ |
| 1.0 | 50 | 3.0 | 0.9 | ✗ |
| 1.0 | 50 | 3.0 | 0.95 | ✗ |
| 1.0 | 50 | 3.0 | 0.99 | ✗ |
| 1.0 | 50 | 5.0 | 0.9 | ✓ observed=1.0 upper-unmeasured |
| 1.0 | 50 | 5.0 | 0.95 | ✗ |
| 1.0 | 50 | 5.0 | 0.99 | ✗ |
| 1.0 | 50 | 10.0 | 0.9 | ✓ observed=1.0 upper-unmeasured |
| 1.0 | 50 | 10.0 | 0.95 | ✗ |
| 1.0 | 50 | 10.0 | 0.99 | ✗ |
| 1.0 | 50 | 20.0 | 0.9 | ✓ observed=1.0 upper-unmeasured |
| 1.0 | 50 | 20.0 | 0.95 | ✗ |
| 1.0 | 50 | 20.0 | 0.99 | ✗ |
| 2.0 | 12 | 3.0 | 0.9 | ✗ |
| 2.0 | 12 | 3.0 | 0.95 | ✗ |
| 2.0 | 12 | 3.0 | 0.99 | ✗ |
| 2.0 | 12 | 5.0 | 0.9 | ✓ observed=1.0 upper-unmeasured |
| 2.0 | 12 | 5.0 | 0.95 | ✓ observed=1.0 upper-unmeasured |
| 2.0 | 12 | 5.0 | 0.99 | ✗ |
| 2.0 | 12 | 10.0 | 0.9 | ✓ observed=1.0 upper-unmeasured |
| 2.0 | 12 | 10.0 | 0.95 | ✓ observed=1.0 upper-unmeasured |
| 2.0 | 12 | 10.0 | 0.99 | ✓ observed=1.0 upper-unmeasured |
| 2.0 | 12 | 20.0 | 0.9 | ✓ observed=1.0 upper-unmeasured |
| 2.0 | 12 | 20.0 | 0.95 | ✓ observed=1.0 upper-unmeasured |
| 2.0 | 12 | 20.0 | 0.99 | ✓ observed=1.0 upper-unmeasured |
| 2.0 | 20 | 3.0 | 0.9 | ✗ |
| 2.0 | 20 | 3.0 | 0.95 | ✗ |
| 2.0 | 20 | 3.0 | 0.99 | ✗ |
| 2.0 | 20 | 5.0 | 0.9 | ✓ observed=1.0 upper-unmeasured |
| 2.0 | 20 | 5.0 | 0.95 | ✓ observed=1.0 upper-unmeasured |
| 2.0 | 20 | 5.0 | 0.99 | ✗ |
| 2.0 | 20 | 10.0 | 0.9 | ✓ observed=1.0 upper-unmeasured |
| 2.0 | 20 | 10.0 | 0.95 | ✓ observed=1.0 upper-unmeasured |
| 2.0 | 20 | 10.0 | 0.99 | ✓ observed=1.0 upper-unmeasured |
| 2.0 | 20 | 20.0 | 0.9 | ✓ observed=1.0 upper-unmeasured |
| 2.0 | 20 | 20.0 | 0.95 | ✓ observed=1.0 upper-unmeasured |
| 2.0 | 20 | 20.0 | 0.99 | ✓ observed=1.0 upper-unmeasured |
| 2.0 | 30 | 3.0 | 0.9 | ✗ |
| 2.0 | 30 | 3.0 | 0.95 | ✗ |
| 2.0 | 30 | 3.0 | 0.99 | ✗ |
| 2.0 | 30 | 5.0 | 0.9 | ✓ observed=1.0 upper-unmeasured |
| 2.0 | 30 | 5.0 | 0.95 | ✓ observed=1.0 upper-unmeasured |
| 2.0 | 30 | 5.0 | 0.99 | ✗ |
| 2.0 | 30 | 10.0 | 0.9 | ✓ observed=1.0 upper-unmeasured |
| 2.0 | 30 | 10.0 | 0.95 | ✓ observed=1.0 upper-unmeasured |
| 2.0 | 30 | 10.0 | 0.99 | ✓ observed=1.0 upper-unmeasured |
| 2.0 | 30 | 20.0 | 0.9 | ✓ observed=1.0 upper-unmeasured |
| 2.0 | 30 | 20.0 | 0.95 | ✓ observed=1.0 upper-unmeasured |
| 2.0 | 30 | 20.0 | 0.99 | ✓ observed=1.0 upper-unmeasured |
| 2.0 | 50 | 3.0 | 0.9 | ✗ |
| 2.0 | 50 | 3.0 | 0.95 | ✗ |
| 2.0 | 50 | 3.0 | 0.99 | ✗ |
| 2.0 | 50 | 5.0 | 0.9 | ✓ observed=1.0 upper-unmeasured |
| 2.0 | 50 | 5.0 | 0.95 | ✓ observed=1.0 upper-unmeasured |
| 2.0 | 50 | 5.0 | 0.99 | ✗ |
| 2.0 | 50 | 10.0 | 0.9 | ✓ observed=1.0 upper-unmeasured |
| 2.0 | 50 | 10.0 | 0.95 | ✓ observed=1.0 upper-unmeasured |
| 2.0 | 50 | 10.0 | 0.99 | ✓ observed=1.0 upper-unmeasured |
| 2.0 | 50 | 20.0 | 0.9 | ✓ observed=1.0 upper-unmeasured |
| 2.0 | 50 | 20.0 | 0.95 | ✓ observed=1.0 upper-unmeasured |
| 2.0 | 50 | 20.0 | 0.99 | ✓ observed=1.0 upper-unmeasured |

## Cost view (research assumptions)

Hourly cost in units of p_4090: n_4090 + r·n_3090, r ∈ [0.5, 0.75, 0.94, 1.0] (all ratios are research assumptions; 0.94 ≈ launch MSRP 1499/1599 amortisation example, others span a sensitivity range; not rental quotes). Fixed-cost comparison is only between candidates with the same cards; 1-GPU candidates are compared via the price model.

| candidate | cards | KV GiB total | r=0.5 occ / idle-billed | r=0.75 occ / idle-billed | r=0.94 occ / idle-billed | r=1.0 occ / idle-billed |
|---|---|---|---|---|---|---|
| C_4090 | 3090+4090 | 4.0 | 1.50 / 1.50 | 1.75 / 1.75 | 1.94 / 1.94 | 2.00 / 2.00 |

alpha_max / cost for a given cell and r = alpha_max_tested(candidate, cell) ÷ cost(candidate, r); computed from the table above and `stage1_cells.csv`, is a tested-grid comparison only; preserve censoring and non-monotone flags rather than claiming exact capacity. GPU-time primary: cards × observation 600 s; Wh per card from gpu.csv (observation window and full replay incl. drain are separate columns in stage1_summary.json).

## Parity (per plan: plan hash, prompt hash, CSV columns, denominators each exactly one non-empty value)

- alpha1_seed901: n=1 ok=True {'plan_sha256': True, 'prompt_sha256': True, 'columns': True, 'denominators': True}
