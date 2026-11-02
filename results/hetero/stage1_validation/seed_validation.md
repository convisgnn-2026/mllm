# Seed validation (Stage 1)

seeds [901, 902, 903]; candidates ['S4_4090', 'S4_3090', 'S3_4090', 'C_4090']; alphas common to all seeds per candidate: S4_4090 [0.5, 0.6, 0.75, 1.0]; S4_3090 [0.5, 0.6, 0.75, 1.0]; S3_4090 [0.5, 0.6, 0.75, 1.0]; C_4090 [0.5, 1.0, 1.1, 1.25]

| objective | cells with a winner on some seed | same winner on all seeds |
|---|---|---|
| α_max | 105 | 64 |
| α_max / cost (r=0.75) | 105 | 67 |

| candidate | cells with identical α_max on all seeds | cells that differ |
|---|---|---|
| S4_4090 | 146 | 46 |
| S4_3090 | 147 | 45 |
| S3_4090 | 168 | 24 |
| C_4090 | 156 | 36 |

Most frequent disagreements (α_max objective):

- {'901': 'S4_3090', '902': 'S4_3090', '903': 'none'}: 8 cells
- {'901': 'none', '902': 'S4_3090', '903': 'none'}: 4 cells
- {'901': 'C_4090', '902': 'tie:S4_4090+C_4090', '903': 'C_4090'}: 4 cells
- {'901': 'none', '902': 'S4_4090', '903': 'none'}: 3 cells
- {'901': 'C_4090', '902': 'C_4090', '903': 'tie:S4_4090+C_4090'}: 3 cells
- {'901': 'C_4090', '902': 'C_4090', '903': 'tie:S4_3090+C_4090'}: 2 cells
- {'901': 'C_4090', '902': 'C_4090', '903': 'S4_3090'}: 2 cells
- {'901': 'C_4090', '902': 'tie:S4_3090+C_4090', '903': 'S4_3090'}: 2 cells
- {'901': 'C_4090', '902': 'C_4090', '903': 'S4_4090'}: 2 cells
- {'901': 'none', '902': 'S4_4090', '903': 'S4_4090'}: 1 cells
