#!/usr/bin/env python3
"""run_ab.sh 결과를 구성 × 셀 표로.
  python bench/hetero/summarize_ab.py            → results/hetero/ab/* 전부
지표: makespan(총 완료 시간), req/s, 출력 tok/s, TTFT p50/p90, TPOT p50/p90, 카드별 GPU 메모리 피크, 오류 수.
A 대비 비율(makespan 기준 = 처리량 이득)을 마지막 열에 붙인다."""
import csv
import glob
import json
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "hetero", "ab")
CONFIGS = ["A", "B", "C", "D", "C_v1runner_cross"]
LABEL = {"A": "복제+RR", "B": "복제+가중 2:1", "C": "PD 4090P→3090D", "D": "PD 3090P→4090D",
         "C_v1runner_cross": "PD 4090P→3090D (구 러너, cross on)"}


def gpu_peak(path):
    peak = {}
    if not os.path.exists(path):
        return peak
    for row in csv.reader(open(path)):
        if len(row) < 4:
            continue
        try:
            idx = row[1].strip(); mem = float(row[3].strip().split()[0])
        except (ValueError, IndexError):
            continue
        peak[idx] = max(peak.get(idx, 0), mem)
    return peak


def main():
    cells = sorted({os.path.basename(p) for c in CONFIGS for p in glob.glob(os.path.join(ROOT, c, "*_n*")) if not p.endswith("_warmup")},
                   key=lambda s: (s.split("_n")[0], int(s.split("_n")[1])))
    for cell in cells:
        print(f"\n## {cell}\n")
        print("| 구성 | makespan s | req/s | out tok/s | TTFT p50/p90 ms | TPOT p50/p90 ms | E2E p50 s | GPU0/GPU1 peak MiB | err | vs A |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for c in CONFIGS:
            f = os.path.join(ROOT, c, cell, "summary.json")
            if not os.path.exists(f):
                continue
            s = json.load(open(f)); g = gpu_peak(os.path.join(ROOT, c, cell, "gpu.csv"))
            af = os.path.join(ROOT, "A", cell, "summary.json"); a = json.load(open(af)) if os.path.exists(af) else None
            ratio = f"{a['makespan_s'] / s['makespan_s']:.2f}x" if a else "-"
            print(f"| {c} {LABEL[c]} | {s['makespan_s']:.1f} | {s['req_per_s']:.2f} | {s['out_tok_per_s']:.0f} | "
                  f"{s['ttft_ms']['p50']:.0f}/{s['ttft_ms']['p90']:.0f} | {s['tpot_ms']['p50']:.1f}/{s['tpot_ms']['p90']:.1f} | "
                  f"{s['e2e_s']['p50']:.1f} | {g.get('0', 0):.0f}/{g.get('1', 0):.0f} | {s['errors']} | {ratio} |")


if __name__ == "__main__":
    main()
