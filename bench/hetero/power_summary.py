#!/usr/bin/env python3
"""셀별 gpu.csv(nvidia-smi 샘플: timestamp,index,util,mem[,power,sm_clk,mem_clk]) → 카드별 평균·피크 전력, 에너지(Wh), 요청당 에너지.
전력 열이 없는 옛 로그는 건너뛴다.   python bench/hetero/power_summary.py results/hetero/open/*/mixed_azure_r*
"""
import csv, json, os, sys
import numpy as np

IDLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "hetero", "power_idle.json")


def cell(d):
    f = os.path.join(d, "gpu.csv")
    if not os.path.exists(f): return None
    P = {}; T = {}
    for row in csv.reader(open(f)):
        if len(row) < 5: return None                                  # 전력 열 없음
        try: g = int(row[1]); p = float(row[4].replace("W", "").strip())
        except ValueError: continue
        P.setdefault(g, []).append(p); T.setdefault(g, []).append(row[0].strip())
    if not P: return None
    s = json.load(open(os.path.join(d, "summary.json"))) if os.path.exists(os.path.join(d, "summary.json")) else {}
    mk = s.get("makespan_s"); out = {}
    for g, v in P.items():
        v = np.array(v); dt = (mk / len(v)) if mk else 1.0
        out[g] = dict(mean_w=float(v.mean()), p95_w=float(np.percentile(v, 95)), peak_w=float(v.max()), samples=len(v), energy_wh=float(v.sum() * dt / 3600))
    tot = sum(o["energy_wh"] for o in out.values())
    return dict(gpu=out, total_wh=tot, wh_per_req=(tot / s["ok"]) if s.get("ok") else None, mean_total_w=sum(o["mean_w"] for o in out.values()))


def main():
    idle = json.load(open(IDLE)) if os.path.exists(IDLE) else {}
    print("| 셀 | 4090 평균/피크 W | 3090 평균/피크 W | 합계 평균 W | 에너지 Wh | Wh/요청 |\n|---|---|---|---|---|---|")
    for d in sys.argv[1:]:
        r = cell(d)
        if not r: print(f"| {d} | (전력 열 없음) | | | | |"); continue
        g0, g1 = r["gpu"].get(0, {}), r["gpu"].get(1, {})
        print(f"| {os.path.relpath(d)} | {g0.get('mean_w', 0):.0f} / {g0.get('peak_w', 0):.0f} | {g1.get('mean_w', 0):.0f} / {g1.get('peak_w', 0):.0f} | {r['mean_total_w']:.0f} | {r['total_wh']:.1f} | {r['wh_per_req'] or 0:.3f} |")
        json.dump(r, open(os.path.join(d, "power.json"), "w"), indent=1)
    if idle: print("\n유휴(모델 적재, 요청 없음):", f"4090 {idle['4090_loaded_idle_w']:.0f} W, 3090 {idle['3090_loaded_idle_w']:.0f} W")


if __name__ == "__main__":
    main()
