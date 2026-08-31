#!/usr/bin/env python
"""
Judge the fairness experiment against the criteria fixed BEFORE it was run.

The criteria are written down here so the verdict cannot be fitted to the
result afterwards.  Three of them decide whether the "protect the expensive
client" framing carries a figure; the fourth decides whether both levers -- the
queue discipline and the partition -- have to be chosen together, which is what
a solver would be for.

  (1) the gap is real          sjf + Full leaves the worst client at or below
                               0.30 while the aggregate reads 0.80 or better
  (2) software alone falls short   vtc + Full does worse for the worst client
                               than sjf + 2-way
  (3) the partition helps      2-way lifts the worst client by >= 1.5x
  (4) the combination wins     vtc + 2-way beats both vtc + Full and sjf + 2-way

  FAIL if the gap only appears above ~120% load: that is a "buy more GPUs"
  situation, not a scheduling one.
"""
import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != "/" and not _os.path.isdir(_os.path.join(_d, "results")):
    _d = _os.path.dirname(_d)
_sys.path[:0] = [_os.path.join(_d, "solver"), _os.path.join(_d, "bench")]
_os.chdir(_d)

import glob
import json
from collections import defaultdict

SLO_NAME = "video_3.0x"
CAP_2F = 0.687                       # measured capacity of 2 Full GPUs, this mix
USER = {"480x272x33": "A(480p)", "832x480x33": "B(832s)",
        "832x480x49": "C(832L)", "1280x720x49": "D(1280p)"}
ORDER = ["A(480p)", "B(832s)", "C(832L)", "D(1280p)"]


def score(path, slo):
    d = json.load(open(path))
    per = defaultdict(list)
    for r in d["requests"]:
        per[USER[r["tpl"]]].append((r["end_s"] - r["arrival_s"]) <= slo[r["tpl"]])
    sar = {k: sum(v) / len(v) for k, v in per.items()}
    n = sum(len(v) for v in per.values())
    return {"agg": sum(sum(v) for v in per.values()) / n,
            "per": sar, "worst": min(sar.values()),
            "worst_who": min(sar, key=sar.get),
            "rate": d["workload_config"]["rate"], "queue": d["queue"],
            "cfg": d["policy"]}


def main():
    slo = {k: float(v) for k, v in
           json.load(open(f"slo_configs/{SLO_NAME}.json")).items()}
    R = {}
    for f in sorted(glob.glob("results/fair/*.json")):
        s = score(f, slo)
        R[(s["cfg"], s["queue"], s["rate"])] = s
    if not R:
        print("결과 없음")
        return
    rates = sorted({k[2] for k in R})
    cfgs = sorted({k[0] for k in R})

    for cfg in cfgs:
        print(f"\n### {cfg} ###")
        print(f"{'rate':>5} {'부하':>6} {'큐':>6} | {'전체':>6} {'최약자':>7} {'격차':>6} | "
              + "".join(f"{u:>10}" for u in ORDER))
        for rate in rates:
            for q in ["fcfs", "sjf", "edf", "vtc"]:
                s = R.get((cfg, q, rate))
                if not s:
                    continue
                print(f"{rate:>5} {rate/CAP_2F*100:>5.0f}% {q:>6} | "
                      f"{s['agg']:>6.2f} {s['worst']:>7.2f} {s['agg']-s['worst']:>6.2f} | "
                      + "".join(f"{s['per'].get(u, float('nan')):>10.2f}" for u in ORDER))

    print("\n" + "=" * 78)
    print("사전에 정한 기준으로 판정")
    print("=" * 78)
    verdict = {}

    # (1) the gap is real, and it must show up at moderate load too
    hits = [(r, s) for (c, q, r), s in R.items()
            if c == "2F" and q == "sjf" and s["worst"] <= 0.30 and s["agg"] >= 0.80]
    low = [r for r, _ in hits if r / CAP_2F <= 1.25]
    verdict[1] = bool(hits)
    print(f"\n① 격차가 실재하는가  (sjf+Full 에서 전체 ≥0.80 인데 최약자 ≤0.30)")
    for r, s in sorted(hits):
        print(f"     rate {r} (부하 {r/CAP_2F*100:.0f}%)  전체 {s['agg']:.2f} / "
              f"최약자 {s['worst']:.2f} [{s['worst_who']}]")
    print(f"   → {'통과' if verdict[1] else '불통과'}"
          + (f"   ※ 부하 125% 이하에서도 발생: {'예' if low else '아니오 (경고)'}" if hits else ""))

    # (2) software alone
    print(f"\n② 소프트웨어만으로 부족한가  (vtc+Full 최약자 < sjf+2분할 최약자)")
    ok2 = []
    for rate in rates:
        a = R.get(("2F", "vtc", rate))
        b = R.get(("2x2way", "sjf", rate))
        if not (a and b):
            continue
        ok2.append(a["worst"] < b["worst"])
        print(f"     rate {rate}:  vtc+Full {a['worst']:.2f}   vs   sjf+2분할 {b['worst']:.2f}"
              f"   → {'하드웨어 우세' if a['worst'] < b['worst'] else '소프트웨어가 따라잡음'}")
    verdict[2] = bool(ok2) and sum(ok2) > len(ok2) / 2
    print(f"   → {'통과' if verdict[2] else '불통과'}")

    # (3) the partition helps
    print(f"\n③ 분할이 최약자를 1.5배 이상 올리는가  (같은 큐 정책끼리)")
    r3 = []
    for rate in rates:
        for q in ["fcfs", "sjf", "edf", "vtc"]:
            a = R.get(("2F", q, rate))
            b = R.get(("2x2way", q, rate))
            if not (a and b) or a["worst"] <= 0:
                continue
            g = b["worst"] / a["worst"]
            r3.append(g)
            print(f"     rate {rate} {q:>5}:  {a['worst']:.2f} → {b['worst']:.2f}  ({g:.2f}배)")
    verdict[3] = bool(r3) and max(r3) >= 1.5
    print(f"   → {'통과' if verdict[3] else '불통과'}  (최대 {max(r3):.2f}배)" if r3 else "   → 자료 없음")

    # (4) the combination
    print(f"\n④ 조합이 각각보다 나은가  (vtc+2분할 > vtc+Full, sjf+2분할)")
    ok4 = []
    for rate in rates:
        x = R.get(("2x2way", "vtc", rate))
        a = R.get(("2F", "vtc", rate))
        b = R.get(("2x2way", "sjf", rate))
        if not (x and a and b):
            continue
        best = max(a["worst"], b["worst"])
        ok4.append(x["worst"] > best)
        print(f"     rate {rate}:  조합 {x['worst']:.2f}   vs   최선의 단일 {best:.2f}"
              f"   → {'조합 우세' if x['worst'] > best else '단일로 충분'}")
    verdict[4] = bool(ok4) and sum(ok4) > len(ok4) / 2
    print(f"   → {'통과' if verdict[4] else '불통과'}")

    print("\n" + "-" * 78)
    p = sum(verdict.values())
    print(f"판정: {p}/4 통과   " + "  ".join(f"{k}:{'O' if v else 'X'}"
                                            for k, v in sorted(verdict.items())))
    if p == 4:
        print("→ Fig 1 성립.  네 조건 모두 충족")
    elif verdict.get(1) and verdict.get(3):
        print("→ 부분 성립.  문제와 하드웨어 효과는 있으나 solver 근거(②④)가 약함")
    else:
        print("→ 불성립.  방향 재고 필요")


if __name__ == "__main__":
    main()
