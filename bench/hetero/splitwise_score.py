#!/usr/bin/env python3
"""Splitwise(ISCA'24) 평가 조건 그대로 재채점.
SLO = 요청별 무경합 지연(느린 카드 = 3090, 스텝 모델) 대비 배율, 아홉 개 모두 만족해야 통과:
   TTFT p50/p90/p99 ≤ 2/3/6×,  TBT p50/p90/p99 ≤ 1.25/1.5/5×,  E2E p50/p90/p99 ≤ 1.25/1.5/5×
지표 = 아홉 SLO 를 모두 만족하는 최대 도착률.
  python bench/hetero/splitwise_score.py <cell dir>...      (셀별 아홉 배율 표 + 통과 여부)
"""
import csv, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_llm import t_pre_raw, KV_TOK, CARDS

REF = "3090"; B1_PRE, B0_PRE = 0.8812, 0.0117; B0_DEC, BKV = 0.02269, 2.251          # results/hetero/model/beta*.json
SLO = {"ttft": (2, 3, 6), "tbt": (1.25, 1.5, 5), "e2e": (1.25, 1.5, 5)}; PCT = (50, 90, 99)


_REF = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "hetero", "uncontended", REF, "ref.json")))
_XS = sorted({int(k.split(":")[0]) for k in _REF}); _TT = [np.mean([v["ttft"] for k, v in _REF.items() if int(k.split(":")[0]) == x]) for x in _XS]
_TP = float(np.median([v["tpot"] for v in _REF.values()]))


def ref(in_len, out_len):
    """실측 무경합(3090, 순차, 프록시 없음): TTFT 는 입력 길이 보간, TPOT 는 중앙값 (results/hetero/uncontended/3090/ref.json)."""
    ttft = max(float(np.interp(in_len, _XS, _TT)), float(os.environ.get("TTFT_FLOOR", "0"))); tbt = _TP   # TTFT_FLOOR: 짧은 프롬프트 기준 하한(s)
    return ttft, tbt, ttft + max(out_len - 1, 0) * tbt


def cell(d):
    rows = [x for x in csv.DictReader(open(os.path.join(d, "requests.csv"))) if not x["err"] and x["first"] not in ("nan", "")]
    R = {"ttft": [], "tbt": [], "e2e": []}
    for x in rows:
        g = int(x.get("usage_completion") or x["out_tokens"]); il = int(x["in_len"])
        tt = float(x["first"]) - float(x["launch"]); tb = (float(x["end"]) - float(x["first"])) / max(g - 1, 1); e2 = float(x["end"]) - float(x["launch"])
        rt, rb, re_ = ref(il, g); R["ttft"].append(tt / rt); R["tbt"].append(tb / rb); R["e2e"].append(e2 / re_)
    out = {}
    for k in R:
        out[k] = [float(np.percentile(R[k], p)) for p in PCT]
    out["pass"] = all(out[k][i] <= SLO[k][i] for k in SLO for i in range(3))
    out["worst"] = max(((out[k][i] / SLO[k][i], f"{k} p{PCT[i]}") for k in SLO for i in range(3)))
    return out


def main():
    print("| 셀 | TTFT p50/p90/p99 (배율, SLO 2/3/6) | TBT p50/p90/p99 (1.25/1.5/5) | E2E p50/p90/p99 (1.25/1.5/5) | 통과 | 최악 |\n|---|---|---|---|---|---|")
    for d in sys.argv[1:]:
        o = cell(d); f = lambda k: " / ".join(f"{v:.1f}" for v in o[k])
        print(f"| {os.path.relpath(d, 'results/hetero/open')} | {f('ttft')} | {f('tbt')} | {f('e2e')} | {'O' if o['pass'] else '–'} | {o['worst'][1]} {o['worst'][0]:.1f}× |")


if __name__ == "__main__":
    main()
