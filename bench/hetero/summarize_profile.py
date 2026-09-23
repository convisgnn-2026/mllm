#!/usr/bin/env python3
"""profile_llm.sh 가 남긴 JSON 을 카드별 표로 정리한다.

  python bench/hetero/summarize_profile.py            # results/hetero/profile_llm/* 전부
  python bench/hetero/summarize_profile.py 4090 3090  # 카드 지정, 있으면 비율 열 추가

prefill_in{S}         : batch 1, output 1          → 지연 = 프리필 시간 T_pre(S)
decode_b{B}_in128     : output 256, batch B        → 스텝 시간 ≈ (지연 − T_pre(128)) / 255
decode_b8_in{S}       : output 256, batch 8        → 스텝 시간 ≈ (지연 − T_pre(S)) / 255
"""
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "hetero", "profile_llm")
OUT_LEN = 256


def load(card):
    d = os.path.join(ROOT, card)
    tab = {}
    if not os.path.isdir(d):
        return tab
    for f in os.listdir(d):
        if f.endswith(".json"):
            j = json.load(open(os.path.join(d, f)))
            tab[f[:-5]] = (j["avg_latency"], j["percentiles"]["50"], j["percentiles"]["99"])
    return tab


def ms(x):
    return f"{x * 1e3:8.1f}"


def main():
    cards = sys.argv[1:] or sorted(os.listdir(ROOT)) if os.path.isdir(ROOT) else []
    tabs = {c: load(c) for c in cards}
    cards = [c for c in cards if tabs[c]]
    if not cards:
        print("no results"); return
    ratio = len(cards) == 2

    def row(label, key, fn):
        cells, vals = [], []
        for c in cards:
            v = fn(tabs[c], key)
            vals.append(v); cells.append(ms(v) if v is not None else "       -")
        if ratio and None not in vals and vals[0]:
            cells.append(f"{vals[1] / vals[0]:6.2f}x")
        print(f"| {label:<14} | " + " | ".join(cells) + " |")

    hdr = "| " + " | ".join(["", *[f"{c} ms" for c in cards], *([f"{cards[1]}/{cards[0]}"] if ratio else [])]) + " |"
    sep = "|---" * (1 + len(cards) + (1 if ratio else 0)) + "|"

    print("## Prefill time  T_pre(S_in)  [batch 1, output 1]\n")
    print(hdr); print(sep)
    for s in (128, 256, 512, 1024, 2048, 4096):
        row(f"S_in={s}", f"prefill_in{s}", lambda t, k: t[k][0] if k in t else None)

    def step(t, key, pre_key):
        if key not in t or pre_key not in t:
            return None
        return (t[key][0] - t[pre_key][0]) / (OUT_LEN - 1)

    print("\n## Decode step time vs batch  [input 128, output 256]\n")
    print(hdr); print(sep)
    for b in (1, 2, 4, 8, 16, 32, 64):
        row(f"B={b}", f"decode_b{b}_in128", lambda t, k: step(t, k, "prefill_in128"))

    print("\n## Decode step time vs context  [batch 8, output 256]\n")
    print(hdr); print(sep)
    for s in (128, 512, 1024, 2048, 4096):
        row(f"S_in={s}", f"decode_b8_in{s}", lambda t, k: step(t, k, f"prefill_in{s}"))

    print("\n## Raw E2E latency (avg / p50 / p99 ms)\n")
    for c in cards:
        print(f"### {c}")
        for k in sorted(tabs[c]):
            a, p50, p99 = tabs[c][k]
            print(f"- {k:<20} {ms(a)} / {ms(p50)} / {ms(p99)}")


if __name__ == "__main__":
    main()
