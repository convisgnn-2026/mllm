#!/usr/bin/env python3
"""Zero-parameter interference model derived from the Nsight diagnostic (inter-process time-slicing), checked
against the measured k of every shared placement/alpha/metric (k = mixed p90 / solo p90, from analyze_k_model).

Mechanism (results/hetero/nsys_*): when two processes share a GPU, the GPU alternates between them in ~2.3 ms
slices; kernels do not slow down, but a step takes (1 + p) times longer where p is the probability that the partner
has work ready during the step. Decode additionally loses ~2 slices per step in the inter-step sync (gap 1 -> 5.5 ms).

Inputs per shared (placement, alpha): solo busy fractions on the shared GPU (nvidia-smi, from the resident-solo runs)
u_me (own stage) and u_ot (partner), and the solo p50 of the metric. No fitted parameter.
  TS0: p = u_ot                                   (partner readiness = its solo busy fraction)
  TS1: feedback  a <- min(1, u_me (1+b)), b <- min(1, u_ot (1+a))  iterated; p = b
       (each process's occupancy inflates by the partner's share, so the partner is ready more often than solo)
  k_step = 1 + p
  TPOT: k = (k_step (T - g) + g + 2 tau p) / T    with T = solo TPOT p50, g = 1.0 ms, tau = 2.3 ms
  TTFT, DiT E2E: k = 1 + p
  TS2: p = 1 (k measured is a p90 ratio; the p90 request is one whose steps overlapped partner activity, so the
       partner was ready throughout it, whatever its mean busy fraction). TTFT: only the prefill-compute share of
       TTFT is inflated; approximated with k = 1 + p * u_me_rel where u_me_rel = min(1, 2 u_me) (crude).
  TS3 (burst-length rule): p = 1 when the partner's activity bursts are much longer than my step (decode step of
       7-27 ms vs DiT image of 1-3 s; DiT image vs a decode-bearing LLM that is active for whole requests), p = TS1
       mean when the partner's bursts are short relative to my duration (DiT image vs prefill-only GPU, bursts ~TTFT).
       TTFT: p = TS1 mean (TTFT has large non-compute parts: queue, KV transfer; crude).
  saturation flag: inflated demand a + b >= 1 from the TS1 feedback iteration (the two processes together ask for
  more than one GPU; prediction not trusted, expect queueing beyond the model)

  python3 bench/hetero/analyze_k_timeshare.py   -> results/hetero/k_model/k_timeshare.{md,json,png}
"""
import json
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import analyze_k_model as K  # noqa: E402
import stage2_report as S2  # noqa: E402

G_MS = 1.0; TAU_MS = 2.3


def feedback(u_me, u_ot):
    a, b = u_me, u_ot
    for _ in range(100):
        a2 = min(1.0, u_me * (1 + b)); b2 = min(1.0, u_ot * (1 + a2))
        if abs(a2 - a) < 1e-6 and abs(b2 - b) < 1e-6:
            break
        a, b = a2, b2
    return a, b


def predict(metric, u_me, u_ot, T_ms, variant, partner_long=True):
    if variant == 'TS0':
        p = u_ot
    elif variant == 'TS1':
        p = feedback(u_me, u_ot)[1]
    elif variant == 'TS2':
        p = 1.0
    else:  # TS3
        p = 1.0 if (metric != 'ttft' and partner_long) else feedback(u_me, u_ot)[1]
    p = min(1.0, p)
    if metric == 'tpot':
        return ((1 + p) * (T_ms - G_MS) + G_MS + 2 * TAU_MS * p) / T_ms
    return 1 + p


def main():
    out = K.ROOT / 'results/hetero/k_model'
    prev = json.loads((out / 'k_model.json').read_text())
    solo, _ = K.runs()
    pts = []
    for r in prev['rows']:
        c, al = r['candidate'], r['alpha']
        sl = solo[('llm', c, al)]; sd = solo[('dit', c, al)]
        l50 = K.p50(sl); d50 = K.p50(sd)
        for m in S2.METRICS:
            f = r['feat'][m]
            if f['u'] <= 0.03:
                continue
            T = l50['tpot'] * 1e3 if m == 'tpot' else None
            fa, fb = feedback(f['uo'], f['u'])
            row = dict(c=c, al=al, m=m, k=r['k'][m], u_ot=f['u'], u_me=f['uo'], T_ms=T, demand=fa + fb, sat=(fa + fb) >= 1.0)
            st = K.STAGES[c]
            # partner bursts long relative to my step? TPOT: partner is DiT (image >> step). DiT: partner LLM on the DiT GPU
            # is long-burst only if that GPU runs decode (active for whole requests); prefill-only GPUs give short bursts.
            partner_long = True if m == 'tpot' else (st['decode'] == st['dit'])
            row['partner_long'] = partner_long
            for v in ('TS0', 'TS1', 'TS2', 'TS3'):
                row[v] = predict(m, f['uo'], f['u'], T, v, partner_long)
            pts.append(row)
    def summary(sel, v):
        e = sorted(abs(p[v] / p['k'] - 1) for p in sel)
        return dict(n=len(e), median=e[len(e) // 2] if e else None, p90=e[int(0.9 * (len(e) - 1))] if e else None, within25=sum(x <= 0.25 for x in e))
    res = dict(points=pts, summary={})
    L = ['# k from the time-slicing mechanism, no fitted parameter', '',
         'k measured = mixed p90 / solo p90 (same points as k_model.md). u_me / u_ot = solo busy fraction of own stage / partner on the shared GPU (nvidia-smi).',
         'TS0: p = u_ot. TS1: feedback-inflated partner readiness. k_step = 1 + p; TPOT adds 2 slices (2.3 ms) of sync loss per step; TTFT and DiT k = 1 + p. sat = u_me + u_ot >= 0.9.', '']
    for scope, sel in (('all shared points', pts), ('unsaturated (inflated demand < 1)', [p for p in pts if not p['sat']]), ('saturated', [p for p in pts if p['sat']])):
        L += [f'## {scope}', '', '| model | n | median abs rel err | p90 | within 25% | median by metric ttft / tpot / dit |', '|---|---|---|---|---|---|']
        for v in ('TS0', 'TS1', 'TS2', 'TS3'):
            s = summary(sel, v); bym = [summary([p for p in sel if p['m'] == m], v)['median'] for m in S2.METRICS]
            res['summary'][f'{scope}|{v}'] = dict(s, by_metric=dict(zip(S2.METRICS, bym)))
            if s['n']:
                L.append(f"| {v} | {s['n']} | {s['median']:.0%} | {s['p90']:.0%} | {s['within25']}/{s['n']} | " + ' / '.join('-' if b is None else f'{b:.0%}' for b in bym) + ' |')
        L.append('')
    fitted = {n: f"{prev['fits'][n]['median']:.0%} (p90 {prev['fits'][n]['p90']:.0%}, {prev['fits'][n]['within25']}/{prev['fits'][n]['n']})" for n in prev['fits']}
    L += ['Reference, fitted models with leave-one-placement-out (k_model.md, alpha <= 1.25 only): ' + ', '.join(f'{n} {v}' for n, v in fitted.items()), '',
          '## Points', '', '| candidate | alpha | metric | u_me | u_ot | demand | sat | k measured | TS0 | TS1 | TS2 | TS3 | err TS3 |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for p in pts:
        L.append(f"| {p['c']} | {p['al']} | {p['m']} | {p['u_me']:.2f} | {p['u_ot']:.2f} | {p['demand']:.2f} | {'y' if p['sat'] else ''} | {p['k']:.2f} | {p['TS0']:.2f} | {p['TS1']:.2f} | {p['TS2']:.2f} | {p['TS3']:.2f} | {p['TS3'] / p['k'] - 1:+.0%} |")
    (out / 'k_timeshare.json').write_text(json.dumps(res, indent=2) + '\n'); (out / 'k_timeshare.md').write_text('\n'.join(L) + '\n')
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.3)); col = {c: plt.cm.tab10(i) for i, c in enumerate(list(K.STAGES))}
    for ax, m in zip(axs, S2.METRICS):
        sel = [p for p in pts if p['m'] == m]
        for p in sel:
            ax.plot(p['k'], p['TS3'], 's' if not p['sat'] else 'x', color=col[p['c']], ms=6, mfc=col[p['c']] if not p['sat'] else 'none')
        lim = [0.9, max(max(p['k'], p['TS1']) for p in sel) * 1.3]
        ax.plot(lim, lim, 'k-', lw=0.6); ax.plot(lim, [l * 1.25 for l in lim], 'k:', lw=0.5); ax.plot(lim, [l / 1.25 for l in lim], 'k:', lw=0.5)
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(lim); ax.set_ylim(lim)
        s = summary([p for p in sel if not p['sat']], 'TS3'); ax.set_title(f"{m.upper()}: TS3 median err {s['median']:.0%} (unsaturated, n={s['n']})", fontsize=8)
        ax.set_xlabel('k measured'); ax.spines[['top', 'right']].set_visible(False)
    axs[0].set_ylabel('k predicted, TS3 time-slicing rule (no fit)')
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=col[c], marker='s', ls='', label=c) for c in sorted({p['c'] for p in pts}, key=list(K.STAGES).index)] + [Line2D([], [], color='k', marker='x', ls='', label='saturated (inflated demand >= 1)')]
    fig.legend(handles=h, loc='upper center', ncol=len(h), frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.04)); fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / 'k_timeshare.png', dpi=200, bbox_inches='tight')
    print('\n'.join(L[:L.index('## Points')]))


if __name__ == '__main__':
    main()
