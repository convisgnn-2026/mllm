#!/usr/bin/env python3
"""Node-mode counterpart of analyze_nsys.py (B300 captures use `--cuda-graph-trace=node`, see profile_mixed_nsys.sh).

  python3 bench/hetero/analyze_nsys_node.py results/hetero/nsys_s4_b300/mixed [.../llm .../dit]

Same metrics and the same JSON keys as analyze_nsys.py, but a decode *step* is reconstructed from the per-node kernel
records instead of CUPTI_ACTIVITY_KIND_GRAPH_TRACE: kernels of the vLLM process that carry a graphId are grouped by
(graphId, correlationId) = one graph launch; the step interval is [min start, max end] of that group. Steps >= 2 ms are kept
(as in analyze_nsys.py). Everything else (DiT kernels, overlap, gaps inside DiT kernels, dmon) is computed identically.
Writes <dir>/nsys_summary.json. analyze_nsys.py itself is unchanged.
"""
import json
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).parent))
from analyze_nsys import merge, total, overlap, covered, pct  # noqa: E402


def analyze(d):
    d = Path(d); c = sqlite3.connect(d / 'report.sqlite')
    procs = {g: (pid, name) for g, pid, name in c.execute('select globalPid, pid, name from PROCESSES')}
    kern_cnt = dict(c.execute('select globalPid, count(*) from CUPTI_ACTIVITY_KIND_KERNEL group by globalPid').fetchall())
    graph_pids = [r[0] for r in c.execute('select globalPid from CUPTI_ACTIVITY_KIND_KERNEL where graphId is not null group by globalPid')]
    dec = graph_pids[0] if graph_pids else None
    dit = max((g for g in kern_cnt if g != dec), key=lambda g: kern_cnt[g], default=None)
    res = dict(run=d.name, mode='node', decode_proc=procs.get(dec), dit_proc=procs.get(dit))
    t_lo = c.execute('select min(start) from CUPTI_ACTIVITY_KIND_KERNEL').fetchone()[0]
    t_hi = c.execute('select max(end) from CUPTI_ACTIVITY_KIND_KERNEL').fetchone()[0]
    win = (t_hi - t_lo) / 1e9; res['window_s'] = win
    dec_busy = []; steps = []
    if dec is not None:
        groups = {}
        for s, e, gid, cid in c.execute('select start, end, graphId, correlationId from CUPTI_ACTIVITY_KIND_KERNEL where globalPid=? and graphId is not null', (dec,)):
            k = (gid, cid); g = groups.get(k)
            if g is None:
                groups[k] = [s, e, 1]
            else:
                g[0] = min(g[0], s); g[1] = max(g[1], e); g[2] += 1
        steps = sorted((g[0], g[1]) for g in groups.values())
        nodes_per_step = [g[2] for g in groups.values()]
        dk = [(s, e) for s, e in c.execute('select start, end from CUPTI_ACTIVITY_KIND_KERNEL where globalPid=? and graphId is null', (dec,))]
        gk = [(s, e) for s, e in c.execute('select start, end from CUPTI_ACTIVITY_KIND_KERNEL where globalPid=? and graphId is not null', (dec,))]
        dec_busy = merge([list(x) for x in gk + dk])  # actual kernel execution time (node level), not the step envelope
        res['decode'] = dict(n_graph_steps=len(steps), n_nongraph_kernels=len(dk), n_graph_node_kernels=len(gk),
                             nodes_per_step_p50=pct(nodes_per_step, .5), busy_s=total(dec_busy) / 1e9, busy_frac=total(dec_busy) / 1e9 / win)
        env = merge([list(x) for x in steps + dk])
        res['decode']['envelope_busy_frac'] = total(env) / 1e9 / win  # comparable to analyze_nsys.py busy_frac (graph envelope)
    dit_busy = []
    if dit is not None:
        dkern = [(s, e) for s, e in c.execute('select start, end from CUPTI_ACTIVITY_KIND_KERNEL where globalPid=?', (dit,))]
        durs = [(e - s) / 1e3 for s, e in dkern]
        dit_busy = merge([list(x) for x in dkern])
        res['dit'] = dict(n_kernels=len(dkern), busy_s=total(dit_busy) / 1e9, busy_frac=total(dit_busy) / 1e9 / win,
                          kernel_us_p50=pct(durs, .5), kernel_us_p90=pct(durs, .9), kernel_us_p99=pct(durs, .99), kernel_us_max=max(durs) if durs else None)
    if dec_busy and dit_busy:
        ov = overlap(dec_busy, dit_busy) / 1e9
        res['overlap'] = dict(both_executing_s=ov, frac_of_decode_busy=ov / (total(dec_busy) / 1e9), frac_of_dit_busy=ov / (total(dit_busy) / 1e9),
                              union_busy_frac=(total(dec_busy) + total(dit_busy) - ov * 1e9) / 1e9 / win)
    steps = [x for x in steps if (x[1] - x[0]) >= 2e6]
    if steps:
        res['decode']['n_steps_ge2ms'] = len(steps)
        dur = [(e - s) / 1e6 for s, e in steps]
        gaps = [(steps[i + 1][0] - steps[i][1]) / 1e6 for i in range(len(steps) - 1)]
        res['decode_step_ms'] = dict(p50=pct(dur, .5), p90=pct(dur, .9), p99=pct(dur, .99))
        res['decode_gap_ms'] = dict(p50=pct(gaps, .5), p90=pct(gaps, .9), p99=pct(gaps, .99), sum_s=sum(gaps) / 1e3)
        if dit_busy:
            starts = [s for s, _ in dit_busy]
            cov = [covered((steps[i][1], steps[i + 1][0]), dit_busy, starts) / 1e6 for i in range(len(steps) - 1)]
            res['decode_gap_ms']['inside_dit_kernels_s'] = sum(cov) / 1e3
            res['decode_gap_ms']['inside_dit_kernels_frac'] = sum(cov) / max(1e-9, sum(gaps))
            with_d = [g for g, cv in zip(gaps, cov) if cv > 0.01]; without = [g for g, cv in zip(gaps, cov) if cv <= 0.01]
            res['decode_gap_ms']['gap_with_dit_p50'] = pct(with_d, .5); res['decode_gap_ms']['gap_without_dit_p50'] = pct(without, .5)
            res['decode_gap_ms']['n_gaps_with_dit'] = len(with_d); res['decode_gap_ms']['n_gaps_without_dit'] = len(without)
            # node level: how much of each step's envelope is DiT execution (time-slice share inside a step)
            scov = [covered(st, dit_busy, starts) / 1e6 for st in steps]
            res['decode_step_ms']['dit_inside_step_frac'] = sum(scov) / max(1e-9, sum(dur))
    try:
        t0 = float((d / 'capture_start_unix.txt').read_text()); t1 = float((d / 'capture_end_unix.txt').read_text())
        gidx = (d / 'dmon_gpu_index.txt').read_text().strip() if (d / 'dmon_gpu_index.txt').exists() else '1'
        import datetime as dt
        sm, mem = [], []
        for line in (d / 'dmon.txt').read_text().splitlines():
            p = line.split()
            if len(p) < 5 or p[0].startswith('#'):
                continue
            day = dt.datetime.fromtimestamp(t0).strftime('%Y-%m-%d')
            ts = dt.datetime.strptime(f'{day} {p[0]}', '%Y-%m-%d %H:%M:%S').timestamp()
            if p[1] == gidx and t0 <= ts <= t1 and p[2] != '-':
                sm.append(float(p[2])); mem.append(float(p[3]))
        if sm:
            res['dmon_gpu1'] = dict(gpu=gidx, n=len(sm), sm_pct_mean=sum(sm) / len(sm), mem_pct_mean=sum(mem) / len(mem), mem_pct_p90=pct(mem, .9))
    except Exception as ex:  # noqa: BLE001
        res['dmon_gpu1'] = f'unavailable: {ex}'
    (d / 'nsys_summary.json').write_text(json.dumps(res, indent=2, default=str) + '\n')
    return res


def main():
    runs = [analyze(d) for d in sys.argv[1:]]
    rows = [('window_s', lambda r: r.get('window_s')), ('decode steps >=2ms', lambda r: r.get('decode', {}).get('n_steps_ge2ms')),
            ('decode step p50 ms', lambda r: r.get('decode_step_ms', {}).get('p50')), ('decode step p90 ms', lambda r: r.get('decode_step_ms', {}).get('p90')),
            ('dit inside step frac', lambda r: r.get('decode_step_ms', {}).get('dit_inside_step_frac')),
            ('gap p50 ms', lambda r: r.get('decode_gap_ms', {}).get('p50')), ('gap with dit p50', lambda r: r.get('decode_gap_ms', {}).get('gap_with_dit_p50')),
            ('gap without dit p50', lambda r: r.get('decode_gap_ms', {}).get('gap_without_dit_p50')),
            ('decode busy frac (kernels)', lambda r: r.get('decode', {}).get('busy_frac')), ('decode envelope frac', lambda r: r.get('decode', {}).get('envelope_busy_frac')),
            ('dit busy frac', lambda r: r.get('dit', {}).get('busy_frac')), ('dit kernel p50 us', lambda r: r.get('dit', {}).get('kernel_us_p50')),
            ('dit kernel p99 us', lambda r: r.get('dit', {}).get('kernel_us_p99')), ('both executing frac of dec', lambda r: r.get('overlap', {}).get('frac_of_decode_busy')),
            ('dmon sm% mean', lambda r: (r.get('dmon_gpu1') or {}).get('sm_pct_mean') if isinstance(r.get('dmon_gpu1'), dict) else None)]
    print(f"{'metric':30s}" + ''.join(f"{r['run']:>14s}" for r in runs))
    for name, f in rows:
        print(f"{name:30s}" + ''.join(f"{(f'{v:.3f}' if isinstance(v, float) else str(v)):>14s}" for v in (f(r) for r in runs)))


if __name__ == '__main__':
    main()
