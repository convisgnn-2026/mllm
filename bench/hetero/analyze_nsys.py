#!/usr/bin/env python3
"""Kernel-level analysis of a profile_mixed_nsys.sh capture (decode vLLM + dit_server on the RTX 3090).

  python3 bench/hetero/analyze_nsys.py results/hetero/nsys_c3090/mixed [results/hetero/nsys_c3090/llm ...]

Per run directory (report.sqlite from `nsys export --type sqlite`):
  * identifies the decode engine process (has CUDA graph traces) and the DiT process (most kernels, no graphs);
  * GPU busy time of each, and the time both had a kernel/graph executing simultaneously (overlap);
  * decode step = one CUDA graph execution: duration and gap-to-next distributions; share of each gap that falls
    inside DiT kernel execution ("waiting behind DiT");
  * DiT kernel durations and total; nvidia-smi dmon sm%/mem% over the capture window.
Writes <dir>/nsys_summary.json and prints a table. If several dirs are given, prints them side by side.
"""
import bisect
import json
from pathlib import Path
import sqlite3
import sys


def merge(iv):
    iv.sort(); out = []
    for s, e in iv:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def total(iv):
    return sum(e - s for s, e in iv)


def overlap(a, b):
    i = j = 0; t = 0
    while i < len(a) and j < len(b):
        s = max(a[i][0], b[j][0]); e = min(a[i][1], b[j][1])
        if e > s:
            t += e - s
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return t


def covered(seg, busy, starts):
    """time of interval seg covered by merged busy intervals (starts = [s for s,e in busy])."""
    s, e = seg; t = 0
    k = bisect.bisect_right(starts, s) - 1
    k = max(k, 0)
    while k < len(busy) and busy[k][0] < e:
        bs, be = busy[k]
        if be > s:
            t += min(be, e) - max(bs, s)
        k += 1
    return t


def pct(v, q):
    if not v:
        return None
    v = sorted(v); return v[min(len(v) - 1, int(q * len(v)))]


def analyze(d):
    d = Path(d); c = sqlite3.connect(d / 'report.sqlite')
    procs = {g: (pid, name) for g, pid, name in c.execute('select globalPid, pid, name from PROCESSES')}
    graph_pids = [r[0] for r in c.execute('select globalPid from CUPTI_ACTIVITY_KIND_GRAPH_TRACE group by globalPid')]
    kern_cnt = dict(c.execute('select globalPid, count(*) from CUPTI_ACTIVITY_KIND_KERNEL group by globalPid').fetchall())
    dec = graph_pids[0] if graph_pids else None
    dit = max((g for g in kern_cnt if g != dec), key=lambda g: kern_cnt[g], default=None)
    res = dict(run=d.name, decode_proc=procs.get(dec), dit_proc=procs.get(dit))
    t_lo = c.execute('select min(start) from CUPTI_ACTIVITY_KIND_KERNEL').fetchone()[0]
    t_hi = c.execute('select max(end) from CUPTI_ACTIVITY_KIND_KERNEL').fetchone()[0]
    win = (t_hi - t_lo) / 1e9; res['window_s'] = win
    # decode activity: graphs + non-graph kernels
    dec_busy = []; steps = []
    if dec is not None:
        # graph-trace records extend outside the kernel capture window (buffered since launch); clip to the kernel window
        steps = [(s, e) for s, e in c.execute('select start, end from CUPTI_ACTIVITY_KIND_GRAPH_TRACE where globalPid=? and start>=? and end<=? order by start', (dec, t_lo, t_hi))]
        dk = [(s, e) for s, e in c.execute('select start, end from CUPTI_ACTIVITY_KIND_KERNEL where globalPid=? and graphId is null', (dec,))]
        dec_busy = merge([list(x) for x in steps + dk])
        res['decode'] = dict(n_graph_steps=len(steps), n_nongraph_kernels=len(dk), busy_s=total(dec_busy) / 1e9, busy_frac=total(dec_busy) / 1e9 / win)
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
    # decode *steps* = graph executions >= 2 ms (the many sub-ms graphs are auxiliary launches, see histogram)
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
            # step-level: gaps that contain any DiT kernel vs none
            with_d = [g for g, cv in zip(gaps, cov) if cv > 0.01]; without = [g for g, cv in zip(gaps, cov) if cv <= 0.01]
            res['decode_gap_ms']['gap_with_dit_p50'] = pct(with_d, .5); res['decode_gap_ms']['gap_without_dit_p50'] = pct(without, .5)
            res['decode_gap_ms']['n_gaps_with_dit'] = len(with_d); res['decode_gap_ms']['n_gaps_without_dit'] = len(without)
    # dmon during capture
    try:
        t0 = float((d / 'capture_start_unix.txt').read_text()); t1 = float((d / 'capture_end_unix.txt').read_text())
        gidx = (d / 'dmon_gpu_index.txt').read_text().strip() if (d / 'dmon_gpu_index.txt').exists() else '1'
        import datetime as dt
        sm, mem = [], []
        for line in (d / 'dmon.txt').read_text().splitlines():
            p = line.split()
            if len(p) < 5 or p[0].startswith('#'):
                continue
            # -o T prints HH:MM:SS first; reconstruct date from capture day
            hms = p[0]; gpu = p[1]
            day = dt.datetime.fromtimestamp(t0).strftime('%Y-%m-%d')
            ts = dt.datetime.strptime(f'{day} {hms}', '%Y-%m-%d %H:%M:%S').timestamp()
            if gpu == gidx and t0 <= ts <= t1 and p[2] != '-':
                sm.append(float(p[2])); mem.append(float(p[3]))
        if sm:
            res['dmon_gpu1'] = dict(gpu=gidx, n=len(sm), sm_pct_mean=sum(sm) / len(sm), mem_pct_mean=sum(mem) / len(mem), mem_pct_p90=pct(mem, .9))
    except Exception as ex:  # noqa: BLE001
        res['dmon_gpu1'] = f'unavailable: {ex}'
    (d / 'nsys_summary.json').write_text(json.dumps(res, indent=2, default=str) + '\n')
    return res


def main():
    runs = [analyze(d) for d in sys.argv[1:]]
    def g(r, *ks):
        for k in ks:
            r = r.get(k, {}) if isinstance(r, dict) else {}
        return r if r != {} else None
    rows = [('window s', lambda r: r['window_s']),
            ('decode: graphs (all / >=2 ms steps)', lambda r: None if g(r, 'decode', 'n_graph_steps') is None else f"{g(r,'decode','n_graph_steps')} / {g(r,'decode','n_steps_ge2ms')}"), ('decode busy frac', lambda r: g(r, 'decode', 'busy_frac')),
            ('DiT kernels', lambda r: g(r, 'dit', 'n_kernels')), ('DiT busy frac', lambda r: g(r, 'dit', 'busy_frac')),
            ('DiT kernel us p50/p90/p99', lambda r: None if not g(r, 'dit', 'kernel_us_p50') else f"{g(r,'dit','kernel_us_p50'):.0f}/{g(r,'dit','kernel_us_p90'):.0f}/{g(r,'dit','kernel_us_p99'):.0f}"),
            ('both executing s', lambda r: g(r, 'overlap', 'both_executing_s')), ('overlap / decode busy', lambda r: g(r, 'overlap', 'frac_of_decode_busy')),
            ('union busy frac', lambda r: g(r, 'overlap', 'union_busy_frac')),
            ('decode step ms p50/p90/p99', lambda r: None if not g(r, 'decode_step_ms', 'p50') else f"{g(r,'decode_step_ms','p50'):.2f}/{g(r,'decode_step_ms','p90'):.2f}/{g(r,'decode_step_ms','p99'):.2f}"),
            ('gap to next step ms p50/p90/p99', lambda r: None if not g(r, 'decode_gap_ms', 'p50') else f"{g(r,'decode_gap_ms','p50'):.2f}/{g(r,'decode_gap_ms','p90'):.2f}/{g(r,'decode_gap_ms','p99'):.2f}"),
            ('gap time inside DiT kernels', lambda r: g(r, 'decode_gap_ms', 'inside_dit_kernels_frac')),
            ('gap p50 with / without DiT ms', lambda r: None if g(r, 'decode_gap_ms', 'gap_with_dit_p50') is None else f"{g(r,'decode_gap_ms','gap_with_dit_p50'):.2f} / {g(r,'decode_gap_ms','gap_without_dit_p50')}"),
            ('dmon GPU1 sm% / mem% mean', lambda r: None if not isinstance(r.get('dmon_gpu1'), dict) else f"{r['dmon_gpu1']['sm_pct_mean']:.0f} / {r['dmon_gpu1']['mem_pct_mean']:.0f} (mem p90 {r['dmon_gpu1']['mem_pct_p90']:.0f})")]
    w = max(len(n) for n, _ in rows)
    print(' ' * w + ' | ' + ' | '.join(f"{r['run']:>22s}" for r in runs))
    for name, f in rows:
        vals = []
        for r in runs:
            v = f(r); vals.append('-' if v is None else (f'{v:.3f}' if isinstance(v, float) else str(v)))
        print(f'{name:{w}s} | ' + ' | '.join(f'{v:>22s}' for v in vals))


if __name__ == '__main__':
    main()
