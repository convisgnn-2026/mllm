"""Event loop, instances, groups, router.

    Cluster = { Group }          a Group is one tenant's instances under one
                                 lever (e.g. "dit/full", "llm/pd-decode")
    Group   = queue + Instances  a central pending queue; when an instance can
                                 admit, the group's discipline picks the
                                 request and its selector picks the instance
    Router  = Request -> Group   the solver's assignment rule r_t

Instances expose two methods so that request-level workers (DiT, S1) and
iteration-level batching engines (LLM, S2) live under the same loop:

    can_admit(req, now) -> bool
    admit(req, now)     -> schedules its own events on the simulator

An instance calls `sim.complete(req, now)` when a request finishes.
"""
import heapq
from dataclasses import dataclass, field

from .spec import Card, Request


# ----------------------------------------------------------------- instances
class Instance:
    name: str
    card: Card
    sim: "Simulator"

    def can_admit(self, req: Request, now: float) -> bool:
        raise NotImplementedError

    def admit(self, req: Request, now: float) -> None:
        raise NotImplementedError

    def serves(self, req: Request) -> bool:
        """Static feasibility: could this instance ever run `req`?"""
        raise NotImplementedError

    @property
    def busy_seconds(self):
        return 0.0


class SlotInstance(Instance):
    """One request at a time, deterministic service time (DiT worker)."""

    def __init__(self, name, card, model):
        self.name, self.card, self.model = name, card, model
        self.busy_until = -1.0
        self._busy = 0.0

    def serves(self, req):
        return self.model.feasible(req.cls, self.card.type.mem_gib)

    def can_admit(self, req, now):
        return self.busy_until <= now + 1e-12 and self.serves(req)

    def admit(self, req, now):
        svc = self.model.service(req.cls, self.card.type.sm)
        req.start, req.service, req.instance = now, svc, self.name
        self.busy_until = now + svc
        self._busy += svc
        self.sim.schedule(self.busy_until, self._finish, req)

    def _finish(self, now, req):
        self.sim.complete(req, now)

    @property
    def busy_seconds(self):
        return self._busy

    @property
    def is_free(self):
        return self.busy_until <= self.sim.now + 1e-12


# --------------------------------------------------------------------- links
class Link:
    """A byte pipe between cards or nodes: bandwidth (GB/s) and per-message
    latency.  `contended=True` serialises transfers FIFO on the link (a shared
    node-to-node line); False charges each transfer independently (a private
    PCIe/NVLink path per instance pair, which is what LLMServingSim assumes)."""

    def __init__(self, name, bw_gbps, latency_s=0.0, contended=False):
        self.name, self.bw, self.lat, self.contended = name, bw_gbps * 1e9, latency_s, contended
        self.busy_until = 0.0
        self.bytes = 0

    def transfer(self, nbytes, start, messages=1):
        """Return the completion time of a transfer that may begin at `start`."""
        self.bytes += nbytes
        dur = nbytes / self.bw + messages * self.lat
        if self.contended:
            t0 = max(start, self.busy_until)
            self.busy_until = t0 + dur
            return self.busy_until
        return start + dur


# -------------------------------------------------------------------- groups
def srpt_key(rank):
    """Discipline: shortest reference work first."""
    return lambda req: rank(req)


class Group:
    """A tenant's instances under one lever, with a shared pending queue.

    discipline  "fcfs" | "srpt"     which pending request goes next
    selector    "any" | "match"     which admitting instance takes it:
                any   = first;  match = biggest requests to the biggest card,
                everything else to the smallest card that fits (serve_bench.py)
    """

    def __init__(self, name, tenant, instances, discipline="srpt", selector="match",
                 rank=None, big=frozenset(), hol=True):
        self.name, self.tenant = name, tenant
        self.instances = list(instances)
        self.discipline, self.selector = discipline, selector
        self.rank = rank or (lambda req: 0.0)
        self.big = big
        self.hol = hol                                 # head-of-line blocking on a request nothing can take
        self.pending: list[Request] = []

    def enqueue(self, req):
        self.pending.append(req)

    def covers(self, req):
        return any(i.serves(req) for i in self.instances)

    def dispatch(self, now):
        """Admit as many pending requests as possible.  Mirrors serve_bench.py:
        pick the request by discipline first; if nothing free can hold it,
        stop (head-of-line blocking is part of the measured behaviour)."""
        skipped = []
        while self.pending:
            if self.discipline == "srpt":
                i = min(range(len(self.pending)), key=lambda k: self.rank(self.pending[k]))
            else:
                i = 0
            req = self.pending[i]
            cands = [ins for ins in self.instances if ins.can_admit(req, now)]
            if not cands:
                if self.hol:
                    break
                skipped.append(self.pending.pop(i))
                continue
            self.pending.pop(i)
            if callable(self.selector):
                ins = self.selector(req, cands)
            elif self.selector == "any":
                ins = cands[0]
            else:
                key = lambda ins: ins.card.type.sm
                ins = max(cands, key=key) if req.cls in self.big else min(cands, key=key)
            ins.admit(req, now)
        if skipped:
            self.pending[:0] = skipped


# -------------------------------------------------------------------- router
class Router:
    """Request -> Group.  `rule(req) -> group name`; the default sends every
    request to the tenant's only group."""

    def __init__(self, groups, rule=None):
        self.groups = {g.name: g for g in groups}
        by_tenant = {}
        for g in groups:
            by_tenant.setdefault(g.tenant, []).append(g.name)
        self._by_tenant = by_tenant
        self.rule = rule or self._default

    def _default(self, req):
        names = self._by_tenant.get(req.tenant, [])
        if len(names) != 1:
            raise ValueError(f"tenant {req.tenant!r} has {len(names)} groups; give a rule")
        return names[0]

    def route(self, req):
        return self.groups[self.rule(req)]


# ----------------------------------------------------------------- simulator
@dataclass(order=True)
class _Ev:
    time: float
    seq: int
    fn: object = field(compare=False)
    arg: object = field(compare=False)


class Simulator:
    """Run a request trace through Router -> Groups -> Instances."""

    def __init__(self, groups, router=None):
        self.groups = list(groups)
        self.router = router or Router(self.groups)
        self.now = 0.0
        self._heap: list[_Ev] = []
        self._seq = 0
        self.done: list[Request] = []
        for g in self.groups:
            for ins in g.instances:
                ins.sim = self

    def schedule(self, t, fn, arg=None):
        self._seq += 1
        heapq.heappush(self._heap, _Ev(t, self._seq, fn, arg))

    def complete(self, req, now):
        req.end = now
        self.done.append(req)

    def run(self, requests):
        reqs = sorted(requests, key=lambda r: r.arrival)
        for r in reqs:
            g = self.router.route(r)
            if not g.covers(r):
                raise ValueError(f"group {g.name} cannot serve class {r.cls!r}")
        nxt = 0
        while nxt < len(reqs) or self._heap:
            # next time point: earliest of (next arrival, next event)
            t_arr = reqs[nxt].arrival if nxt < len(reqs) else float("inf")
            t_ev = self._heap[0].time if self._heap else float("inf")
            self.now = min(t_arr, t_ev)
            # arrivals first, then completions -- same order as serve_bench.py
            while nxt < len(reqs) and reqs[nxt].arrival <= self.now + 1e-12:
                self.router.route(reqs[nxt]).enqueue(reqs[nxt])
                nxt += 1
            while self._heap and self._heap[0].time <= self.now + 1e-12:
                ev = heapq.heappop(self._heap)
                ev.fn(self.now, ev.arg)
            for g in self.groups:
                g.dispatch(self.now)
        return Result(self.done, self.groups)


class Result:
    def __init__(self, requests, groups):
        self.requests = requests
        self.groups = groups
        self.instances = [i for g in groups for i in g.instances]
        t0 = min(r.arrival for r in requests)
        self.makespan = max(r.end for r in requests) - t0
        self.busy = sum(i.busy_seconds for i in self.instances)
        self.utilization = self.busy / (len(self.instances) * self.makespan)
