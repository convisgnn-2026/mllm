#!/usr/bin/env python
"""
Queue disciplines, as a plug-in.

Each policy answers one question: of the requests waiting, which goes next?
They differ in what they optimise, and that difference is the point -- a policy
that maximises the aggregate hit rate is not the one that protects the client
who sends the expensive requests.

  fcfs   arrival order.  vLLM's default; the null policy.
  sjf    shortest job first.  Maximises the aggregate hit rate by clearing
         cheap work, which is exactly how the expensive client gets buried.
  edf    earliest deadline first.  Classic real-time; optimal when every
         deadline is feasible, known to degrade under overload.
  vtc    fair queueing, after Sheng et al. (OSDI'24).  Their Virtual Token
         Counter tracks cumulative tokens served per client and always serves
         the client that has received least; we count GPU-seconds instead of
         tokens, which is the same idea with the service unit that fits video
         diffusion.  A client returning from idle has its counter lifted to the
         minimum among active clients, or it would monopolise on re-entry.

A client here is a request class -- a tenant is characterised by the kind of
request it sends (a shorts service sends 480p clips; a preview service sends
720p), which is how the mapping from template to client is meant.
"""


class Policy:
    """pick() chooses from the pending list; on_dispatch() updates state."""

    def __init__(self, cost, slo=None):
        self.cost = cost                 # template -> full-GPU service seconds
        self.slo = slo or {}             # template -> deadline seconds

    def pick(self, pending, now):
        raise NotImplementedError

    def on_arrival(self, req):
        pass

    def on_dispatch(self, req):
        pass


class FCFS(Policy):
    name = "fcfs"

    def pick(self, pending, now):
        return 0


class SJF(Policy):
    name = "sjf"

    def pick(self, pending, now):
        return min(range(len(pending)), key=lambda i: self.cost[pending[i]["tpl"]])


class EDF(Policy):
    name = "edf"

    def pick(self, pending, now):
        return min(range(len(pending)),
                   key=lambda i: pending[i]["arrival_s"] + self.slo[pending[i]["tpl"]])


class VTC(Policy):
    """Serve the client that has received the least service so far."""
    name = "vtc"

    def __init__(self, cost, slo=None):
        super().__init__(cost, slo)
        self.served = {}                 # client -> cumulative GPU-seconds
        self.active = set()              # clients with something pending

    @staticmethod
    def client(req):
        return req["tpl"]

    def on_arrival(self, req):
        c = self.client(req)
        if c not in self.served:
            self.served[c] = 0.0
        if c not in self.active and self.active:
            # a client back from idle starts no earlier than the current front,
            # otherwise its stale low counter would let it monopolise
            self.served[c] = max(self.served[c],
                                 min(self.served[k] for k in self.active))
        self.active.add(c)

    def pick(self, pending, now):
        return min(range(len(pending)),
                   key=lambda i: (self.served.get(self.client(pending[i]), 0.0),
                                  pending[i]["arrival_s"]))

    def on_dispatch(self, req):
        c = self.client(req)
        self.served[c] = self.served.get(c, 0.0) + self.cost[req["tpl"]]

    def note_pending(self, pending):
        """Clients with nothing waiting are no longer active."""
        self.active = {self.client(r) for r in pending}


REGISTRY = {p.name: p for p in (FCFS, SJF, EDF, VTC)}


def make(name, cost, slo=None):
    if name not in REGISTRY:
        raise ValueError(f"unknown queue policy {name!r}; have {sorted(REGISTRY)}")
    return REGISTRY[name](cost, slo)
