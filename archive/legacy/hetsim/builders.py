"""Builders for the instance groups the solver can choose from.  Each returns
the list of Groups to hand to Simulator plus, where needed, the router rule."""
from .spec import CardType, Card
from .core import Group, Link
from .engine_llm import BatchInstance, HostTier, least_load
from .model import llm

CARD_4090 = CardType("RTX4090", sm=128, mem_gib=24.0, tflops=165.0, bw_gbps=1008.0)


def llm_colo(n, card=CARD_4090, spec=llm.LLAMA31_8B, prof=llm.RTX4090, host=None,
             name="llm/colo", tenant="llm", id0=0):
    inst = [BatchInstance(f"{tenant}{id0+i}", Card(f"g{id0+i}", card), spec, prof, host=host)
            for i in range(n)]
    return [Group(name, tenant, inst, discipline="fcfs", selector=least_load(inst))]


def llm_pd(n_prefill, n_decode, card=CARD_4090, spec=llm.LLAMA31_8B, prof=llm.RTX4090,
           link=None, host=None, tenant="llm", id0=0):
    """Prefill/decode disaggregation.  Requests enter the prefill group; a
    finished prefill is handed to the decode group, which admits it on the
    least-loaded decode instance with room for its whole context, or parks it
    in the group queue until one frees up (DistServe staging)."""
    link = link or Link("pcie", bw_gbps=16.0, latency_s=20e-6)
    dec = [BatchInstance(f"{tenant}d{id0+i}", Card(f"g{id0+i}", card), spec, prof,
                         role="decode", host=host) for i in range(n_decode)]
    dgroup = Group(f"{tenant}/pd-decode", tenant, dec, discipline="fcfs",
                   selector=least_load(dec), hol=False)
    pre = [BatchInstance(f"{tenant}p{id0+i}", Card(f"g{id0+n_decode+i}", card), spec, prof,
                         role="prefill", link=link, handoff=lambda r, t: dgroup.enqueue(r))
           for i in range(n_prefill)]
    pgroup = Group(f"{tenant}/pd-prefill", tenant, pre, discipline="fcfs", selector=least_load(pre))
    return [pgroup, dgroup]


def pd_rule(groups):
    """Router rule: every request of a tenant enters its prefill group (or its
    only group)."""
    entry = {}
    for g in groups:
        if g.name.endswith("pd-decode"):
            continue
        entry.setdefault(g.tenant, g.name)
    return lambda req: entry[req.tenant]
