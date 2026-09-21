"""Iteration-level LLM engine: vLLM-style continuous batching with chunked
prefill, block KV allocation, full-ISL admission gate, tail preemption and
prefix-cache recovery of a preempted request's blocks (LRU eviction) -- the
scheduler LLMServingSim 2.0 reproduces, reduced to what moves the step time.
Cross-request prefix sharing is not modelled (ShareGPT hit ratio 1-2%).

    admit()  -> request joins this instance's waiting queue (routing happened
                at arrival, as in vLLM multi-instance deployments)
    _step()  -> build one iteration, schedule its end, apply results, repeat

Per-request state lives in req.attrs:
    input, output      lengths (given)
    computed           tokens whose KV is resident
    generated          tokens emitted so far
    blocks             KV blocks currently allocated
    cached             blocks kept in the free pool after preemption (recoverable)
    ttft, itl, tpot    timings
"""
from .core import Instance
from .spec import Request


class HostTier:
    """KV parking tier (host memory reached over PCIe): evicted blocks are
    written through off the critical path and recalled at `bw_gbps` on resume."""

    def __init__(self, bw_gbps=50.0, capacity_gib=128.0):
        self.bw = bw_gbps * 1e9
        self.capacity = capacity_gib * (1024 ** 3)
        self.used = 0
        self.recalled_bytes = 0


class BatchInstance(Instance):
    """role: "colo" (prefill+decode), "prefill" (hands finished prefills to
    `handoff(req, now)`), "decode" (takes them via `admit_decode`)."""

    def __init__(self, name, card, spec, profile, max_seqs=256, token_budget=2048, block=16,
                 slowdown=1.0, role="colo", link=None, host=None, handoff=None):
        self.name, self.card, self.spec, self.profile = name, card, spec, profile
        self.max_seqs, self.budget, self.block = max_seqs, token_budget, block
        self.slowdown = slowdown                       # colocation share (S4): step time x 1/share
        self.role, self.link, self.host, self.handoff = role, link, host, handoff
        self.cap_tokens = profile.kv_capacity_tokens(spec, card.type.mem_gib, block)
        self.free_blocks = self.cap_tokens // block    # unallocated, incl. cached
        self._lru: list[list] = []                     # [owner_req | None, n_blocks], head = evict first
        self._cached_total = 0
        self.waiting: list[Request] = []
        self.running: list[Request] = []
        self.in_step = False
        self._busy = 0.0
        self.steps = 0
        self.preemptions = 0
        self.recomputed = 0
        self.recalled = 0                              # tokens recovered from the host tier

    # ---------------------------------------------------------------- Instance API
    def serves(self, req):
        return req.attrs["input"] + 1 <= self.cap_tokens

    def can_admit(self, req, now):
        if self.role == "decode":                      # P->D handoff: whole context must fit now
            return self._blocks_for(req.attrs["input"] + 1) <= self.free_blocks
        return True                                    # instance-internal queue

    def admit_decode(self, req, now):
        """Take over a request whose prefill ran elsewhere (KV already shipped)."""
        st = req.attrs
        st["blocks"] = 0
        ok = self._alloc(req, st["input"] + 1)
        assert ok
        req.instance = self.name
        self.running.append(req)
        if not self.in_step:
            self._step(now)

    def admit(self, req, now):
        if self.role == "decode":
            return self.admit_decode(req, now)
        st = req.attrs
        for k, v in (("computed", 0), ("generated", 0), ("blocks", 0), ("cached", 0),
                     ("parked", 0), ("itl", []), ("preempted", 0)):
            st.setdefault(k, v if not isinstance(v, list) else list(v))
        req.instance = self.name
        self.waiting.append(req)
        if not self.in_step:
            self._step(now)

    @property
    def busy_seconds(self):
        return self._busy

    @property
    def load(self):
        """LLMServingSim/vLLM least-load score: (4*waiting + running) / max_seqs."""
        return (4 * len(self.waiting) + len(self.running)) / self.max_seqs

    # ---------------------------------------------------------------- block pool
    def _blocks_for(self, tokens):
        return -(-tokens // self.block)

    def _evict(self, n):
        """Drop n cached blocks, LRU first; a request's blocks go tail-first so
        its recoverable prefix shrinks from the end."""
        while n > 0 and self._lru:
            ent = self._lru[0]
            take = min(n, ent[1])
            ent[1] -= take
            n -= take
            self._cached_total -= take
            if ent[0] is not None:
                ent[0].attrs["cached"] -= take
                if self.host is not None:              # parking: evicted blocks survive on the host
                    nbytes = take * self.block * self.spec.kv_bytes_per_token
                    if self.host.used + nbytes <= self.host.capacity:
                        self.host.used += nbytes
                        ent[0].attrs["parked"] += take
            if ent[1] == 0:
                self._lru.pop(0)

    def _alloc(self, req, upto_tokens):
        n = self._blocks_for(upto_tokens) - req.attrs["blocks"]
        if n <= 0:
            return True
        if n > self.free_blocks:
            return False
        uncached_free = self.free_blocks - self._cached_total
        if n > uncached_free:
            self._evict(n - uncached_free)
        self.free_blocks -= n
        req.attrs["blocks"] += n
        return True

    def _release(self, req, keep_owner):
        """Return a request's blocks to the pool as cached (recoverable) blocks."""
        n = req.attrs["blocks"]
        if n == 0:
            return
        self.free_blocks += n
        self._cached_total += n
        req.attrs["blocks"] = 0
        owner = req if keep_owner else None
        if owner is not None:
            owner.attrs["cached"] = n
        self._lru.append([owner, n])

    def _recoverable(self, req):
        """(cached_blocks, parked_blocks) usable on resume.  The prefix is
        contiguous: cached blocks are the head, parked ones follow (they were
        evicted tail-first, so what is parked is the tail of the cached run)."""
        st = req.attrs
        cap = (st["input"] + st["generated"] - 1) // self.block   # vLLM: one token always recomputed
        c = min(st["cached"], cap)
        pk = min(st["parked"], cap - c) if st["parked"] else 0
        return c, pk

    def _recover(self, req):
        """On re-admission, take back the cached prefix and recall the parked
        part from the host tier.  Returns recall bytes to charge to this step."""
        st = req.attrs
        c, pk = self._recoverable(req)
        for ent in self._lru:
            if ent[0] is req:
                self._lru.remove(ent)
                self._cached_total -= ent[1]
                break
        self.free_blocks -= c
        recall = 0
        if pk:
            if not self._alloc_raw(pk):
                pk = 0
            else:
                recall = pk * self.block * self.spec.kv_bytes_per_token
                self.host.recalled_bytes += recall
                self.recalled += pk * self.block
        if st["parked"]:
            self.host.used -= st["parked"] * self.block * self.spec.kv_bytes_per_token
        st["cached"] = st["parked"] = 0
        st["blocks"] = c + pk
        st["computed"] = (c + pk) * self.block
        return recall

    def _alloc_raw(self, n):
        if n > self.free_blocks:
            return False
        uncached_free = self.free_blocks - self._cached_total
        if n > uncached_free:
            self._evict(n - uncached_free)
        self.free_blocks -= n
        return True

    def _preempt(self, victim):
        self._release(victim, keep_owner=True)
        victim.attrs["computed"] = 0                   # vLLM: forget progress, re-derive from caches on resume
        victim.attrs["preempted"] += 1
        self.preemptions += 1
        self.waiting.insert(0, victim)

    # ---------------------------------------------------------------- scheduling
    def _step(self, now):
        budget = self.budget
        sched = []                                     # (req, num_new, computed_before)
        preempted = False
        recall_bytes = 0
        # phase A: running requests catch up to their reached length (1 token in decode)
        i = 0
        while i < len(self.running) and budget > 0:
            req = self.running[i]
            st = req.attrs
            reached = st["input"] + st["generated"]
            num_new = min(reached - st["computed"], budget)
            if num_new <= 0:
                i += 1
                continue
            ok = False
            while True:
                if self._alloc(req, st["computed"] + num_new):
                    ok = True
                    break
                victim = self.running.pop()
                self._preempt(victim)
                preempted = True
                if victim is req:
                    break
            if not ok:
                break
            sched.append((req, num_new, st["computed"]))
            budget -= num_new
            i += 1
        # phase B: admit from waiting, FCFS, never preempting
        if not preempted:
            while self.waiting and budget > 0 and len(self.running) < self.max_seqs:
                req = self.waiting[0]
                if req.arrival > now + 1e-12:
                    break
                st = req.attrs
                reached = st["input"] + st["generated"]
                # recoverable prefix (cached + parked blocks, capped at reached-1 tokens)
                rec = sum(self._recoverable(req)) if st["blocks"] == 0 else 0
                computed = rec * self.block if rec else st["computed"]
                num_new = min(reached - computed, budget)
                if num_new <= 0:
                    break
                # reserve full sequence (vLLM scheduler_reserve_full_isl): the length
                # reached so far -- the prompt on first admission, prompt+generated
                # on a resume -- must fit before admission; recovered blocks come
                # out of the same free pool
                need = max(self._blocks_for(reached), self._blocks_for(computed + num_new)) - rec
                if need > self.free_blocks - rec:
                    break
                if rec:
                    recall_bytes += self._recover(req)
                if not self._alloc(req, st["computed"] + num_new):
                    break                              # cannot happen after the check above
                self.waiting.pop(0)
                if req.start is None:
                    req.start = now
                if st["preempted"]:
                    self.recomputed += reached - st["computed"]     # oracle definition: reached - recovered
                self.running.append(req)
                sched.append((req, num_new, st["computed"]))
                budget -= num_new
        if not sched:
            self.in_step = False
            return
        # step time from the batch composition
        B = K = P = 0
        for req, n, before in sched:
            st = req.attrs
            reached = st["input"] + st["generated"]
            if n == 1 and before == reached - 1 and st["generated"] > 0:
                B += 1                                 # steady-state decode token
            else:
                P += n                                 # prefill / recompute chunk
            K += before
        t_ms = self.profile.step_ms(self.spec, B, K, P) / self.slowdown
        dt = t_ms / 1e3
        if recall_bytes and self.host is not None:     # parked KV read back over PCIe
            dt += recall_bytes / self.host.bw
        if self.role == "prefill" and self.link is not None and P:
            # ship this step's chunk KV to the decode side, one message per layer
            end = self.link.transfer(P * self.spec.kv_bytes_per_token, now + dt,
                                     messages=self.spec.n_layer)
            dt = end - now
        self.in_step = True
        self.steps += 1
        self._busy += dt
        self.sim.schedule(now + dt, self._end, sched)

    def _end(self, now, sched):
        for req, n, _ in sched:
            st = req.attrs
            st["computed"] += n
            if st["computed"] == st["input"] + st["generated"]:      # this step emitted a token
                if st["generated"] == 0:
                    st["ttft"] = now - req.arrival
                else:
                    st["itl"].append(now - st["last_tok"])
                st["generated"] += 1
                st["last_tok"] = now
                if st["generated"] >= st["output"]:
                    self._finish(req, now)
                elif self.role == "prefill" and st["generated"] == 1:
                    self._handoff(req, now)
        self._step(now)

    def _handoff(self, req, now):
        """Prefill done: release blocks here, pass to the decode side."""
        self.running.remove(req)
        self._release(req, keep_owner=False)
        self.handoff(req, now)

    def _finish(self, req, now):
        self.running.remove(req)
        self._release(req, keep_owner=False)
        st = req.attrs
        st["tpot"] = (now - req.arrival - st["ttft"]) / max(st["output"] - 1, 1)
        self.sim.complete(req, now)


def least_load(instances):
    """Selector for Group: LLMServingSim's rotating least-load pick."""
    state = {"start": 0}

    def pick(req, cands):
        n = len(cands)
        best, best_s = None, float("inf")
        for off in range(n):
            ins = cands[(state["start"] + off) % n]
            s = ins.load
            if s < best_s:
                best, best_s = ins, s
        state["start"] = (cands.index(best) + 1) % n
        return best
    return pick
