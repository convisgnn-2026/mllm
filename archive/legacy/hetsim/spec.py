"""Hardware and request descriptions.  Plain data, no behaviour."""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CardType:
    """One kind of accelerator, or one MIG slice of it.

    sm        streaming multiprocessors (the DiT model scales with this)
    mem_gib   device memory available to a worker
    tflops    dense bf16 compute (LLM prefill scales with this)   [S2+]
    bw_gbps   HBM bandwidth                                        [S2+]
    """
    name: str
    sm: int
    mem_gib: float
    tflops: float = 0.0
    bw_gbps: float = 0.0


@dataclass(frozen=True)
class Card:
    """A concrete device the cluster owns."""
    id: str
    type: CardType


@dataclass
class Request:
    """One request of one tenant.  `cls` is the request class (a DiT
    template, an LLM (S_in, S_out) bucket, ...), `attrs` its parameters."""
    id: int
    tenant: str
    cls: str
    arrival: float
    attrs: dict = field(default_factory=dict)
    # filled by the simulator
    start: float | None = None
    end: float | None = None
    instance: str | None = None
    service: float | None = None

    @property
    def latency(self):
        return None if self.end is None else self.end - self.arrival
