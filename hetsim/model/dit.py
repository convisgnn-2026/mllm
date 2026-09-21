"""Video-DiT service-time model, fitted on B300 (7,130 requests, MAPE 2.0%).

    t(tpl, sm) = t_full(tpl) * ( FULL_SM * (1 - b) / sm + b )

A share b of the work does not scale with SM count (launches, memory-bound VAE
decode, host serialisation); the rest scales inversely with SM.  The form goes
through t_full exactly at FULL_SM, so a new card needs one full-card
measurement per template -- that is how the card axis (S7) enters: a new
`DiTModel` with its own t_full table.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DiTModel:
    full_sm: int
    t_full: dict            # template -> seconds on a full card (20 steps + VAE)
    b_tpl: dict             # template -> non-scaling share
    b_default: float
    act_peak: dict          # template -> peak activation GiB
    resident_gib: float     # weights resident in the worker

    def service(self, tpl, sm):
        b = self.b_tpl.get(tpl, self.b_default)
        return self.t_full[tpl] * (self.full_sm * (1 - b) / sm + b)

    def feasible(self, tpl, mem_gib):
        return self.act_peak[tpl] + self.resident_gib <= mem_gib

    def work(self, tpl):
        """Reference work of one request: full-card seconds."""
        return self.t_full[tpl]


# --- B300, Wan2.1-1.3B, 20 steps, T5 disaggregated (solver/model.py) -------
W_DIT, W_VAE, W_T5 = 2.65, 0.47, 10.58

B300 = DiTModel(
    full_sm=148,
    t_full={"480x272x33": 0.85, "832x480x33": 2.62,
            "832x480x49": 4.06, "1280x720x49": 12.22},
    b_tpl={"480x272x33": 0.232, "832x480x33": 0.154,
           "832x480x49": 0.154, "1280x720x49": 0.156},
    b_default=0.184,
    act_peak={"480x272x33": 3.30, "832x480x33": 9.10, "832x480x49": 9.20,
              "1280x720x49": 20.40, "1920x1088x81": 46.40},
    resident_gib=W_DIT + W_VAE,
)


def scaled(base: DiTModel, factor: float) -> DiTModel:
    """Same model on a card `factor`x slower per SM (placeholder until measured)."""
    return DiTModel(base.full_sm, {k: v * factor for k, v in base.t_full.items()},
                    base.b_tpl, base.b_default, base.act_peak, base.resident_gib)


# B300 MIG slices: name -> (SM, GiB).  Used as CardTypes for MIG experiments.
MIG_PROFILES = {
    "7g.269gb": (148, 268.0), "4g.135gb": (72, 133.5), "3g.135gb": (70, 133.5),
    "2g.67gb": (36, 66.5), "1g.67gb": (30, 66.5), "1g.34gb": (18, 30.5),
}
