"""LLM step-time model in LLM-Pilot form, calibrated per card.

One engine iteration (vLLM-style continuous batching with chunked prefill)
carries B decode sequences with a total context of K tokens, plus P prefill
tokens.  Its wall time on card k is

    B > 0 (decode or mixed, CUDA-graph step)
    T_step = beta0 + beta_w * W / BW_k                 weights read once per step
           + beta_kv * K * kv_bytes / BW_k             decode attention reads KV
           + beta_mix * beta_c * 2 * params * P / C_k  prefill chunk riding along
           + beta_seq * B                              per-sequence overhead
    B == 0 (pure prefill, eager step)
    T_step = beta0 + beta_pre + beta_c * 2 * params * P / C_k

Decode tokens' GEMM work is not a separate term: with B << C/(2*params*BW)
it hides under the weight read (memory-bound), which is why the decode side
of the oracle is linear in B with a slope of tens of microseconds.  In a pure
prefill step the weight read hides under the compute instead.

The physical terms (W, kv_bytes, params, BW_k, C_k) come from the model card
and the GPU spec; the betas absorb what the ideal numbers miss (LLM-Pilot
§IV-C).  They are fitted once per (model, card) against an oracle -- here
LLMServingSim 2.0's vLLM-validated RTX 4090 profile: 79,826 decode-only steps
(MAPE 2.3%) and 1,897 prefill steps -- and later against desktop measurements
for the 3090.
"""
from dataclasses import dataclass

GIB = 1024 ** 3


@dataclass(frozen=True)
class LLMSpec:
    name: str
    params: float           # parameters
    n_layer: int
    kv_heads: int
    head_dim: int
    bytes_per_elem: int = 2

    @property
    def weight_bytes(self):
        return self.params * self.bytes_per_elem

    @property
    def kv_bytes_per_token(self):
        return 2 * self.kv_heads * self.head_dim * self.n_layer * self.bytes_per_elem


@dataclass(frozen=True)
class LLMCardProfile:
    """Card numbers plus the fitted betas for one (model, card) pair.

    Two step regimes, as vLLM runs them:
      decode / mixed (B > 0)   CUDA-graph step: weights read once (beta_w), KV
                               read for the decode context (beta_kv), per-seq
                               overhead (beta_seq), prefill tokens at beta_c*beta_mix
      pure prefill (B == 0)    eager step: compute-bound, the weight read hides
                               under the GEMM; fixed eager overhead beta_pre_ms
                               plus prefill tokens at beta_c
    """
    card: str
    tflops: float           # dense bf16, TFLOP/s
    bw_gbps: float          # HBM GB/s
    mem_util: float         # share of card memory the engine may use
    beta0_ms: float = 0.0
    beta_w: float = 1.0
    beta_kv: float = 1.0
    beta_c: float = 1.0
    beta_seq_ms: float = 0.0
    beta_pre_ms: float = 0.0
    beta_mix: float = 1.0

    def step_ms(self, spec: LLMSpec, B: int, K: int, P: int) -> float:
        bw = self.bw_gbps * 1e9
        flops = self.tflops * 1e12
        t = self.beta0_ms
        t += self.beta_kv * K * spec.kv_bytes_per_token / bw * 1e3
        t += self.beta_seq_ms * B
        per_tok = self.beta_c * 2 * spec.params / flops * 1e3
        if B > 0:
            t += self.beta_w * spec.weight_bytes / bw * 1e3
            t += self.beta_mix * per_tok * P
        else:
            t += self.beta_pre_ms + per_tok * P
        return t

    def kv_capacity_tokens(self, spec: LLMSpec, mem_gib: float, block: int = 16) -> int:
        kv_bytes = int(mem_gib * GIB * self.mem_util) - spec.weight_bytes
        blocks = int(kv_bytes // (spec.kv_bytes_per_token * block))
        return blocks * block


LLAMA31_8B = LLMSpec("meta-llama/Llama-3.1-8B", params=8.03e9, n_layer=32, kv_heads=8, head_dim=128)

# RTX 4090, fitted against LLMServingSim 2.0 (see module docstring).
#   decode-only steps (79,826):  16.43 ms + 34.8 us * B + 0.211 us * K        MAPE 2.3%
#     16.43 = 15.93 ms weights (16.06 GB / 1008 GB/s) + 0.50 ms
#     0.211 us / KV token = 1.62 x ideal (0.130 us)
#   mixed steps (1,897):         + 85.7 us * P                                 (extra over decode baseline)
#   pure prefill steps (1,063):  12.30 ms + 76.9 us * P                        MAPE 10.2%
#     76.9 us / token = 0.79 x ideal (97.3 us at 165 TFLOP/s); 85.7 / 76.9 = 1.114
RTX4090 = LLMCardProfile("RTX4090", tflops=165.0, bw_gbps=1008.0, mem_util=0.833919,
                         beta0_ms=0.50, beta_w=1.0, beta_kv=1.62, beta_c=0.790, beta_seq_ms=0.0348,
                         beta_pre_ms=11.80, beta_mix=1.114)
