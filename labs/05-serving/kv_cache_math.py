"""
KV-cache memory math -- the Deeper Dive artifact for Post 5.

Why a KV cache exists
---------------------
A decoder-only transformer generates one token at a time. To produce token t it
needs attention over tokens 0..t-1, and attention needs every earlier token's
Key and Value vectors in every layer. Recomputing them each step would make
generation quadratic, so the server keeps them: the KV cache.

The formula
-----------
For every token, every layer stores one K vector and one V vector per KV head:

    bytes per token = 2            (one K, one V)
                    x n_layers
                    x n_kv_heads   (== n_heads for classic multi-head attention)
                    x head_dim
                    x bytes_per_element   (2 for fp16/bf16, 1 for int8/fp8)

    total KV bytes  = bytes per token x seq_len x batch

Notice what is NOT in it: vocabulary size, MLP width, parameter count. The KV
cache depends only on the attention shape, the context length and how many
sequences you serve at once. That is why a 7B model can need more memory for
its cache than for its weights.

Grouped-query attention (GQA)
-----------------------------
GQA lets several query heads share one K/V head. Query heads stay at n_heads,
but only n_kv_heads K/V pairs are stored, so the cache shrinks by
n_heads / n_kv_heads. Llama 2 70B and Llama 3 8B both use 8 KV heads.

Run it:   python kv_cache_math.py
"""

from __future__ import annotations

from dataclasses import dataclass

KIB = 1024
MIB = 1024 ** 2
GIB = 1024 ** 3

# Bytes per stored element for the dtypes the lab compares.
DTYPE_BYTES = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1, "fp8": 1}


@dataclass(frozen=True)
class ModelSpec:
    """Just the numbers the KV cache (and weight memory) depend on."""

    name: str
    n_layers: int
    n_heads: int          # query heads
    n_kv_heads: int       # stored K/V heads (== n_heads without GQA)
    head_dim: int
    n_params: float       # used only for weight memory, never for the KV cache
    max_ctx: int

    @property
    def gqa_ratio(self) -> int:
        return self.n_heads // self.n_kv_heads


# --------------------------------------------------------------------------- #
# Public architectures. Shapes are from the published model configs.
# --------------------------------------------------------------------------- #
# Llama 2 7B: hidden 4096 = 32 heads x 128. Classic multi-head attention.
LLAMA2_7B = ModelSpec("Llama-2-7B", n_layers=32, n_heads=32, n_kv_heads=32,
                      head_dim=128, n_params=6.74e9, max_ctx=4096)

# Llama 3 8B (original release): same 32x32x128 shape, but GQA with 8 KV heads.
LLAMA3_8B = ModelSpec("Llama-3-8B", n_layers=32, n_heads=32, n_kv_heads=8,
                      head_dim=128, n_params=8.03e9, max_ctx=8192)

# Llama 2 70B: hidden 8192 = 64 heads x 128, 80 layers, GQA with 8 KV heads.
LLAMA2_70B = ModelSpec("Llama-2-70B", n_layers=80, n_heads=64, n_kv_heads=8,
                       head_dim=128, n_params=69e9, max_ctx=4096)

PUBLIC_MODELS = (LLAMA2_7B, LLAMA3_8B, LLAMA2_70B)


# --------------------------------------------------------------------------- #
# The math
# --------------------------------------------------------------------------- #
def kv_bytes_per_token(spec: ModelSpec, dtype: str = "fp16") -> int:
    """2 (K and V) x layers x kv_heads x head_dim x bytes_per_element."""
    return 2 * spec.n_layers * spec.n_kv_heads * spec.head_dim * DTYPE_BYTES[dtype]


def kv_cache_bytes(spec: ModelSpec, seq_len: int, batch: int = 1,
                   dtype: str = "fp16") -> int:
    """Total KV-cache bytes for `batch` sequences of `seq_len` tokens each."""
    return kv_bytes_per_token(spec, dtype) * seq_len * batch


def weight_bytes(spec: ModelSpec, bytes_per_param: float = 2.0) -> float:
    """Weight memory. 2 bytes/param for fp16, 1 for int8, 0.5 for int4.

    (Real int4 formats also store per-group scales, so they come out slightly
    larger than 0.5 bytes/param. The lab ignores that overhead.)
    """
    return spec.n_params * bytes_per_param


def max_sequences(spec: ModelSpec, kv_budget_bytes: float, seq_len: int,
                  dtype: str = "fp16") -> int:
    """How many full-length sequences fit in a given KV budget."""
    return int(kv_budget_bytes // kv_cache_bytes(spec, seq_len, 1, dtype))


def spec_from_lab02(cfg) -> ModelSpec:
    """Turn a Lab 02 TrainConfig into a ModelSpec (no GQA in Lab 02's GPT)."""
    return ModelSpec(f"Lab 02 {cfg.name}", n_layers=cfg.n_layer, n_heads=cfg.n_head,
                     n_kv_heads=cfg.n_head, head_dim=cfg.n_embd // cfg.n_head,
                     n_params=0.0, max_ctx=cfg.block_size)


# --------------------------------------------------------------------------- #
# The hand-checked example the tests pin
# --------------------------------------------------------------------------- #
# Llama-2-7B, fp16:
#     2 x 32 layers x 32 kv_heads x 128 head_dim x 2 bytes
#   = 2 x 32 x 32 x 128 x 2
#   = 524,288 bytes                  (32 x 32 = 1,024; x 128 = 131,072;
#                                     x 2 = 262,144; x 2 = 524,288)
#   = 0.5 MiB per token              (524,288 / 1,048,576 = 0.5)
#
# Full 4,096-token context, one sequence:
#     0.5 MiB x 4,096 = 2,048 MiB = 2 GiB
#
# Same model with int8 KV: half of that, 256 KiB per token.
# Llama-3-8B (8 KV heads instead of 32): a quarter, 128 KiB per token.
HAND_CHECKED_LLAMA2_7B_FP16_BYTES_PER_TOKEN = 524_288
HAND_CHECKED_LLAMA2_7B_FULL_CTX_BYTES = 2 * GIB


def human(nbytes: float) -> str:
    """Binary units: KiB / MiB / GiB."""
    for unit, size in (("GiB", GIB), ("MiB", MIB), ("KiB", KIB)):
        if nbytes >= size:
            return f"{nbytes / size:.2f} {unit}"
    return f"{nbytes:.0f} B"


def kv_table(include_lab02: bool = True) -> str:
    """The KV-cache table run_demo.py prints."""
    rows = []
    specs = list(PUBLIC_MODELS)
    if include_lab02:
        from labs_path import lab02_configs

        configs = lab02_configs()
        if configs is not None:
            specs.append(spec_from_lab02(configs["real"]))

    header = (f"  {'model':<15}| {'layers':>6} | {'q/kv heads':>10} | {'dtype':>5} | "
              f"{'per token':>10} | {'one full-ctx seq':>22}")
    rows.append(header)
    rows.append("  " + "-" * (len(header) - 2))
    for spec in specs:
        for dtype in ("fp16", "int8"):
            per_tok = kv_bytes_per_token(spec, dtype)
            full = kv_cache_bytes(spec, spec.max_ctx, 1, dtype)
            rows.append(
                f"  {spec.name:<15}| {spec.n_layers:>6} | "
                f"{spec.n_heads:>4} / {spec.n_kv_heads:<3} | {dtype:>5} | "
                f"{human(per_tok):>10} | {human(full):>10} @ {spec.max_ctx:>5} tok"
            )
    return "\n".join(rows)


def budget_example(hbm_gib: float = 80.0) -> str:
    """How many concurrent 4K-token conversations fit next to the weights?"""
    lines = [f"  Illustrative {hbm_gib:.0f} GiB accelerator, full 4,096-token sequences, fp16 weights:"]
    for spec in PUBLIC_MODELS:
        w = weight_bytes(spec, 2.0)
        budget = hbm_gib * GIB - w
        if budget <= 0:
            lines.append(f"    {spec.name:<12} weights {human(w):>10}  -> do not fit on one device")
            continue
        n16 = max_sequences(spec, budget, 4096, "fp16")
        n8 = max_sequences(spec, budget, 4096, "int8")
        lines.append(f"    {spec.name:<12} weights {human(w):>10}  KV budget {human(budget):>10}"
                     f"  -> {n16:>3} seqs fp16 KV, {n8:>3} seqs int8 KV")
    return "\n".join(lines)


if __name__ == "__main__":
    print(kv_table())
    print()
    print(budget_example())
    print()
    no_gqa = ModelSpec("Llama-2-70B without GQA", 80, 64, 64, 128, 69e9, 4096)
    print(f"  GQA saving, Llama-2-70B: {human(kv_bytes_per_token(LLAMA2_70B))}/token with 8 KV heads "
          f"vs {human(kv_bytes_per_token(no_gqa))}/token if all 64 heads stored K/V "
          f"({LLAMA2_70B.gqa_ratio}x smaller)")
