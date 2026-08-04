"""
Lab 02 — Deeper Dive: count the FLOPs of your training run, explicitly.

    python flops.py

Run this and you can say exactly what your run cost, in the same unit the
frontier labs report. Everything below comes from one observation:

    A matrix multiply that uses a weight once costs 2 FLOPs per weight
    (one multiply, one add).

From there:

    forward       2N   FLOPs per token   (every weight used once)
    backward      4N   FLOPs per token   (gradient w.r.t. inputs AND weights)
    ------------------------------------
    total         6N   FLOPs per token

That "6N" is the number behind every compute estimate you have ever read,
including the Chinchilla scaling laws.

The 6N approximation ignores attention itself -- the QKᵀ and the weighting of V
are matrix multiplies whose size depends on the *context length*, not on the
parameter count. For each layer that is 2 · (2 · block_size · n_embd) FLOPs per
token forward, tripled for the backward pass:

    attention term = 12 · n_layer · n_embd · block_size   FLOPs per token

At GPT-3 scale this term is a rounding error, which is why people drop it. At
laptop scale with a short context it is small too -- but it is the term that
explodes when you extend the context window, so it is worth carrying.
"""

from __future__ import annotations

# Reference points, so the number you compute means something.
GPT3_TRAINING_FLOPS = 3.14e23      # 175B params, 300B tokens
CHINCHILLA_TOKENS_PER_PARAM = 20   # Hoffmann et al. 2022: the compute-optimal ratio


def flops_per_token(n_params: int, n_layer: int, n_embd: int, block_size: int,
                    backward: bool = True) -> float:
    """FLOPs to process one token, weights + attention."""
    multiplier = 6 if backward else 2
    weight_term = multiplier * n_params
    attention_term = (multiplier // 2) * 2 * 2 * n_layer * n_embd * block_size
    return weight_term + attention_term


def training_flops(n_params: int, n_layer: int, n_embd: int, block_size: int,
                   batch_size: int, steps: int) -> float:
    """Total FLOPs for a whole run.

    Note that every position in a sequence is a training signal, so a step
    processes block_size × batch_size tokens -- not batch_size tokens.
    """
    tokens = block_size * batch_size * steps
    return flops_per_token(n_params, n_layer, n_embd, block_size) * tokens


def chinchilla_optimal_tokens(n_params: int) -> int:
    """How many tokens this many parameters 'wants' to see (~20 per parameter)."""
    return n_params * CHINCHILLA_TOKENS_PER_PARAM


def tokens_per_param(n_params: int, tokens_seen: int) -> float:
    return tokens_seen / n_params


def human(value: float) -> str:
    for threshold, suffix in ((1e18, "EFLOPs"), (1e15, "PFLOPs"),
                              (1e12, "TFLOPs"), (1e9, "GFLOPs"), (1e6, "MFLOPs")):
        if value >= threshold:
            return f"{value / threshold:.2f} {suffix}"
    return f"{value:.0f} FLOPs"


def report(n_params: int, n_layer: int, n_embd: int, block_size: int,
           batch_size: int, steps: int) -> str:
    """A block of text the post can screenshot."""
    tokens = block_size * batch_size * steps
    per_token = flops_per_token(n_params, n_layer, n_embd, block_size)
    total = training_flops(n_params, n_layer, n_embd, block_size, batch_size, steps)
    optimal = chinchilla_optimal_tokens(n_params)
    ratio = tokens_per_param(n_params, tokens)

    weight_term = 6 * n_params
    attn_term = 12 * n_layer * n_embd * block_size

    lines = [
        "  parameters (N) ................ {:,}".format(n_params),
        "  tokens processed (D) .......... {:,}  ({} steps x {} batch x {} ctx)".format(
            tokens, steps, batch_size, block_size),
        "",
        "  FLOPs per token ............... {:,.0f}".format(per_token),
        "    weights   6N ................ {:,} ({:.0f}%)".format(
            weight_term, 100 * weight_term / per_token),
        "    attention 12·L·d·ctx ........ {:,} ({:.0f}%)".format(
            attn_term, 100 * attn_term / per_token),
        "",
        "  TOTAL TRAINING COMPUTE ........ {}".format(human(total)),
        "    as a fraction of GPT-3 ...... {:.2e}x".format(total / GPT3_TRAINING_FLOPS),
        "",
        "  Chinchilla check:",
        "    tokens seen per parameter ... {:.1f}".format(ratio),
        "    compute-optimal would be .... {} tokens/param ({:,} tokens)".format(
            CHINCHILLA_TOKENS_PER_PARAM, optimal),
        "    verdict ..................... {}".format(
            "under-trained for its size" if ratio < CHINCHILLA_TOKENS_PER_PARAM
            else "past the compute-optimal point (fine — we want a usable model,"
                 " not a compute-optimal one)"),
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    import data
    from model import GPT, GPTConfig
    from train import TINY

    ds = data.build_dataset()
    model = GPT(GPTConfig(vocab_size=ds.vocab_size, block_size=TINY.block_size,
                          n_layer=TINY.n_layer, n_head=TINY.n_head, n_embd=TINY.n_embd))

    print("=" * 66)
    print("  LAB 02 — DEEPER DIVE: what did this training run actually cost?")
    print("=" * 66)
    print(report(model.num_params(), TINY.n_layer, TINY.n_embd,
                 TINY.block_size, TINY.batch_size, TINY.max_steps))
    print("=" * 66)
