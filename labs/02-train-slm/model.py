"""
Lab 02 — a GPT, written out in full.

This is the same architecture as GPT-2, just small. Reading order:

    GPT.forward
      └─ token embedding + position embedding
      └─ Block  (xN)
           ├─ LayerNorm -> CausalSelfAttention -> add to residual stream
           └─ LayerNorm -> MLP                 -> add to residual stream
      └─ final LayerNorm
      └─ output head (weights TIED to the token embedding)

Two details worth pausing on, because they are the ones people get wrong:

  * **The residual stream.** Every block *adds to* `x`, never replaces it. That
    unbroken path from the embedding to the loss is why gradients survive depth.
  * **Weight tying.** The output head reuses the token-embedding matrix. Same
    parameters, used to turn tokens into vectors on the way in and vectors back
    into token scores on the way out. Fewer parameters, better small models.

One sequence at a time: `forward` takes a list of token ids of length T and
returns a (T, vocab_size) tensor of logits. Batching happens in the trainer by
summing gradients across sequences, which is what a batch *is*.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from autograd import (
    Tensor,
    add,
    causal_softmax,
    concat_cols,
    cross_entropy,
    embed,
    gelu,
    layernorm,
    matmul,
    scale,
    slice_cols,
    transpose,
)


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int      # context window: how far back the model can see
    n_layer: int
    n_head: int
    n_embd: int

    def __post_init__(self) -> None:
        assert self.n_embd % self.n_head == 0, "n_embd must divide evenly among heads"

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head


def _affine(x: Tensor, gain: Tensor, bias: Tensor) -> Tensor:
    """LayerNorm's learnable per-feature scale and shift."""
    out = Tensor(
        [[x.data[i][j] * gain.data[0][j] + bias.data[0][j] for j in range(x.cols)]
         for i in range(x.rows)],
        (x, gain, bias),
    )

    def _backward() -> None:
        for i in range(x.rows):
            g_row, x_row = out.grad[i], x.data[i]
            for j in range(x.cols):
                g = g_row[j]
                x.grad[i][j] += g * gain.data[0][j]
                gain.grad[0][j] += g * x_row[j]
                bias.grad[0][j] += g

    out._backward = _backward
    return out


class GPT:
    def __init__(self, config: GPTConfig, seed: int = 1337):
        self.config = config
        rng = random.Random(seed)
        c = config

        # Token and position embeddings.
        self.wte = Tensor.randn(c.vocab_size, c.n_embd, rng)
        self.wpe = Tensor.randn(c.block_size, c.n_embd, rng)

        self.blocks = []
        for _ in range(c.n_layer):
            # The residual projections get a 1/sqrt(2*n_layer) init so that the
            # residual stream's variance does not grow with depth (GPT-2 trick).
            resid_std = 0.02 / math.sqrt(2 * c.n_layer)
            self.blocks.append({
                "ln1_g": Tensor([[1.0] * c.n_embd]),
                "ln1_b": Tensor.zeros(1, c.n_embd),
                "wq": Tensor.randn(c.n_embd, c.n_embd, rng),
                "bq": Tensor.zeros(1, c.n_embd),
                "wk": Tensor.randn(c.n_embd, c.n_embd, rng),
                "bk": Tensor.zeros(1, c.n_embd),
                "wv": Tensor.randn(c.n_embd, c.n_embd, rng),
                "bv": Tensor.zeros(1, c.n_embd),
                "wo": Tensor.randn(c.n_embd, c.n_embd, rng, std=resid_std),
                "bo": Tensor.zeros(1, c.n_embd),
                "ln2_g": Tensor([[1.0] * c.n_embd]),
                "ln2_b": Tensor.zeros(1, c.n_embd),
                "fc_w": Tensor.randn(c.n_embd, 4 * c.n_embd, rng),
                "fc_b": Tensor.zeros(1, 4 * c.n_embd),
                "proj_w": Tensor.randn(4 * c.n_embd, c.n_embd, rng, std=resid_std),
                "proj_b": Tensor.zeros(1, c.n_embd),
            })

        self.lnf_g = Tensor([[1.0] * c.n_embd])
        self.lnf_b = Tensor.zeros(1, c.n_embd)
        # No separate lm_head: the output head is `wte` transposed (weight tying).

    # -- parameter plumbing --------------------------------------------------
    def parameters(self) -> list[Tensor]:
        params = [self.wte, self.wpe]
        for b in self.blocks:
            params.extend(b.values())
        params.extend([self.lnf_g, self.lnf_b])
        return params

    def num_params(self) -> int:
        return sum(p.rows * p.cols for p in self.parameters())

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    # -- the forward pass ----------------------------------------------------
    def _attention(self, x: Tensor, b: dict) -> Tensor:
        """Multi-head causal self-attention over one sequence."""
        c = self.config
        q = add(matmul(x, b["wq"]), b["bq"])
        k = add(matmul(x, b["wk"]), b["bk"])
        v = add(matmul(x, b["wv"]), b["bv"])

        head_outputs = []
        for h in range(c.n_head):
            lo, hi = h * c.head_dim, (h + 1) * c.head_dim
            qh, kh, vh = slice_cols(q, lo, hi), slice_cols(k, lo, hi), slice_cols(v, lo, hi)
            # scores[t, s] = how much position t attends to position s
            scores = scale(matmul(qh, transpose(kh)), 1.0 / math.sqrt(c.head_dim))
            weights = causal_softmax(scores)   # the mask lives inside this op
            head_outputs.append(matmul(weights, vh))

        merged = concat_cols(head_outputs) if c.n_head > 1 else head_outputs[0]
        return add(matmul(merged, b["wo"]), b["bo"])

    def _mlp(self, x: Tensor, b: dict) -> Tensor:
        """Position-wise feed-forward: widen 4x, GELU, project back."""
        hidden = gelu(add(matmul(x, b["fc_w"]), b["fc_b"]))
        return add(matmul(hidden, b["proj_w"]), b["proj_b"])

    def forward(self, ids: list[int]) -> Tensor:
        c = self.config
        assert 0 < len(ids) <= c.block_size, f"sequence must be 1..{c.block_size} tokens"

        # Residual stream starts as "what token is this" + "where am I".
        x = add(embed(self.wte, ids), embed(self.wpe, list(range(len(ids)))))

        for b in self.blocks:
            # Pre-norm: normalize the *input* of each sublayer, add its output.
            x = add(x, self._attention(_affine(layernorm(x), b["ln1_g"], b["ln1_b"]), b))
            x = add(x, self._mlp(_affine(layernorm(x), b["ln2_g"], b["ln2_b"]), b))

        x = _affine(layernorm(x), self.lnf_g, self.lnf_b)
        return matmul(x, transpose(self.wte))   # tied output head

    def loss(self, ids: list[int], targets: list[int]) -> Tensor:
        return cross_entropy(self.forward(ids), targets)

    # -- sampling ------------------------------------------------------------
    def generate(self, prompt_ids: list[int], max_new_tokens: int, rng: random.Random,
                 temperature: float = 1.0, top_k: int | None = None) -> list[int]:
        """Autoregressive sampling. temperature=0 means greedy (argmax).

        Note the context crop: once the sequence outgrows `block_size` we only
        feed the model the most recent `block_size` tokens. That window IS the
        model's entire memory.
        """
        ids = list(prompt_ids)
        for _ in range(max_new_tokens):
            context = ids[-self.config.block_size:]
            logits = self.forward(context).data[-1]

            if temperature <= 0.0:
                ids.append(max(range(len(logits)), key=logits.__getitem__))
                continue

            scaled = [v / temperature for v in logits]
            if top_k is not None and top_k < len(scaled):
                cutoff = sorted(scaled, reverse=True)[top_k - 1]
                scaled = [v if v >= cutoff else -float("inf") for v in scaled]

            m = max(scaled)
            exps = [math.exp(v - m) if v != -float("inf") else 0.0 for v in scaled]
            total = sum(exps)
            r = rng.random() * total
            acc = 0.0
            for token_id, e in enumerate(exps):
                acc += e
                if acc >= r:
                    ids.append(token_id)
                    break
            else:  # pragma: no cover - float rounding at the very end
                ids.append(len(exps) - 1)
        return ids
