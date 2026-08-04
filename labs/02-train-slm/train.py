"""
Lab 02 — the training loop.

Everything that turns a randomly-initialized transformer into a language model:

    batching -> forward -> loss -> backward -> clip -> AdamW step -> repeat

with a learning-rate schedule on top and checkpoints falling out along the way.

The four pieces people underestimate, in the order they bite you:

  1. **The LR schedule.** Warm up linearly, then cosine-decay to a floor. Skip
     the warmup and the first few steps -- when Adam's variance estimates are
     still garbage -- can wreck the model before it learns anything.
  2. **Gradient clipping.** One bad batch produces a huge gradient. Clipping the
     global norm shortens that step without changing its direction.
  3. **Decoupled weight decay.** The "W" in AdamW. Decay is applied to the
     weights directly, not folded into the gradient, so it does not get rescaled
     by Adam's per-parameter learning rates.
  4. **Checkpoint hashing.** Not a training concern -- a *provenance* concern.
     A checkpoint you cannot hash is a checkpoint you cannot prove is the one
     you trained. See SECURITY.md.

Same code path for both configs: TINY (CPU, minutes) and REAL (a genuinely-sized
small model). Only the numbers change.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass, field

from data import Dataset, get_batch
from model import GPT, GPTConfig


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    name: str = "tiny"

    # Architecture
    n_layer: int = 3
    n_head: int = 3
    n_embd: int = 48
    block_size: int = 32

    # Optimization
    batch_size: int = 8
    max_steps: int = 300
    learning_rate: float = 3e-3
    min_lr: float = 3e-4
    warmup_steps: int = 30
    grad_clip: float = 1.0
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95

    # Bookkeeping
    eval_interval: int = 50
    eval_batches: int = 8
    sample_every: int = 100
    sample_tokens: int = 40
    temperature: float = 0.8
    top_k: int = 20
    seed: int = 1337
    out_dir: str = "out"


# The mandatory path: no GPU, no API key, finishes on a laptop.
TINY = TrainConfig(name="tiny")

# The same code, at a size where the samples stop being cute. This is the
# "recommended GPU" config in the sense that you want real hardware -- on a CPU
# with numpy it runs for hours, which is itself a useful thing to feel.
REAL = TrainConfig(
    name="real",
    n_layer=6, n_head=8, n_embd=256, block_size=128,
    batch_size=32, max_steps=3000,
    learning_rate=1e-3, min_lr=1e-4, warmup_steps=200,
    eval_interval=250, sample_every=500, sample_tokens=120,
)

CONFIGS = {"tiny": TINY, "real": REAL}


# --------------------------------------------------------------------------- #
# Learning-rate schedule: linear warmup -> cosine decay -> floor
# --------------------------------------------------------------------------- #
def lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.learning_rate * step / max(1, cfg.warmup_steps)
    if step > cfg.max_steps:
        return cfg.min_lr
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))   # 1 -> 0
    return cfg.min_lr + cosine * (cfg.learning_rate - cfg.min_lr)


# --------------------------------------------------------------------------- #
# Gradient clipping
# --------------------------------------------------------------------------- #
def clip_grad_norm(params, max_norm: float) -> float:
    """Rescale all gradients so their combined L2 norm is at most `max_norm`.

    Returns the norm BEFORE clipping -- watch it in the logs, because a sudden
    spike is the earliest warning that a run is about to diverge.
    """
    total = 0.0
    for p in params:
        for row in p.grad:
            for v in row:
                total += v * v
    norm = math.sqrt(total)

    if norm > max_norm and norm > 0.0:
        factor = max_norm / norm
        for p in params:
            for row in p.grad:
                for i in range(len(row)):
                    row[i] *= factor
    return norm


# --------------------------------------------------------------------------- #
# AdamW
# --------------------------------------------------------------------------- #
class AdamW:
    """Adam with *decoupled* weight decay.

    Adam keeps two running averages per weight: the mean of the gradient (m)
    and the mean of its square (v). The step is m / sqrt(v) -- so a weight with
    consistently small-but-steady gradients moves as fast as one with large
    noisy gradients. That per-parameter scaling is why it beats plain SGD here.
    """

    def __init__(self, params, beta1: float = 0.9, beta2: float = 0.95,
                 eps: float = 1e-8, weight_decay: float = 0.01):
        self.params = list(params)
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = [[[0.0] * p.cols for _ in range(p.rows)] for p in self.params]
        self.v = [[[0.0] * p.cols for _ in range(p.rows)] for p in self.params]

    def step(self, lr: float) -> None:
        self.t += 1
        b1, b2, eps = self.beta1, self.beta2, self.eps
        bias1 = 1.0 - b1 ** self.t
        bias2 = 1.0 - b2 ** self.t

        for idx, p in enumerate(self.params):
            m_p, v_p = self.m[idx], self.v[idx]
            # Biases and LayerNorm gains are 1-D; decaying them hurts. Standard
            # practice is to decay only the 2-D weight matrices.
            decay = self.weight_decay if p.rows > 1 else 0.0
            for i in range(p.rows):
                g_row, w_row, m_row, v_row = p.grad[i], p.data[i], m_p[i], v_p[i]
                for j in range(p.cols):
                    g = g_row[j]
                    m_row[j] = b1 * m_row[j] + (1 - b1) * g
                    v_row[j] = b2 * v_row[j] + (1 - b2) * g * g
                    m_hat = m_row[j] / bias1
                    v_hat = v_row[j] / bias2
                    # Decoupled: decay touches the weight, not the gradient.
                    w_row[j] -= lr * (m_hat / (math.sqrt(v_hat) + eps) + decay * w_row[j])

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #
def _canonical(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def save_checkpoint(path: str, model: GPT, step: int, loss: float,
                    extra: dict | None = None) -> str:
    """Write the model to JSON and return the sha256 of its weights+config.

    JSON, not pickle, and deliberately: a checkpoint format that can execute
    code on load is a supply-chain problem, not a convenience. (Lab 11.)
    """
    c = model.config
    payload = {
        "config": {
            "vocab_size": c.vocab_size, "block_size": c.block_size,
            "n_layer": c.n_layer, "n_head": c.n_head, "n_embd": c.n_embd,
        },
        "params": [p.data for p in model.parameters()],
    }
    digest = hashlib.sha256(_canonical(payload)).hexdigest()

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump({**payload, "step": step, "loss": loss,
                   "sha256": digest, **(extra or {})}, fh)
    return digest


def checkpoint_hash(path: str) -> str:
    """Recompute the hash from a file on disk -- the integrity check."""
    with open(path) as fh:
        saved = json.load(fh)
    return hashlib.sha256(
        _canonical({"config": saved["config"], "params": saved["params"]})
    ).hexdigest()


def load_model(path: str) -> GPT:
    with open(path) as fh:
        saved = json.load(fh)
    model = GPT(GPTConfig(**saved["config"]), seed=0)
    for p, values in zip(model.parameters(), saved["params"]):
        p.data = [list(row) for row in values]
    return model


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
@dataclass
class Sample:
    step: int
    text: str


@dataclass
class CheckpointRef:
    step: int
    path: str
    sha256: str


@dataclass
class TrainResult:
    losses: list[float] = field(default_factory=list)
    val_losses: list[tuple[int, float]] = field(default_factory=list)
    grad_norms: list[float] = field(default_factory=list)
    lrs: list[float] = field(default_factory=list)
    samples: list[Sample] = field(default_factory=list)
    checkpoints: list[CheckpointRef] = field(default_factory=list)
    final_loss: float = 0.0
    final_val_loss: float = 0.0
    wall_seconds: float = 0.0
    num_params: int = 0

    def loss_curve_text(self) -> str:
        lines = ["step\tlr\ttrain_loss\tgrad_norm"]
        for i, (loss, lr, gn) in enumerate(zip(self.losses, self.lrs, self.grad_norms)):
            lines.append(f"{i}\t{lr:.6f}\t{loss:.6f}\t{gn:.6f}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def estimate_loss(model: GPT, tokens: list[int], cfg: TrainConfig,
                  rng: random.Random, batches: int) -> float:
    """Average loss over a few batches. No backward pass, so no gradients."""
    total = 0.0
    count = 0
    for _ in range(batches):
        for x, y in get_batch(tokens, cfg.block_size, 1, rng):
            total += model.loss(x, y).data[0][0]
            count += 1
    return total / max(1, count)


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #
def train(cfg: TrainConfig, dataset: Dataset, verbose: bool = True) -> TrainResult:
    """Train a GPT and return everything the post needs to plot."""
    # One RNG per concern, all seeded off cfg.seed, so a run is reproducible
    # without one stream's consumption shifting another's.
    batch_rng = random.Random(cfg.seed)
    eval_rng = random.Random(cfg.seed + 1)
    sample_rng = random.Random(cfg.seed + 2)

    model = GPT(
        GPTConfig(vocab_size=dataset.vocab_size, block_size=cfg.block_size,
                  n_layer=cfg.n_layer, n_head=cfg.n_head, n_embd=cfg.n_embd),
        seed=cfg.seed,
    )
    params = model.parameters()
    opt = AdamW(params, beta1=cfg.beta1, beta2=cfg.beta2, weight_decay=cfg.weight_decay)

    result = TrainResult(num_params=model.num_params())
    started = time.time()

    if verbose:
        print(f"  config '{cfg.name}': {model.num_params():,} parameters · "
              f"{cfg.n_layer} layers · {cfg.n_embd} dim · context {cfg.block_size}")
        print(f"  {len(dataset.train_ids):,} train tokens · vocab {dataset.vocab_size}\n")
        print("  step |     lr    | train loss | grad norm")
        print("  -----+-----------+------------+----------")

    for step in range(cfg.max_steps):
        lr = lr_at(step, cfg)

        # --- forward + backward over the batch -----------------------------
        # A "batch" here is a loop that accumulates gradients before stepping.
        # Summing the loss over B sequences and dividing by B is exactly what a
        # batched matrix version computes -- just without the extra dimension.
        opt.zero_grad()
        batch_loss = 0.0
        batch = get_batch(dataset.train_ids, cfg.block_size, cfg.batch_size, batch_rng)
        for x, y in batch:
            loss = model.loss(x, y)
            batch_loss += loss.data[0][0]
            loss.backward()
        batch_loss /= len(batch)

        # Gradients accumulated the SUM over the batch; scale to the mean.
        for p in params:
            for row in p.grad:
                for i in range(len(row)):
                    row[i] /= len(batch)

        # --- clip, then step ------------------------------------------------
        grad_norm = clip_grad_norm(params, cfg.grad_clip)
        opt.step(lr)

        result.losses.append(batch_loss)
        result.lrs.append(lr)
        result.grad_norms.append(grad_norm)

        if verbose and (step % max(1, cfg.max_steps // 20) == 0 or step == cfg.max_steps - 1):
            print(f"  {step:4d} | {lr:.7f} |   {batch_loss:7.4f}  |  {grad_norm:7.4f}")

        # --- periodic validation --------------------------------------------
        is_last = step == cfg.max_steps - 1
        if cfg.eval_interval and ((step + 1) % cfg.eval_interval == 0 or is_last):
            val = estimate_loss(model, dataset.val_ids, cfg, eval_rng, cfg.eval_batches)
            result.val_losses.append((step, val))
            result.final_val_loss = val

        # --- checkpoint + sample --------------------------------------------
        # Step 0 is sampled deliberately: it is the "before" picture, and an
        # untrained model's output is the only honest baseline for "improved".
        if cfg.sample_every and (step == 0 or (step + 1) % cfg.sample_every == 0 or is_last):
            ids = model.generate(
                dataset.train_ids[:1], max_new_tokens=cfg.sample_tokens,
                rng=sample_rng, temperature=cfg.temperature, top_k=cfg.top_k,
            )
            result.samples.append(Sample(step=step, text=dataset.tokenizer.decode(ids)))

            path = os.path.join(cfg.out_dir, f"ckpt_step{step:05d}.json")
            digest = save_checkpoint(path, model, step=step, loss=batch_loss,
                                     extra={"config_name": cfg.name, "seed": cfg.seed})
            result.checkpoints.append(CheckpointRef(step=step, path=path, sha256=digest))

    result.final_loss = result.losses[-1] if result.losses else 0.0
    result.wall_seconds = time.time() - started
    return result
