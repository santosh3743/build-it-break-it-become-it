"""
Lab 03 — the post-training stack, running on Lab 02's engine.

Three stages, three different things being learned:

    pretrain  ->  fluent, useless      (learns the language)
    SFT       ->  follows instructions (learns the *format* of being asked)
    DPO       ->  follows preferences  (learns which valid answer we want)

Everything here uses Lab 02's autograd, GPT and AdamW unchanged. That is the
point: post-training is not a different kind of machine, it is the same gradient
descent pointed at a different objective.

The one genuinely new mechanism is the DPO loss, and specifically the *frozen
reference model*. DPO optimizes the policy relative to where it started. Without
that anchor, "prefer hedged answers" collapses into "emit the word 'roughly'
forever" — which is exactly the reward-hacking the KL term in RLHF exists to
prevent. Here it is not a penalty bolted on; it is inside the loss.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field

from labs_path import lab01_pipeline, lab02_engine

autograd, gpt_model, lab02_train = lab02_engine()
Tensor = autograd.Tensor
GPT, GPTConfig = gpt_model.GPT, gpt_model.GPTConfig


# --------------------------------------------------------------------------- #
# Two autograd ops Lab 02 did not need
# --------------------------------------------------------------------------- #
def select_rows(x: Tensor, indices: list[int]) -> Tensor:
    """Keep only certain rows (positions) of a tensor.

    DPO scores the *response* tokens, not the prompt. Without this the model
    would be rewarded for how it phrases the question it was asked.
    """
    out = Tensor([x.data[i][:] for i in indices], (x,))

    def _backward() -> None:
        for pos, src in enumerate(indices):
            g_row, x_row = out.grad[pos], x.grad[src]
            for j in range(x.cols):
                x_row[j] += g_row[j]

    out._backward = _backward
    return out


def log_sigmoid(x: Tensor) -> Tensor:
    """Elementwise log σ(x) on a 1x1 tensor.  d/dx log σ(x) = σ(-x)."""
    value = x.data[0][0]
    if value >= 0:
        out_value = -math.log1p(math.exp(-value))
    else:
        out_value = value - math.log1p(math.exp(value))
    out = Tensor([[out_value]], (x,))

    def _backward() -> None:
        # σ(-v), computed stably.
        v = value
        sig_neg = 1.0 / (1.0 + math.exp(v)) if v >= 0 else math.exp(-v) / (1.0 + math.exp(-v))
        x.grad[0][0] += out.grad[0][0] * sig_neg

    out._backward = _backward
    return out


# --------------------------------------------------------------------------- #
# Tokenization -- reuse Lab 01's tokenizer
# --------------------------------------------------------------------------- #
@dataclass
class Corpus:
    tokenizer: object
    pretrain_ids: list[int] = field(default_factory=list)
    sft_sequences: list[list[int]] = field(default_factory=list)
    pairs: list[tuple[list[int], list[int], list[int]]] = field(default_factory=list)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size


def build_corpus(seed: int = 3) -> Corpus:
    """One shared vocabulary across all three stages.

    This matters: comparing base / SFT / DPO is only meaningful if they speak
    the same token language. Re-tokenizing between stages is a classic way to
    produce a comparison table that means nothing.
    """
    import preference_data as pd

    pipeline = lab01_pipeline()
    pre_texts = pd.pretrain_texts(seed)
    sft = pd.sft_texts(seed)
    pairs = pd.preference_pairs(seed)

    everything = pre_texts + sft + [p + " " + c + " " + r for p, c, r in pairs]
    tokenizer = pipeline.Tokenizer().build_vocab(everything, max_vocab=2000)

    return Corpus(
        tokenizer=tokenizer,
        pretrain_ids=[t for text in pre_texts for t in tokenizer.encode(text)],
        sft_sequences=[tokenizer.encode(t) for t in sft],
        pairs=[(tokenizer.encode(p), tokenizer.encode(c), tokenizer.encode(r))
               for p, c, r in pairs],
    )


# --------------------------------------------------------------------------- #
# Stage 1 + 2: pretraining and SFT (same loop, different data)
# --------------------------------------------------------------------------- #
@dataclass
class StageResult:
    losses: list[float] = field(default_factory=list)

    @property
    def first(self) -> float:
        return self.losses[0] if self.losses else 0.0

    @property
    def final(self) -> float:
        return self.losses[-1] if self.losses else 0.0


def _make_model(corpus: Corpus, block_size: int, seed: int,
                n_layer: int = 2, n_head: int = 3, n_embd: int = 48) -> GPT:
    """The base architecture for all three stages.

    `block_size` must exceed the longest instruction+response sequence, or the
    model never sees a response *end* and learns to ramble. Check it with
    `max(len(s) for s in corpus.sft_sequences)` before you change it.
    """
    return GPT(GPTConfig(vocab_size=corpus.vocab_size, block_size=block_size,
                         n_layer=n_layer, n_head=n_head, n_embd=n_embd), seed=seed)


def train_on_stream(model: GPT, tokens: list[int], steps: int, batch_size: int = 8,
                    lr: float = 3e-3, seed: int = 0) -> StageResult:
    """Plain next-token training over a flat token stream (the base model)."""
    rng = random.Random(seed)
    opt = lab02_train.AdamW(model.parameters(), weight_decay=0.01)
    block = model.config.block_size
    result = StageResult()

    for step in range(steps):
        opt.zero_grad()
        total = 0.0
        for _ in range(batch_size):
            i = rng.randint(0, len(tokens) - block - 2)
            x, y = tokens[i:i + block], tokens[i + 1:i + block + 1]
            loss = model.loss(x, y)
            total += loss.data[0][0]
            loss.backward()
        _scale_grads(model, batch_size)
        lab02_train.clip_grad_norm(model.parameters(), 1.0)
        opt.step(_cosine_lr(step, steps, lr))
        result.losses.append(total / batch_size)
    return result


def sft_train(model: GPT, corpus: Corpus, steps: int, batch_size: int = 8,
              lr: float = 2e-3, seed: int = 0) -> StageResult:
    """Supervised fine-tuning on instruction/response pairs.

    The loss is masked to the *response* tokens. Training on the prompt tokens
    teaches the model to generate questions, which is not the job.
    """
    rng = random.Random(seed)
    opt = lab02_train.AdamW(model.parameters(), weight_decay=0.01)
    block = model.config.block_size
    result = StageResult()

    for step in range(steps):
        opt.zero_grad()
        total = 0.0
        for _ in range(batch_size):
            seq = rng.choice(corpus.sft_sequences)[: block + 1]
            if len(seq) < 3:
                continue
            x, y = seq[:-1], seq[1:]
            loss = model.loss(x, y)
            total += loss.data[0][0]
            loss.backward()
        _scale_grads(model, batch_size)
        lab02_train.clip_grad_norm(model.parameters(), 1.0)
        opt.step(_cosine_lr(step, steps, lr))
        result.losses.append(total / batch_size)
    return result


# --------------------------------------------------------------------------- #
# Stage 3: DPO
# --------------------------------------------------------------------------- #
def sequence_logprob(model: GPT, prompt_ids: list[int], response_ids: list[int],
                     differentiable: bool = True, return_count: bool = False):
    """log π(response | prompt), summed over the response tokens only.

    Built out of Lab 02's `cross_entropy`, which returns the *mean* negative
    log-likelihood over the rows it scores. So:

        sum log π  =  -(mean NLL) × (number of response tokens)
    """
    block = model.config.block_size
    full = (prompt_ids + response_ids)[:block + 1]
    if len(full) < 2:
        raise ValueError("sequence too short to score")

    x, y = full[:-1], full[1:]
    logits = model.forward(x)

    # Rows of `logits` predicting a response token. Position i of x predicts
    # y[i] = full[i+1]; that is a response token once i+1 >= len(prompt_ids).
    response_rows = [i for i in range(len(x)) if i + 1 >= len(prompt_ids)]
    if not response_rows:
        raise ValueError("prompt fills the whole context; nothing left to score")

    selected_logits = select_rows(logits, response_rows)
    targets = [y[i] for i in response_rows]
    mean_nll = autograd.cross_entropy(selected_logits, targets)
    total = autograd.scale(mean_nll, -float(len(response_rows)))
    value = total if differentiable else total.data[0][0]
    return (value, len(response_rows)) if return_count else value


@dataclass
class DPOResult:
    losses: list[float] = field(default_factory=list)
    margins: list[float] = field(default_factory=list)
    accuracies: list[float] = field(default_factory=list)

    @property
    def final_margin(self) -> float:
        return self.margins[-1] if self.margins else 0.0

    @property
    def final_accuracy(self) -> float:
        return self.accuracies[-1] if self.accuracies else 0.0


def dpo_train(model: GPT, corpus: Corpus, steps: int, beta: float = 0.1,
              batch_size: int = 4, lr: float = 5e-4, seed: int = 0,
              nll_weight: float = 0.5) -> DPOResult:
    """Direct Preference Optimization against a frozen copy of the start model.

    `reference` is deep-copied once and never updated. Every gradient is
    computed relative to it, which is what keeps the policy from drifting off
    into text that scores well and reads like nothing.

    ------------------------------------------------------------------
    Why there is an NLL term in a DPO loss
    ------------------------------------------------------------------
    Pure DPO has a well-known pathology, and you can watch it happen in this
    lab: set `nll_weight=0.0` and run the demo. Preference accuracy hits 100%,
    the margin climbs past 8, the loss goes to ~0.03 -- every number says the
    run is a triumph -- and the samples come out as

        "is . based . based . based . based . based on . measure"

    The loss only ever asked for logπ(chosen) - logπ(rejected) to grow. It never
    asked for logπ(chosen) to stay *high*, and the cheapest way to satisfy it is
    to crush the rejected response by wrecking the shared distribution. DPO is
    known to reduce the likelihood of the chosen response too.

    So we add a small supervised term on the chosen response:

        L = -log σ(β·margin)  +  λ · NLL(chosen | prompt)

    This is what Llama 3 shipped as "DPO with an NLL regularizer" (the RPO
    objective). The DPO term decides which answer wins; the NLL term insists the
    winner still be fluent text. `λ = 0.5` here.

    The lesson generalizes past this lab: a preference objective optimizes the
    gap between two things, and a gap can always be widened from the wrong end.
    """
    rng = random.Random(seed)
    reference = copy.deepcopy(model)
    opt = lab02_train.AdamW(model.parameters(), weight_decay=0.0)
    result = DPOResult()

    for step in range(steps):
        opt.zero_grad()
        step_loss = 0.0
        step_margin = 0.0
        correct = 0
        used = 0

        for _ in range(batch_size):
            prompt, chosen, rejected = rng.choice(corpus.pairs)
            try:
                # Reference log-probs are constants: no graph, no gradient.
                ref_c = sequence_logprob(reference, prompt, chosen, differentiable=False)
                ref_r = sequence_logprob(reference, prompt, rejected, differentiable=False)
                pol_c, n_chosen = sequence_logprob(model, prompt, chosen, return_count=True)
                pol_r = sequence_logprob(model, prompt, rejected)
            except ValueError:
                continue

            # margin = β[(logπ_c - logref_c) - (logπ_r - logref_r)]
            diff = autograd.add(pol_c, autograd.scale(pol_r, -1.0))
            margin = autograd.scale(diff, beta)
            const = beta * (ref_c - ref_r)
            margin = autograd.add(margin, Tensor([[-const]]))

            loss = autograd.scale(log_sigmoid(margin), -1.0)   # -log σ(margin)

            # + λ·NLL(chosen): keeps the winning answer fluent. See the
            # docstring -- without this the run reward-hacks into repetition.
            if nll_weight > 0.0:
                nll_chosen = autograd.scale(pol_c, -1.0 / n_chosen)
                loss = autograd.add(loss, autograd.scale(nll_chosen, nll_weight))

            loss.backward()

            step_loss += loss.data[0][0]
            step_margin += margin.data[0][0]
            correct += 1 if margin.data[0][0] > 0 else 0
            used += 1

        if used == 0:
            continue
        _scale_grads(model, used)
        lab02_train.clip_grad_norm(model.parameters(), 1.0)
        opt.step(_cosine_lr(step, steps, lr))

        result.losses.append(step_loss / used)
        result.margins.append(step_margin / used)
        result.accuracies.append(correct / used)
    return result


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _scale_grads(model: GPT, n: int) -> None:
    """Gradients accumulated the SUM over a batch; convert to the mean."""
    for p in model.parameters():
        for row in p.grad:
            for i in range(len(row)):
                row[i] /= n


def _cosine_lr(step: int, total: int, peak: float, warmup_frac: float = 0.1) -> float:
    warmup = max(1, int(total * warmup_frac))
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def generate_response(model: GPT, corpus: Corpus, instruction: str,
                      max_new_tokens: int = 22, seed: int = 0,
                      temperature: float = 0.7, top_k: int = 12) -> str:
    """Generate the response half only, so the table shows answers not echoes."""
    import preference_data as pd

    prompt_ids = corpus.tokenizer.encode(pd.format_prompt(instruction))
    prompt_ids = prompt_ids[-model.config.block_size:]
    out = model.generate(prompt_ids, max_new_tokens=max_new_tokens,
                         rng=random.Random(seed), temperature=temperature, top_k=top_k)
    return corpus.tokenizer.decode(out[len(prompt_ids):])
