"""
Lab 03 — the whole post-training pipeline as one callable.

`run_post_training()` returns the three models plus every number the demo and
the tests need, so both read the same run rather than each doing their own.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import evaluate
import post_training as pt
import preference_data as pd


@dataclass
class PipelineConfig:
    # Must exceed the longest instruction+response sequence (currently 30
    # tokens) or SFT trains on truncated answers and the model never learns to
    # stop. `tests/test_post_training.py` asserts this holds.
    block_size: int = 32
    pretrain_steps: int = 120
    sft_steps: int = 260
    dpo_steps: int = 60
    dpo_beta: float = 0.1
    dpo_lr: float = 3e-4
    # Set to 0.0 to watch pure DPO reward-hack itself into repetition --
    # see the `dpo_train` docstring and Exercise 2.
    dpo_nll_weight: float = 0.5
    batch_size: int = 8
    dpo_batch_size: int = 4
    seed: int = 11
    gen_tokens: int = 20


@dataclass
class PipelineResult:
    corpus: object
    base: object
    sft: object
    dpo: object
    pretrain_losses: list[float] = field(default_factory=list)
    sft_losses: list[float] = field(default_factory=list)
    dpo_losses: list[float] = field(default_factory=list)
    dpo_margins: list[float] = field(default_factory=list)
    dpo_accuracies: list[float] = field(default_factory=list)
    scored: list[evaluate.ModelScores] = field(default_factory=list)

    def by_name(self, name: str) -> evaluate.ModelScores:
        return next(m for m in self.scored if m.name == name)


def run_post_training(cfg: PipelineConfig | None = None,
                      verbose: bool = False) -> PipelineResult:
    cfg = cfg or PipelineConfig()
    corpus = pt.build_corpus(seed=cfg.seed)

    # --- Stage 1: pretrain. Fluent, and completely unable to follow an order.
    if verbose:
        print("  [1/3] pretraining the base model...")
    base = pt._make_model(corpus, cfg.block_size, seed=cfg.seed)
    pre = pt.train_on_stream(base, corpus.pretrain_ids, steps=cfg.pretrain_steps,
                             batch_size=cfg.batch_size, seed=cfg.seed)

    # --- Stage 2: SFT. Branch from the base so the comparison is honest --
    # base and SFT share an ancestor rather than being two unrelated runs.
    if verbose:
        print("  [2/3] supervised fine-tuning on instruction pairs...")
    sft_model = copy.deepcopy(base)
    sft = pt.sft_train(sft_model, corpus, steps=cfg.sft_steps,
                       batch_size=cfg.batch_size, seed=cfg.seed + 1)

    # --- Stage 3: DPO, branching from SFT (never from the base -- DPO on a
    # model that cannot follow the format has nothing to express a preference
    # over).
    if verbose:
        print("  [3/3] DPO on preference pairs...")
    dpo_model = copy.deepcopy(sft_model)
    dpo = pt.dpo_train(dpo_model, corpus, steps=cfg.dpo_steps, beta=cfg.dpo_beta,
                       batch_size=cfg.dpo_batch_size, lr=cfg.dpo_lr, seed=cfg.seed + 2,
                       nll_weight=cfg.dpo_nll_weight)

    # --- Evaluate all three on the same fixed prompts, same sampling seed.
    scored = []
    for name, model in (("base", base), ("SFT", sft_model), ("DPO", dpo_model)):
        responses = [
            pt.generate_response(model, corpus, instruction,
                                 max_new_tokens=cfg.gen_tokens, seed=cfg.seed + i)
            for i, instruction in enumerate(pd.EVAL_INSTRUCTIONS)
        ]
        scored.append(evaluate.score_model(name, responses))

    return PipelineResult(
        corpus=corpus, base=base, sft=sft_model, dpo=dpo_model,
        pretrain_losses=pre.losses, sft_losses=sft.losses,
        dpo_losses=dpo.losses, dpo_margins=dpo.margins,
        dpo_accuracies=dpo.accuracies, scored=scored,
    )
