"""
Lab 02 — the data side: import Lab 01, train on what it produces.

This is the first cross-lab link in the series, and it is deliberate. Lab 02
owns no tokenizer, no dedup, no PII scrubbing. It calls `run_pipeline` from
`labs/01-data-engine/pipeline.py` and trains on the shards that come back. Break
Lab 01 and this lab's loss curve gets worse -- which is exactly the relationship
a real training stack has with its data engine.

The batching contract is the one every language model shares:

    x = tokens[i     : i + block_size]
    y = tokens[i + 1 : i + block_size + 1]

Predict the next token at every position at once. One forward pass produces
`block_size` training signals, not one. That is why this works at all.
"""

from __future__ import annotations

import importlib.util
import os
import random
import sys
from dataclasses import dataclass, field
from types import ModuleType

LAB01_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "01-data-engine")

_lab01_cache: ModuleType | None = None


def load_lab01() -> ModuleType:
    """Import `labs/01-data-engine/pipeline.py` as a module.

    It lives in a sibling folder with a leading digit, so it is not importable
    as a normal package name -- we load it from its file path instead.
    """
    global _lab01_cache
    if _lab01_cache is not None:
        return _lab01_cache

    path = os.path.join(LAB01_DIR, "pipeline.py")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Lab 02 trains on Lab 01's output, but {path} is missing.\n"
            "Clone the whole repo (not just this folder) and run from "
            "labs/02-train-slm/."
        )

    spec = importlib.util.spec_from_file_location("lab01_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["lab01_pipeline"] = module
    spec.loader.exec_module(module)
    _lab01_cache = module
    return module


@dataclass
class Dataset:
    shards: list[list[int]]
    tokenizer: object
    report: object
    all_ids: list[int] = field(default_factory=list)
    train_ids: list[int] = field(default_factory=list)
    val_ids: list[int] = field(default_factory=list)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size


def build_dataset(shard_size: int = 256, val_fraction: float = 0.1) -> Dataset:
    """Run Lab 01's pipeline over Lab 02's raw corpus and split the result.

    Deterministic end to end: same corpus in, same token ids out, every time.
    """
    lab01 = load_lab01()
    from sample_corpus import EVAL_TEXTS, RAW_TEXTS

    shards, tokenizer, report = lab01.run_pipeline(
        RAW_TEXTS, EVAL_TEXTS, shard_size=shard_size
    )

    all_ids = [t for shard in shards for t in shard]
    # A contiguous tail is held out, not a random sample: random holdout on a
    # token stream leaks context across the split.
    n_val = max(1, int(len(all_ids) * val_fraction))
    return Dataset(
        shards=shards,
        tokenizer=tokenizer,
        report=report,
        all_ids=all_ids,
        train_ids=all_ids[:-n_val],
        val_ids=all_ids[-n_val:],
    )


def get_batch(tokens: list[int], block_size: int, batch_size: int,
              rng: random.Random) -> list[tuple[list[int], list[int]]]:
    """Sample `batch_size` random windows. Returns (inputs, targets) pairs."""
    if len(tokens) < block_size + 1:
        raise ValueError(
            f"need at least {block_size + 1} tokens to form a batch, got {len(tokens)}"
        )
    high = len(tokens) - block_size - 1
    batch = []
    for _ in range(batch_size):
        i = rng.randint(0, high)
        batch.append((tokens[i:i + block_size], tokens[i + 1:i + block_size + 1]))
    return batch
