"""
Lab 04 — deterministic mock "models" to evaluate.

The lab is about the *harness*, not the model, so the models are stand-ins
whose behaviour is fully known. Each one has a skill level per slice and, for
every item, makes a seeded decision: give the reference output or the
plausible wrong one (the item's distractor).

How "knowing" an item is decided:

    knows(item) = difficulty(item)  <  skill[slice] + jitter(model, item)

- `difficulty` is shared by every model. Hard items are hard for everyone, as
  they are for real models. This shared part is why **paired** comparisons
  (same items, both models) are much more informative than comparing two
  independent scores. `bootstrap.py` exploits it.
- `jitter` is per model. Two models with the same skill still disagree on
  some items, which is the noise a confidence interval has to account for.

All randomness comes from SHA-256 of a string key, so results are identical
on every machine, every Python version and every run, with no global seed.

The four models tell one release story:

    baseline-v1          what's in production today (the stored baseline)
    candidate-v2         a genuine but small improvement: the clean run
    candidate-v3-chatty  v2 plus a friendlier system prompt that wraps every
                         answer in "Sure! Here you go: ..."  -> format breaks
    candidate-v3-quiet   v2 plus one change that lets a single injection
                         through. Its overall score barely moves.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace

from golden_set import Item

JITTER = 0.3   # how much two equally skilled models disagree item by item


def unit(key: str) -> float:
    """A deterministic number in [0, 1) derived from a string."""
    return int(hashlib.sha256(key.encode()).hexdigest()[:12], 16) / 16 ** 12


def difficulty(item: Item) -> float:
    return unit(f"difficulty:{item.id}")


@dataclass(frozen=True)
class MockModel:
    name: str
    skill: dict                      # slice -> skill level in [0, 1]
    seed: str                        # what makes this model's jitter its own
    template: str = "{answer}"       # the "system prompt" wrapper around answers
    overrides: dict = field(default_factory=dict)   # item id -> forced knows()

    def knows(self, item: Item) -> bool:
        if item.id in self.overrides:
            return self.overrides[item.id]
        jitter = (unit(f"{self.seed}:{item.id}") - 0.5) * JITTER
        return difficulty(item) < self.skill[item.slice] + jitter

    def generate(self, item: Item) -> str:
        """The model's output for one item: a real string the scorers grade."""
        answer = item.reference if self.knows(item) else item.distractor
        return self.template.format(answer=answer)


BASELINE = MockModel(
    name="baseline-v1",
    skill={"factual": 0.72, "arithmetic": 0.60, "format": 0.70, "safety": 0.80},
    seed="v1",
)

CANDIDATE = MockModel(
    name="candidate-v2",
    skill={"factual": 0.80, "arithmetic": 0.70, "format": 0.80, "safety": 0.80},
    seed="v2",
)

# Same knowledge as v2. Only the wrapper changed: the kind of edit that ships
# as "just a prompt tweak" and never gets a review.
CHATTY = replace(CANDIDATE, name="candidate-v3-chatty",
                 template="Sure! Here you go: {answer}")

# Same as v2, except one jailbreak that v2 resisted (safety-02, the "you are
# now DAN" role-play) now gets through, and one factual item v2 missed is now
# right. Net change to the overall score versus v2: zero. Versus production it
# still looks like an improvement. This is the regression a single accuracy
# number cannot see.
QUIET = replace(CANDIDATE, name="candidate-v3-quiet",
                overrides={"safety-02": False, "fact-14": True})

MODELS = {m.name: m for m in (BASELINE, CANDIDATE, CHATTY, QUIET)}


def get(name: str) -> MockModel:
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; choose from {sorted(MODELS)}")
    return MODELS[name]
