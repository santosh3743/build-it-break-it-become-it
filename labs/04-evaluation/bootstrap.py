"""
Lab 04 — Deeper Dive: confidence intervals on eval scores, by bootstrap.

An eval score is an estimate. You ran the model on 60 items, but you care
about how it does on the thousands of similar items you didn't write down. A
different, equally reasonable set of 60 would give a different number. The
confidence interval says how different.

The bootstrap (Efron, 1979) answers "how much would this number move?"
without any formula: treat the items you have as the population, redraw a
same-size set from them *with replacement* many times, and look at the spread
of the score across redraws.

    observed:   [1 1 0 1 0 1 1 1 0 1]          accuracy 0.70
    resample 1: [1 0 1 1 1 1 0 1 1 1]          0.80
    resample 2: [0 0 1 1 0 1 1 0 1 1]          0.60
    ...  2000 times  ...
    95% CI = the 2.5th and 97.5th percentiles of those 2000 accuracies

Two functions:

- `bootstrap_ci`        one model's score, with an interval
- `paired_bootstrap`    the *difference* between two models on the same items.
                        This is the one to use when comparing models. Each
                        resample draws the same items for both, so the shared
                        "this item is hard for everyone" noise cancels out.

Everything is seeded, so the same inputs always give the same interval.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

N_RESAMPLES = 2000
SEED = 1234


@dataclass(frozen=True)
class Estimate:
    value: float      # the observed score (or difference)
    low: float        # lower end of the CI
    high: float       # upper end of the CI
    n: int            # number of items it was computed on

    @property
    def half_width(self) -> float:
        return (self.high - self.low) / 2

    def excludes_zero(self) -> bool:
        return self.low > 0 or self.high < 0


def _percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolated percentile of an already sorted list (q in [0, 1])."""
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def bootstrap_ci(outcomes: list[int], level: float = 0.95,
                 n_resamples: int = N_RESAMPLES, seed: int = SEED) -> Estimate:
    """Percentile bootstrap CI for the mean of 0/1 outcomes (i.e. accuracy)."""
    n = len(outcomes)
    if n == 0:
        raise ValueError("cannot bootstrap an empty list")
    rng = random.Random(seed)
    means = []
    for _ in range(n_resamples):
        total = 0
        for _ in range(n):
            total += outcomes[int(rng.random() * n)]
        means.append(total / n)
    means.sort()
    tail = (1 - level) / 2
    return Estimate(sum(outcomes) / n, _percentile(means, tail),
                    _percentile(means, 1 - tail), n)


def paired_bootstrap(candidate: list[int], baseline: list[int], level: float = 0.95,
                     n_resamples: int = N_RESAMPLES, seed: int = SEED) -> Estimate:
    """CI for mean(candidate) - mean(baseline), resampling items jointly.

    `candidate[i]` and `baseline[i]` must be the same item. Resampling item
    indices (not the two lists separately) is what makes this "paired".
    """
    if len(candidate) != len(baseline):
        raise ValueError("paired bootstrap needs both models scored on the same items")
    diffs = [c - b for c, b in zip(candidate, baseline)]
    n = len(diffs)
    rng = random.Random(seed)
    means = []
    for _ in range(n_resamples):
        total = 0
        for _ in range(n):
            total += diffs[int(rng.random() * n)]
        means.append(total / n)
    means.sort()
    tail = (1 - level) / 2
    return Estimate(sum(diffs) / n, _percentile(means, tail),
                    _percentile(means, 1 - tail), n)


def intervals_overlap(a: Estimate, b: Estimate) -> bool:
    return a.low <= b.high and b.low <= a.high
