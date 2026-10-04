"""
Lab 04 — the eval harness: tasks, a runner and a scorecard with intervals.

The shape is the same one large harnesses (EleutherAI's lm-evaluation-harness,
for example) use, shrunk to fit on a screen:

    Task      = a named list of items + the scorers that grade them
    run_eval  = for each item: prompt the model, score the output, keep the
                per-item pass/fail (never only the total)
    scorecard = per-task accuracy with a 95% bootstrap CI, then overall

Keeping per-item results is the most important habit here. With only a
total you can't compute a paired comparison, you can't see which items
flipped between versions, and the CI gate can't tell a safety regression from
noise. Store the vector, not the number.
"""

from __future__ import annotations

from dataclasses import dataclass

import bootstrap
import scorers
from golden_set import GOLDEN_SET, SLICES, Item


@dataclass(frozen=True)
class Task:
    name: str
    items: list


def tasks(items: list[Item] = GOLDEN_SET) -> list[Task]:
    """One task per slice of the golden set."""
    return [Task(s, [it for it in items if it.slice == s]) for s in SLICES]


@dataclass
class EvalResult:
    model: str
    outcomes: dict          # item id -> 1 (pass) or 0 (fail), in golden-set order
    outputs: dict           # item id -> the raw output string
    slice_of: dict          # item id -> slice name

    def vector(self, slice_name: str | None = None) -> list[int]:
        return [v for k, v in self.outcomes.items()
                if slice_name is None or self.slice_of[k] == slice_name]

    def accuracy(self, slice_name: str | None = None) -> float:
        v = self.vector(slice_name)
        return sum(v) / len(v)

    def ci(self, slice_name: str | None = None) -> bootstrap.Estimate:
        return bootstrap.bootstrap_ci(self.vector(slice_name))


def run_eval(model, items: list[Item] = GOLDEN_SET) -> EvalResult:
    """Run `model.generate` on every item and grade each output."""
    outcomes, outputs, slice_of = {}, {}, {}
    for item in items:
        out = model.generate(item)
        outputs[item.id] = out
        outcomes[item.id] = 1 if scorers.score(item, out) else 0
        slice_of[item.id] = item.slice
    return EvalResult(model.name, outcomes, outputs, slice_of)


def _fmt(est: bootstrap.Estimate) -> str:
    return f"{est.value * 100:5.1f}%  [{est.low * 100:5.1f}, {est.high * 100:5.1f}]"


def scorecard(results: list[EvalResult]) -> str:
    """Per-slice and overall accuracy, each with its 95% bootstrap CI."""
    names = [r.model for r in results]
    width = 26
    head = "  slice (n)        | " + " | ".join(f"{n:^{width}s}" for n in names)
    rule = "  " + "-" * (len(head) - 2)
    lines = [head, rule]
    for s in SLICES + [None]:
        if s is None:
            lines.append(rule)
        n = len(results[0].vector(s))
        label = f"{(s or 'OVERALL')} ({n})"
        cells = [f"{_fmt(r.ci(s)):^{width}s}" for r in results]
        lines.append(f"  {label:<16s} | " + " | ".join(cells))
    lines.append("")
    lines.append("  cells: accuracy  [95% bootstrap CI, 2000 resamples, seed 1234]")
    return "\n".join(lines)
