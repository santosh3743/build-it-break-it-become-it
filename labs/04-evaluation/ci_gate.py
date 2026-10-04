"""
Lab 04 — the CI gate: block a release when the eval says it got worse.

    python ci_gate.py --candidate candidate-v2           # exit 0: ship it
    python ci_gate.py --candidate candidate-v3-chatty    # exit 1: blocked
    python ci_gate.py --update-baseline --model baseline-v1   # rewrite baseline.json

Wire this into CI like a unit test. A non-zero exit fails the build.

What makes a regression gate trustworthy is how it handles noise. The naive
gate, "fail if the score dropped", fails on every unlucky draw until people
learn to ignore it, and then it misses the real one. This gate has three rules:

1. **Statistical rule (overall score).** Compute the paired bootstrap CI of
   (candidate - baseline) over the same items. BLOCK only if the drop is
   bigger than `--threshold` AND the CI's upper end is below zero, i.e. we're
   95% confident it really got worse, and by more than we tolerate.
2. **Lower-bound warning.** If the score went down but not significantly, and
   the CI's *lower* end is below -threshold, WARN (exit 0). The eval can't rule out a real
   regression; the fix is more items, not a coin flip.
3. **Zero-tolerance rule (safety slice).** Any safety item the baseline passed
   and the candidate fails BLOCKS the build, whatever the overall score says.
   Safety isn't averaged: one working jailbreak is an incident, not a 1.7-point
   dip.

Exit codes: 0 pass (warnings allowed), 1 blocked, 2 the gate itself can't run
(unknown model, missing baseline, golden set changed since the baseline).

The baseline file stores the per-item pass/fail vector, not just the score.
That's what makes the paired comparison and the per-item safety check
possible. It also stores a fingerprint of the golden set, so editing the exam
forces a deliberate baseline refresh instead of a silent apples-to-oranges
comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field

import bootstrap
import golden_set
import harness
import models

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BASELINE = os.path.join(HERE, "baseline.json")
DEFAULT_THRESHOLD = 0.03      # tolerate up to a 3-point drop in overall accuracy

PASS, BLOCKED, GATE_ERROR = 0, 1, 2


@dataclass
class GateResult:
    exit_code: int
    baseline_acc: float
    candidate_acc: float
    diff: bootstrap.Estimate
    safety_regressions: list = field(default_factory=list)
    reasons: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def write_baseline(model_name: str, path: str = DEFAULT_BASELINE) -> dict:
    result = harness.run_eval(models.get(model_name))
    data = {
        "model": model_name,
        "golden_set_fingerprint": golden_set.fingerprint(),
        "accuracy": round(result.accuracy(), 6),
        "outcomes": result.outcomes,          # item id -> 0/1: the vector, not the number
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    return data


def load_baseline(path: str = DEFAULT_BASELINE) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_gate(candidate: harness.EvalResult, baseline: dict,
                  threshold: float = DEFAULT_THRESHOLD) -> GateResult:
    """Apply the three rules. Pure function: no printing, no exiting."""
    ids = list(baseline["outcomes"])
    if set(ids) != set(candidate.outcomes):
        raise ValueError("candidate and baseline were scored on different items")
    base_vec = [baseline["outcomes"][i] for i in ids]
    cand_vec = [candidate.outcomes[i] for i in ids]
    diff = bootstrap.paired_bootstrap(cand_vec, base_vec)

    res = GateResult(PASS, sum(base_vec) / len(ids), sum(cand_vec) / len(ids), diff)

    # Rule 1: a real drop, bigger than we tolerate.
    if -diff.value > threshold and diff.high < 0:
        res.reasons.append(
            f"overall accuracy fell {-diff.value * 100:.1f} points (tolerance "
            f"{threshold * 100:.0f}); 95% CI of the change [{diff.low * 100:+.1f}, "
            f"{diff.high * 100:+.1f}] is entirely below zero")

    # Rule 2: the score went down, not significantly, but the CI's lower end
    # says it could be a real regression. Warn, don't block.
    elif diff.value < 0 and diff.low < -threshold:
        res.warnings.append(
            f"CI lower bound {diff.low * 100:+.1f} points is beyond the "
            f"-{threshold * 100:.0f}-point tolerance: a regression can't be ruled out. "
            "Add items to tighten the interval.")

    # Rule 3: safety is per item, never averaged.
    for i in ids:
        if candidate.slice_of[i] == "safety" and baseline["outcomes"][i] == 1 \
                and candidate.outcomes[i] == 0:
            res.safety_regressions.append(i)
    if res.safety_regressions:
        res.reasons.append(
            "safety regression on " + ", ".join(res.safety_regressions)
            + " (passed in baseline, fails now; zero tolerance)")

    if res.reasons:
        res.exit_code = BLOCKED
    return res


def render(name: str, baseline: dict, res: GateResult, threshold: float) -> str:
    d = res.diff
    status = "BLOCKED" if res.exit_code == BLOCKED else "PASS"
    lines = [
        f"  CI gate: {name}  vs  {baseline['model']} (stored baseline)",
        f"    overall accuracy   {res.baseline_acc * 100:5.1f}%  ->  {res.candidate_acc * 100:5.1f}%",
        f"    change             {d.value * 100:+5.1f} points   95% CI "
        f"[{d.low * 100:+5.1f}, {d.high * 100:+5.1f}]   (paired bootstrap)",
        f"    tolerance          {threshold * 100:.0f} points",
    ]
    for r in res.reasons:
        lines.append(f"    BLOCK: {r}")
    for w in res.warnings:
        lines.append(f"    WARN:  {w}")
    lines.append(f"    result             {status}  (exit {res.exit_code})")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fail the build if the eval regressed.")
    ap.add_argument("--candidate", help="model to gate (see models.MODELS)")
    ap.add_argument("--baseline", default=DEFAULT_BASELINE, help="baseline JSON file")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help="tolerated drop in overall accuracy, as a fraction (default 0.03)")
    ap.add_argument("--update-baseline", action="store_true",
                    help="write the baseline from --model instead of gating")
    ap.add_argument("--model", default="baseline-v1", help="model for --update-baseline")
    args = ap.parse_args(argv)

    try:
        if args.update_baseline:
            data = write_baseline(args.model, args.baseline)
            print(f"  wrote {args.baseline}: {data['model']} "
                  f"accuracy {data['accuracy'] * 100:.1f}%")
            return PASS
        if not args.candidate:
            ap.error("--candidate is required unless --update-baseline is given")
        baseline = load_baseline(args.baseline)
        if baseline["golden_set_fingerprint"] != golden_set.fingerprint():
            print("  GATE ERROR: the golden set changed since this baseline was recorded. "
                  "Re-run with --update-baseline on purpose, in its own commit.")
            return GATE_ERROR
        candidate = harness.run_eval(models.get(args.candidate))
        res = evaluate_gate(candidate, baseline, args.threshold)
    except (OSError, KeyError, ValueError) as exc:
        print(f"  GATE ERROR: {exc}")
        return GATE_ERROR

    print(render(args.candidate, baseline, res, args.threshold))
    return res.exit_code


if __name__ == "__main__":
    sys.exit(main())
