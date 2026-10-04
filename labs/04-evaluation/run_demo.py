"""
Lab 04 — evaluation that actually means something.

    python run_demo.py

Five sections, each one a claim from the post, each printed from a real run:

  1. A scorecard with 95% bootstrap confidence intervals.
  2. Why the headline number misleads: "+5 points" that the data can't confirm.
  3. An LLM judge flipping its verdict when you swap the answer order, and the
     mitigations (position swap + explicit rubric) that fix it.
  4. pass@k with the unbiased estimator, and a model ranking that flips with k.
  5. A CI gate: a clean release passes, two seeded regressions get blocked,
     with the real process exit codes.

Offline, deterministic, standard library only. Runs in a few seconds.
"""

from __future__ import annotations

import os
import subprocess
import sys

import bootstrap
import harness
import judge
import models
import pass_at_k

HERE = os.path.dirname(os.path.abspath(__file__))
BAR = "=" * 78


def section(title: str) -> None:
    print("\n" + BAR)
    print(f"  {title}")
    print(BAR + "\n")


def main() -> None:
    print(BAR)
    print("  LAB 04 — EVALUATION THAT ACTUALLY MEANS SOMETHING")
    print(BAR)
    print("""
  Golden set: 60 hand-written items in 4 slices (factual, arithmetic,
  format, safety). Models: deterministic mocks, so every number below is
  reproducible to the last digit.""")

    # ------------------------------------------------------------------ 1 --
    section("1. SCORECARD  (accuracy with 95% bootstrap confidence intervals)")
    base = harness.run_eval(models.BASELINE)
    cand = harness.run_eval(models.CANDIDATE)
    print(harness.scorecard([base, cand]))

    # ------------------------------------------------------------------ 2 --
    section("2. IS candidate-v2 BETTER?  (why a single number misleads)")
    b_ci, c_ci = base.ci(), cand.ci()
    diff = bootstrap.paired_bootstrap(cand.vector(), base.vector())
    gained = sum(1 for k in base.outcomes if cand.outcomes[k] > base.outcomes[k])
    lost = sum(1 for k in base.outcomes if cand.outcomes[k] < base.outcomes[k])
    print(f"  headline:          {base.accuracy() * 100:.1f}% -> {cand.accuracy() * 100:.1f}%"
          f"   (\"+{(cand.accuracy() - base.accuracy()) * 100:.1f} points!\")")
    print(f"  under the hood:    v2 fixed {gained} items and broke {lost}; net +{gained - lost}"
          f" of {len(base.outcomes)}")
    print(f"  the two CIs:       [{b_ci.low * 100:.1f}, {b_ci.high * 100:.1f}] vs "
          f"[{c_ci.low * 100:.1f}, {c_ci.high * 100:.1f}]   overlap: "
          f"{'yes' if bootstrap.intervals_overlap(b_ci, c_ci) else 'no'}")
    print(f"  paired difference: {diff.value * 100:+.1f} points, 95% CI "
          f"[{diff.low * 100:+.1f}, {diff.high * 100:+.1f}]   includes zero: "
          f"{'yes' if not diff.excludes_zero() else 'no'}")
    print(f"  safety slice:      n = {len(base.vector('safety'))}, CI width "
          f"{(base.ci('safety').high - base.ci('safety').low) * 100:.0f} points: "
          "too few items to rank anything")
    print(f"""
  Verdict: on these {diff.n} items, "v2 is {diff.value * 100:.0f} points better" is not
  supported. The data is consistent with anything from a {-diff.low * 100:.0f}-point loss
  to a {diff.high * 100:.0f}-point gain. A leaderboard would still print {cand.accuracy() * 100:.1f}
  above {base.accuracy() * 100:.1f}.""")
    worse = [s for s in ("factual", "arithmetic", "format")
             if cand.accuracy(s) < base.accuracy(s)]
    if worse:
        print(f"""
  And note {', '.join(worse)}: v2 has the higher skill setting there by
  construction, yet it scores lower. Small slices are noisy.""")

    # ------------------------------------------------------------------ 3 --
    section("3. LLM-AS-JUDGE  (position bias, verbosity bias, and the fix)")
    pair = next(p for p in judge.PAIRS if p.id == "j02")
    j = judge.MockJudge()
    print(f"  question: {pair.question}")
    print(f"  concise : \"{pair.concise}\"")
    print(f"            ({len(pair.concise.split())} words, covers "
          f"{pair.points(pair.concise)}/{len(pair.key_points)} key points)")
    print(f"  verbose : \"{pair.verbose[:70]}...\"")
    print(f"            ({len(pair.verbose.split())} words, covers "
          f"{pair.points(pair.verbose)}/{len(pair.key_points)} key points)")
    print(f"  gold verdict by rubric: {pair.gold()}\n")
    v1 = judge.judge_once(j, pair, "concise")
    v2 = judge.judge_once(j, pair, "verbose")
    print(f"  biased judge, concise shown first  -> winner: {v1}")
    print(f"  biased judge, verbose shown first  -> winner: {v2}"
          f"      <- same pair, verdict flipped")
    print(f"  swap only (keep consistent wins)   -> winner: "
          f"{judge.judge_swapped(j, pair)}")
    print(f"  rubric + swap                      -> winner: "
          f"{judge.judge_swapped(j, pair, rubric=True)}\n")
    print(f"  Across all {len(judge.PAIRS)} pairs (6 concise-better, 3 verbose-better, "
          "3 equal):\n")
    print(judge.report_table(j))
    both = judge.evaluate_judge(j, rubric=True, swap=True)
    print(f"""
  Swapping alone stops position from awarding wins, but turns them into
  ties, not right answers. The rubric alone removes the length bonus, but
  position still decides the equal-quality pairs. Together: {both.agreement * 100:.0f}%.""")

    # ------------------------------------------------------------------ 4 --
    section("4. pass@k  (unbiased estimator, Chen et al. 2021)")
    print(pass_at_k.table())
    div = pass_at_k.benchmark(pass_at_k.DIVERSE)
    print(f"""
  Same 10 problems, n = {pass_at_k.N_SAMPLES} samples each, every sample checked by
  a sandboxed verifier. Ranked by pass@1 the focused model wins; ranked by
  pass@10 the diverse one does. "Which model is better?" depends on k.

  The naive estimate 1 - (1 - c/n)^k says the diverse model's pass@10 is
  {div['naive@10'] * 100:.1f}%. The unbiased 1 - C(n-c,k)/C(n,k) says {div['pass@10'] * 100:.1f}%. The naive
  form can only ever come out lower.""")

    # ------------------------------------------------------------------ 5 --
    section("5. CI GATE  (python ci_gate.py --candidate <model>; real exit codes)")
    codes = {}
    for name in ("candidate-v2", "candidate-v3-chatty", "candidate-v3-quiet"):
        proc = subprocess.run([sys.executable, os.path.join(HERE, "ci_gate.py"),
                               "--candidate", name],
                              capture_output=True, text=True, cwd=HERE)
        print(proc.stdout.rstrip())
        print(f"  $ echo $?  ->  {proc.returncode}\n")
        codes[name] = proc.returncode

    # BEFORE/AFTER: a naive gate vs this gate.
    print(BAR)
    print("  BEFORE / AFTER   naive gate (\"fail if accuracy dropped > 3 points\")")
    print("                   vs this gate (paired CI + zero-tolerance safety)")
    print(BAR + "\n")
    print("  release               what changed                 overall   naive gate   this gate")
    print("  " + "-" * 92)
    story = {
        "candidate-v2": "small real improvement",
        "candidate-v3-chatty": "friendlier system prompt",
        "candidate-v3-quiet": "one jailbreak now works",
    }
    for name, what in story.items():
        r = harness.run_eval(models.get(name))
        naive = "BLOCK" if base.accuracy() - r.accuracy() > 0.03 else "pass"
        mine = "BLOCK" if codes[name] else "pass"
        print(f"  {name:<21s} {what:<28s} {r.accuracy() * 100:6.1f}%   {naive:>10s}"
              f"   {mine:>9s} (exit {codes[name]})")
    print(f"""
  The quiet regression scores {harness.run_eval(models.QUIET).accuracy() * 100:.1f}% overall, higher than production,
  and a score-only gate waves it through. It's blocked because safety
  items are checked one by one, not averaged.""")


if __name__ == "__main__":
    main()
