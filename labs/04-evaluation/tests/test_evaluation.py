"""
Lab 04 acceptance tests.

From the spec: the scorecard renders with CIs; position-swap changes a biased
judge's verdict (the bias), then the mitigation removes the inconsistency; the
CI gate returns non-zero on a seeded regression and zero on a clean run. Plus
the claims the README makes: pass@k matches hand-checked values, the golden set
grades what it says it grades, and the headline "+5 points" isn't significant.

Runs with `python -m pytest -q` or `python tests/test_evaluation.py`.
"""

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
from math import comb

LAB_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, LAB_DIR)

import bootstrap  # noqa: E402
import ci_gate  # noqa: E402
import golden_set  # noqa: E402
import harness  # noqa: E402
import judge  # noqa: E402
import models  # noqa: E402
import pass_at_k as pk  # noqa: E402
import scorers  # noqa: E402

_CACHE = {}


def result(name):
    if name not in _CACHE:
        _CACHE[name] = harness.run_eval(models.get(name))
    return _CACHE[name]


def run_gate(*args):
    return subprocess.run([sys.executable, os.path.join(LAB_DIR, "ci_gate.py"), *args],
                          capture_output=True, text=True, cwd=LAB_DIR)


# --------------------------------------------------------------------------- #
# The golden set grades what it claims to grade
# --------------------------------------------------------------------------- #
def test_golden_set_size_slices_and_unique_ids():
    items = golden_set.GOLDEN_SET
    assert 40 <= len(items) <= 80
    assert {it.slice for it in items} == set(golden_set.SLICES)
    assert len({it.id for it in items}) == len(items)
    assert sum(it.slice == "safety" for it in items) >= 5


def test_every_reference_passes_and_every_distractor_fails():
    """If this fails, the exam is broken, and every score built on it is wrong."""
    for it in golden_set.GOLDEN_SET:
        assert scorers.score(it, it.reference), f"{it.id}: reference fails its own check"
        assert not scorers.score(it, it.distractor), f"{it.id}: distractor passes"


def test_contains_matches_whole_words_only():
    au = next(it for it in golden_set.GOLDEN_SET if it.reference == "Au")
    assert scorers.score(au, "The symbol is Au.")
    assert not scorers.score(au, "Australia")


def test_chatty_wrapper_breaks_format_but_not_content():
    chatty = models.CHATTY.template
    fact = next(it for it in golden_set.GOLDEN_SET if it.slice == "factual")
    fmt = next(it for it in golden_set.GOLDEN_SET if it.check == "json")
    assert scorers.score(fact, chatty.format(answer=fact.reference))
    assert not scorers.score(fmt, chatty.format(answer=fmt.reference))


def test_safety_items_fail_when_the_canary_leaks():
    for it in golden_set.GOLDEN_SET:
        if it.slice == "safety":
            assert not scorers.score(it, f"ok: {it.args['forbidden'][0]}")


# --------------------------------------------------------------------------- #
# Bootstrap confidence intervals (Deeper Dive)
# --------------------------------------------------------------------------- #
def test_bootstrap_is_deterministic_and_brackets_the_estimate():
    v = result("baseline-v1").vector()
    a, b = bootstrap.bootstrap_ci(v), bootstrap.bootstrap_ci(v)
    assert a == b
    assert a.low < a.value < a.high
    assert 0.0 <= a.low and a.high <= 1.0


def test_bootstrap_degenerate_and_small_n_cases():
    perfect = bootstrap.bootstrap_ci([1] * 30)
    assert perfect.low == perfect.high == 1.0
    small = bootstrap.bootstrap_ci([1, 0] * 4)       # n = 8
    large = bootstrap.bootstrap_ci([1, 0] * 100)     # n = 200
    assert small.half_width > large.half_width


def test_paired_bootstrap_of_a_model_against_itself_is_exactly_zero():
    v = result("candidate-v2").vector()
    d = bootstrap.paired_bootstrap(v, v)
    assert d.value == d.low == d.high == 0.0


# --------------------------------------------------------------------------- #
# Scorecard renders with CIs (acceptance 1)
# --------------------------------------------------------------------------- #
def test_scorecard_renders_with_cis_for_every_slice_and_overall():
    card = harness.scorecard([result("baseline-v1"), result("candidate-v2")])
    assert "baseline-v1" in card and "candidate-v2" in card
    for label in golden_set.SLICES + ["OVERALL"]:
        row = next(line for line in card.splitlines() if line.strip().startswith(label))
        assert row.count("[") == 2 and row.count("]") == 2, row
        assert "%" in row
    assert "95% bootstrap CI" in card


def test_headline_improvement_is_not_significant():
    """The 'single number misleads' claim: +5 points, overlapping CIs, CI of diff spans 0."""
    b, c = result("baseline-v1"), result("candidate-v2")
    assert c.accuracy() > b.accuracy()
    assert bootstrap.intervals_overlap(b.ci(), c.ci())
    diff = bootstrap.paired_bootstrap(c.vector(), b.vector())
    assert diff.low < 0 < diff.high
    assert not diff.excludes_zero()


# --------------------------------------------------------------------------- #
# LLM-as-judge: the bias, then the mitigation (acceptance 2)
# --------------------------------------------------------------------------- #
def test_position_swap_changes_the_biased_judges_verdict():
    j = judge.MockJudge()
    pair = next(p for p in judge.PAIRS if p.id == "j02")
    first = judge.judge_once(j, pair, "concise")
    second = judge.judge_once(j, pair, "verbose")
    assert first != second
    assert first == "concise" and second == "verbose"   # slot A wins both times


def test_biased_judge_always_picks_slot_a_on_equal_quality_pairs():
    j = judge.MockJudge()
    for p in judge.PAIRS:
        if p.gold() == "tie":
            assert judge.judge_once(j, p, "concise") == "concise"
            assert judge.judge_once(j, p, "verbose") == "verbose"


def test_raw_judge_is_inconsistent_and_rewards_verbosity():
    raw = judge.evaluate_judge(judge.MockJudge(), rubric=False, swap=False)
    assert raw.flip_rate > 0.5
    assert raw.position_decided > 0
    assert raw.verbose_wrong > 0
    assert raw.agreement < 0.6


def test_swap_removes_every_position_decided_win():
    for rubric in (False, True):
        r = judge.evaluate_judge(judge.MockJudge(), rubric=rubric, swap=True)
        assert r.position_decided == 0


def test_rubric_plus_swap_matches_gold_on_every_pair():
    j = judge.MockJudge()
    for p in judge.PAIRS:
        assert judge.judge_swapped(j, p, rubric=True) == p.gold(), p.id
    both = judge.evaluate_judge(j, rubric=True, swap=True)
    assert both.agreement == 1.0 and both.verbose_wrong == 0


def test_each_mitigation_alone_is_not_enough():
    j = judge.MockJudge()
    swap_only = judge.evaluate_judge(j, rubric=False, swap=True)
    rubric_only = judge.evaluate_judge(j, rubric=True, swap=False)
    assert swap_only.agreement < 1.0
    assert rubric_only.position_decided > 0


def test_judge_pairs_cover_all_three_gold_outcomes():
    golds = [p.gold() for p in judge.PAIRS]
    assert {"concise", "verbose", "tie"} == set(golds)


# --------------------------------------------------------------------------- #
# pass@k matches hand-checked values
# --------------------------------------------------------------------------- #
def test_pass_at_k_hand_checked_values():
    assert abs(pk.pass_at_k(10, 3, 1) - 0.3) < 1e-12               # k=1 is just c/n
    assert abs(pk.pass_at_k(10, 3, 5) - (1 - 21 / 252)) < 1e-12     # C(7,5)=21, C(10,5)=252
    assert abs(pk.pass_at_k(20, 1, 10) - 0.5) < 1e-12               # C(19,10)/C(20,10) = 1/2
    assert abs(pk.pass_at_k(4, 2, 2) - 5 / 6) < 1e-12               # 1 - C(2,2)/C(4,2)
    assert pk.pass_at_k(5, 0, 3) == 0.0
    assert pk.pass_at_k(5, 5, 1) == 1.0
    assert pk.pass_at_k(10, 8, 3) == 1.0                            # n - c < k


def test_product_form_equals_binomial_form():
    for n in range(1, 25):
        for c in range(0, n + 1):
            for k in range(1, n + 1):
                assert abs(pk.pass_at_k(n, c, k) - pk.pass_at_k_product(n, c, k)) < 1e-9


def test_naive_estimator_never_exceeds_the_unbiased_one():
    for n in (5, 10, 20):
        for c in range(n + 1):
            for k in range(1, n + 1):
                assert pk.naive_pass_at_k(n, c, k) <= pk.pass_at_k(n, c, k) + 1e-12
    assert pk.naive_pass_at_k(20, 2, 10) < pk.pass_at_k(20, 2, 10)


def test_pass_at_k_rejects_bad_arguments():
    for args in ((5, 6, 1), (5, 2, 0), (5, 2, 6)):
        try:
            pk.pass_at_k(*args)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid {args}")


def test_model_ranking_flips_with_k():
    focused, diverse = pk.benchmark(pk.FOCUSED), pk.benchmark(pk.DIVERSE)
    assert focused["pass@1"] > diverse["pass@1"]
    assert diverse["pass@10"] > focused["pass@10"]


def test_verifier_accepts_correct_forms_and_rejects_bugs():
    for p in pk.PROBLEMS:
        for form in p.correct_forms + (p.reference,):
            assert pk.verify(p, form), (p.id, form)
        for form in p.wrong_forms:
            assert not pk.verify(p, form), (p.id, form)


def test_verifier_is_a_sandbox():
    for evil in ("__import__('os').system('echo hi')", "open('x')", "x + 1",
                 "2**10000", "1/0", "(1).__class__"):
        try:
            pk.safe_eval(evil)
        except ValueError:
            continue
        raise AssertionError(f"safe_eval accepted {evil!r}")


# --------------------------------------------------------------------------- #
# The CI gate (acceptance 3) — real process exit codes
# --------------------------------------------------------------------------- #
def test_stored_baseline_is_current():
    data = ci_gate.load_baseline()
    assert data["golden_set_fingerprint"] == golden_set.fingerprint()
    assert data["outcomes"] == result("baseline-v1").outcomes


def test_gate_exits_zero_on_clean_run():
    proc = run_gate("--candidate", "candidate-v2")
    assert proc.returncode == 0, proc.stdout
    assert "PASS" in proc.stdout


def test_gate_exits_nonzero_on_seeded_format_regression():
    proc = run_gate("--candidate", "candidate-v3-chatty")
    assert proc.returncode == 1, proc.stdout
    assert "BLOCK" in proc.stdout


def test_gate_exits_nonzero_on_quiet_safety_regression():
    proc = run_gate("--candidate", "candidate-v3-quiet")
    assert proc.returncode == 1, proc.stdout
    assert "safety-02" in proc.stdout


def test_gate_errors_on_unknown_model_or_changed_golden_set():
    assert run_gate("--candidate", "no-such-model").returncode == 2
    data = ci_gate.load_baseline()
    data["golden_set_fingerprint"] = "0" * 16
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "stale.json")
        with open(path, "w") as f:
            json.dump(data, f)
        assert run_gate("--candidate", "candidate-v2", "--baseline", path).returncode == 2


def test_gate_rules_fire_for_the_right_reasons():
    base = ci_gate.load_baseline()
    chatty = ci_gate.evaluate_gate(result("candidate-v3-chatty"), base)
    assert chatty.exit_code == 1 and chatty.diff.high < 0 and not chatty.safety_regressions
    quiet = ci_gate.evaluate_gate(result("candidate-v3-quiet"), base)
    assert quiet.exit_code == 1 and quiet.diff.value > 0
    assert quiet.safety_regressions == ["safety-02"]
    same = ci_gate.evaluate_gate(result("baseline-v1"), base)
    assert same.exit_code == 0 and not same.warnings


def test_a_score_only_gate_misses_the_quiet_regression():
    """The BEFORE in the demo: overall accuracy alone can't see it."""
    base = result("baseline-v1").accuracy()
    assert result("candidate-v3-quiet").accuracy() >= base


def test_gate_warns_but_passes_when_a_drop_is_not_significant():
    base = ci_gate.load_baseline()
    # Take the baseline and break two non-safety items: a small, noisy drop.
    r = result("baseline-v1")
    broken = dict(r.outcomes)
    flipped = [k for k, v in broken.items() if v == 1 and r.slice_of[k] != "safety"][:2]
    for k in flipped:
        broken[k] = 0
    noisy = harness.EvalResult("noisy", broken, r.outputs, r.slice_of)
    res = ci_gate.evaluate_gate(noisy, base, threshold=0.03)
    assert res.exit_code == 0 and res.warnings and res.diff.value < 0


def test_update_baseline_round_trips():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "b.json")
        assert run_gate("--update-baseline", "--model", "baseline-v1",
                        "--baseline", path).returncode == 0
        with open(path) as f:
            assert json.load(f) == ci_gate.load_baseline()


# --------------------------------------------------------------------------- #
# The demo itself
# --------------------------------------------------------------------------- #
def test_run_demo_prints_every_section_and_the_gate_exit_codes():
    import run_demo

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run_demo.main()
    out = buf.getvalue()
    for marker in ("1. SCORECARD", "2. IS candidate-v2 BETTER", "3. LLM-AS-JUDGE",
                   "4. pass@k", "5. CI GATE", "BEFORE / AFTER",
                   "verdict flipped", "includes zero: yes"):
        assert marker in out, marker
    assert "$ echo $?  ->  0" in out and out.count("$ echo $?  ->  1") == 2


if __name__ == "__main__":
    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
                passed += 1
            except Exception as exc:  # noqa: BLE001
                print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
                failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
