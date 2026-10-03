"""
Lab 03 acceptance tests.

From the spec: SFT then DPO run end-to-end on the CPU path; the three-way
comparison renders; DPO output measurably shifts toward the preferred style on
the toy metric; the pipeline stages run and the comparison table is populated.

The full pipeline takes ~40s, so it is built once and shared.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dpo_loss  # noqa: E402
import evaluate  # noqa: E402
import pipeline  # noqa: E402
import post_training as pt  # noqa: E402
import preference_data as pd  # noqa: E402

_RESULT = None


def result():
    global _RESULT
    if _RESULT is None:
        _RESULT = pipeline.run_post_training()
    return _RESULT


# --------------------------------------------------------------------------- #
# The DPO maths (Deeper Dive artifact) -- checkable by hand
# --------------------------------------------------------------------------- #
def test_dpo_loss_is_ln2_when_policy_equals_reference():
    """Every DPO run starts here: no preference expressed yet."""
    import math

    loss = dpo_loss.dpo_loss(-10.0, -10.0, -10.0, -10.0, beta=0.1)
    assert abs(loss - math.log(2)) < 1e-9


def test_dpo_loss_falls_when_the_policy_prefers_the_chosen_response():
    indifferent = dpo_loss.dpo_loss(-10.0, -10.0, -10.0, -10.0)
    preferring = dpo_loss.dpo_loss(-5.0, -15.0, -10.0, -10.0)
    backwards = dpo_loss.dpo_loss(-15.0, -5.0, -10.0, -10.0)
    assert preferring < indifferent < backwards


def test_gradient_weight_is_largest_where_the_policy_is_most_wrong():
    """σ(-margin) is an automatic difficulty weight -- the point of the Dive."""
    already_right = dpo_loss.dpo_grad_weight(-5.0, -15.0, -10.0, -10.0)
    wrong = dpo_loss.dpo_grad_weight(-15.0, -5.0, -10.0, -10.0)
    assert wrong > already_right


def test_subtracting_the_reference_cancels_a_preference_it_already_had():
    """If the reference preferred chosen just as much, the margin is zero."""
    margin = dpo_loss.dpo_margin(-5.0, -15.0, -5.0, -15.0)
    assert abs(margin) < 1e-12


def test_grpo_advantages_are_zero_mean_and_reward_the_correct_answers():
    adv = dpo_loss.grpo_advantages([1.0, 0.0, 1.0, 0.0])
    assert abs(sum(adv)) < 1e-9
    assert adv[0] > 0 > adv[1]


def test_grpo_gives_no_signal_when_every_answer_scores_the_same():
    assert dpo_loss.grpo_advantages([1.0, 1.0, 1.0]) == [0.0, 0.0, 0.0]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def test_sft_data_contains_both_styles_so_dpo_has_something_to_do():
    texts = pd.sft_texts()
    assert any(m in t for t in texts for m in pd.HEDGE_MARKERS)
    assert any(m in t for t in texts for m in pd.ABSOLUTE_MARKERS)


def test_preference_pairs_are_ranked_the_way_the_metric_scores_them():
    for _, chosen, rejected in pd.preference_pairs():
        assert evaluate.style_score(chosen) > evaluate.style_score(rejected)


def test_context_window_fits_the_longest_training_sequence():
    """The bug that silently ruins this lab: truncated responses mean the model
    never sees an answer end."""
    corpus = pt.build_corpus()
    longest = max(len(s) for s in corpus.sft_sequences)
    assert pipeline.PipelineConfig().block_size >= longest - 1, (
        f"block_size too small: longest SFT sequence is {longest} tokens"
    )


def test_corpus_is_deterministic():
    assert pt.build_corpus().sft_sequences == pt.build_corpus().sft_sequences


# --------------------------------------------------------------------------- #
# The three stages actually run
# --------------------------------------------------------------------------- #
def test_pretraining_reduces_loss():
    r = result()
    assert r.pretrain_losses[-1] < r.pretrain_losses[0]


def test_sft_reduces_loss():
    r = result()
    assert r.sft_losses[-1] < r.sft_losses[0]


def test_dpo_starts_at_ln2_and_reduces_loss():
    import math

    r = result()
    assert abs(r.dpo_losses[0] - math.log(2)) < 0.6, "DPO must start ~indifferent"
    assert r.dpo_losses[-1] < r.dpo_losses[0]


def test_dpo_drives_the_implicit_reward_margin_positive():
    r = result()
    assert r.dpo_margins[0] < r.dpo_margins[-1]
    assert r.dpo_margins[-1] > 0


def test_dpo_preference_accuracy_ends_high():
    r = result()
    assert r.dpo_accuracies[-1] >= 0.75


# --------------------------------------------------------------------------- #
# The acceptance claim: DPO shifts style, and the table renders
# --------------------------------------------------------------------------- #
def test_dpo_measurably_shifts_output_toward_the_preferred_style():
    r = result()
    assert r.by_name("DPO").mean_style > r.by_name("SFT").mean_style, (
        f"DPO {r.by_name('DPO').mean_style:.2f} did not beat "
        f"SFT {r.by_name('SFT').mean_style:.2f}"
    )


def test_dpo_beats_sft_on_the_pairwise_win_rate():
    r = result()
    wins, losses, _ = evaluate.win_rate(r.by_name("DPO"), r.by_name("SFT"))
    assert wins > losses


def test_dpo_stays_fluent_rather_than_reward_hacking_into_repetition():
    """The NLL regularizer's job. Without it this test fails loudly."""
    for response in result().by_name("DPO").responses:
        words = response.split()
        if len(words) < 6:
            continue
        most_common = max(set(words), key=words.count)
        assert words.count(most_common) / len(words) < 0.5, (
            f"degenerate repetition in: {response!r}"
        )


def test_every_model_produces_non_empty_responses():
    for model in result().scored:
        assert len(model.responses) == len(pd.EVAL_INSTRUCTIONS)
        for response in model.responses:
            assert response.strip()


def test_comparison_table_is_populated():
    table = evaluate.win_rate_table(result().scored)
    for name in ("base", "SFT", "DPO"):
        assert name in table
    assert table.count("\n") >= 4, "expected a row per matchup"


def test_scorecard_renders_all_three_models():
    card = evaluate.scorecard(result().scored)
    assert "base" in card and "SFT" in card and "DPO" in card


# --------------------------------------------------------------------------- #
# Scoring helpers
# --------------------------------------------------------------------------- #
def test_style_score_counts_hedges_up_and_absolutes_down():
    assert evaluate.style_score("based on the corpus roughly") > 0
    assert evaluate.style_score("definitely always guaranteed") < 0
    assert evaluate.style_score("the batch size") == 0


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
            passed += 1
    print(f"\n{passed} passed")
