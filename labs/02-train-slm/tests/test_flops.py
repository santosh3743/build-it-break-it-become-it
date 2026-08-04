"""
Tests for the FLOPs accounting (the post's Deeper Dive artifact).

The numbers must be derivable by hand, or the "compute your own training run"
exercise is just trusting a function.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import flops  # noqa: E402


def test_flops_per_token_matches_the_hand_derived_formula():
    """6N for the weights (2 forward + 4 backward), plus the attention term
    that the 6N approximation leaves out."""
    n_params, n_layer, n_embd, block = 100_000, 3, 48, 32
    expected = 6 * n_params + 12 * n_layer * n_embd * block
    assert flops.flops_per_token(n_params, n_layer, n_embd, block) == expected


def test_forward_only_is_one_third_of_forward_plus_backward():
    """Backward costs about twice forward: 2N forward, 4N backward."""
    fwd = flops.flops_per_token(100_000, 3, 48, 32, backward=False)
    both = flops.flops_per_token(100_000, 3, 48, 32, backward=True)
    assert abs(both / fwd - 3.0) < 1e-9


def test_total_training_flops_scales_with_steps_batch_and_context():
    base = flops.training_flops(n_params=100_000, n_layer=3, n_embd=48,
                                block_size=32, batch_size=8, steps=100)
    assert flops.training_flops(100_000, 3, 48, 32, 8, 200) == 2 * base
    assert flops.training_flops(100_000, 3, 48, 32, 16, 100) == 2 * base


def test_total_equals_flops_per_token_times_tokens_processed():
    args = dict(n_params=100_000, n_layer=3, n_embd=48, block_size=32,
                batch_size=8, steps=100)
    tokens = args["block_size"] * args["batch_size"] * args["steps"]
    per_token = flops.flops_per_token(args["n_params"], args["n_layer"],
                                      args["n_embd"], args["block_size"])
    assert flops.training_flops(**args) == per_token * tokens


def test_chinchilla_optimum_is_twenty_tokens_per_parameter():
    assert flops.chinchilla_optimal_tokens(1_000_000) == 20_000_000


def test_tokens_per_param_reports_how_far_off_chinchilla_a_run_is():
    ratio = flops.tokens_per_param(n_params=100_000, tokens_seen=2_000_000)
    assert abs(ratio - 20.0) < 1e-9


def test_human_readable_formats_large_numbers():
    assert flops.human(1.5e12) == "1.50 TFLOPs"
    assert flops.human(2.0e15) == "2.00 PFLOPs"
    assert flops.human(3.0e9) == "3.00 GFLOPs"


def test_report_mentions_the_total_and_the_chinchilla_comparison():
    text = flops.report(n_params=100_000, n_layer=3, n_embd=48, block_size=32,
                        batch_size=8, steps=100)
    assert "FLOP" in text
    assert "Chinchilla" in text


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
            passed += 1
    print(f"\n{passed} passed")
