"""
Tests for the GPT itself.

The two that matter most:
  * initial loss must be ~ln(vocab_size) -- proves the model starts out
    honestly uncertain rather than accidentally confident.
  * causality -- position t's logits must not move when you change token t+1.
    A model that fails this is cheating, and its loss curve is a lie.
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import GPT, GPTConfig  # noqa: E402

CFG = GPTConfig(vocab_size=17, block_size=8, n_layer=2, n_head=2, n_embd=8)


def test_forward_returns_logits_per_position():
    model = GPT(CFG, seed=0)
    logits = model.forward([1, 2, 3, 4])
    assert (logits.rows, logits.cols) == (4, CFG.vocab_size)


def test_initial_loss_is_close_to_log_vocab_size():
    """An untrained model should be maximally unsure: loss ~ ln(V)."""
    model = GPT(CFG, seed=0)
    loss = model.loss([1, 2, 3, 4, 5], [2, 3, 4, 5, 6]).data[0][0]
    assert abs(loss - math.log(CFG.vocab_size)) < 0.25, loss


def test_model_is_causal():
    """Changing a later token must not change any earlier position's logits."""
    model = GPT(CFG, seed=0)
    a = model.forward([1, 2, 3, 4])
    b = model.forward([1, 2, 3, 9])  # only the LAST token differs

    for pos in range(3):  # positions 0..2 cannot see position 3
        for j in range(CFG.vocab_size):
            assert abs(a.data[pos][j] - b.data[pos][j]) < 1e-9, f"leak at position {pos}"

    # ...and the last position SHOULD differ, or the test proves nothing.
    assert any(abs(a.data[3][j] - b.data[3][j]) > 1e-9 for j in range(CFG.vocab_size))


def test_same_seed_gives_identical_parameters():
    a = GPT(CFG, seed=42)
    b = GPT(CFG, seed=42)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert pa.data == pb.data


def test_different_seed_gives_different_parameters():
    a = GPT(CFG, seed=1)
    b = GPT(CFG, seed=2)
    assert any(pa.data != pb.data for pa, pb in zip(a.parameters(), b.parameters()))


def test_backward_populates_gradients_on_every_weight_matrix():
    model = GPT(CFG, seed=0)
    model.zero_grad()
    model.loss([1, 2, 3, 4], [2, 3, 4, 5]).backward()

    for i, p in enumerate(model.parameters()):
        # The token embedding table only receives gradient on the rows it used,
        # so we check "some entry is non-zero" rather than "all are".
        nonzero = any(abs(v) > 0 for row in p.grad for v in row)
        assert nonzero, f"parameter {i} ({p.rows}x{p.cols}) received no gradient"


def test_generate_produces_the_requested_number_of_in_vocab_tokens():
    import random

    model = GPT(CFG, seed=0)
    out = model.generate([1, 2], max_new_tokens=6, rng=random.Random(0))
    assert len(out) == 8, "prompt + new tokens"
    assert all(0 <= t < CFG.vocab_size for t in out)


def test_generate_respects_the_context_window():
    """Generating past block_size must crop context, not crash."""
    import random

    model = GPT(CFG, seed=0)
    out = model.generate([1, 2, 3], max_new_tokens=12, rng=random.Random(0))
    assert len(out) == 15


def test_generate_is_deterministic_for_a_fixed_rng_seed():
    import random

    model = GPT(CFG, seed=0)
    a = model.generate([1, 2], max_new_tokens=6, rng=random.Random(7))
    b = model.generate([1, 2], max_new_tokens=6, rng=random.Random(7))
    assert a == b


def test_greedy_generation_ignores_the_rng():
    import random

    model = GPT(CFG, seed=0)
    a = model.generate([1, 2], max_new_tokens=5, rng=random.Random(1), temperature=0.0)
    b = model.generate([1, 2], max_new_tokens=5, rng=random.Random(999), temperature=0.0)
    assert a == b


def test_num_params_matches_the_hand_computed_formula():
    c = CFG
    per_block = (
        4 * (c.n_embd * c.n_embd + c.n_embd)          # q, k, v, out projections + biases
        + (c.n_embd * 4 * c.n_embd + 4 * c.n_embd)    # mlp fc + bias
        + (4 * c.n_embd * c.n_embd + c.n_embd)        # mlp proj + bias
        + 4 * c.n_embd                                # two layernorm gain+bias pairs
    )
    expected = (
        c.vocab_size * c.n_embd        # token embeddings (tied to the output head)
        + c.block_size * c.n_embd      # position embeddings
        + c.n_layer * per_block
        + 2 * c.n_embd                 # final layernorm gain+bias
    )
    assert GPT(CFG, seed=0).num_params() == expected


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
            passed += 1
    print(f"\n{passed} passed")
