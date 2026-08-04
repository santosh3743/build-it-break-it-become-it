"""
Tests for the training loop -- these are the lab's acceptance criteria.

Every claim the post makes gets an assertion here:
  * the tiny config trains end to end on CPU in seconds-to-minutes
  * the loss goes down
  * a fixed seed reproduces the same final loss exactly
  * checkpoints are hashable and re-loadable
  * generated samples are non-empty
"""

import math
import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data  # noqa: E402
import train  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402

# Building the dataset runs Lab 01's whole pipeline, so do it once.
_DS = None


def dataset():
    global _DS
    if _DS is None:
        _DS = data.build_dataset()
    return _DS


def micro_config(**overrides):
    """A model small enough that the full suite stays fast in CI."""
    cfg = train.TrainConfig(
        name="micro", n_layer=1, n_head=2, n_embd=16, block_size=8,
        batch_size=4, max_steps=40, learning_rate=0.01, warmup_steps=5,
        eval_interval=20, eval_batches=2, sample_every=20, sample_tokens=8, seed=7,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# --------------------------------------------------------------------------- #
# Learning-rate schedule  (the Word of the Week)
# --------------------------------------------------------------------------- #
def test_lr_warms_up_linearly_then_decays_to_the_floor():
    cfg = micro_config(max_steps=100, warmup_steps=10, learning_rate=1.0, min_lr=0.1)

    assert train.lr_at(0, cfg) < train.lr_at(5, cfg) < train.lr_at(10, cfg)
    assert abs(train.lr_at(10, cfg) - 1.0) < 1e-9, "peak LR is reached at warmup end"
    assert abs(train.lr_at(100, cfg) - 0.1) < 1e-9, "decays to min_lr at the end"
    # Cosine decay is monotone after the peak.
    after = [train.lr_at(s, cfg) for s in range(10, 101)]
    assert all(a >= b - 1e-12 for a, b in zip(after, after[1:]))


def test_lr_never_exceeds_the_configured_peak():
    cfg = micro_config(max_steps=50, warmup_steps=7, learning_rate=0.3, min_lr=0.01)
    assert max(train.lr_at(s, cfg) for s in range(60)) <= 0.3 + 1e-12


# --------------------------------------------------------------------------- #
# Gradient clipping
# --------------------------------------------------------------------------- #
def test_clipping_rescales_an_exploding_gradient_to_the_max_norm():
    from autograd import Tensor

    p = Tensor([[0.0, 0.0]])
    p.grad = [[300.0, 400.0]]          # norm = 500
    norm = train.clip_grad_norm([p], max_norm=1.0)
    assert abs(norm - 500.0) < 1e-6, "should report the norm BEFORE clipping"
    clipped = math.sqrt(sum(v * v for v in p.grad[0]))
    assert abs(clipped - 1.0) < 1e-9
    # Direction is preserved -- clipping shortens the step, it does not turn it.
    assert abs(p.grad[0][0] / p.grad[0][1] - 0.75) < 1e-9


def test_clipping_leaves_a_small_gradient_untouched():
    from autograd import Tensor

    p = Tensor([[0.0, 0.0]])
    p.grad = [[0.03, 0.04]]            # norm = 0.05
    train.clip_grad_norm([p], max_norm=1.0)
    assert p.grad[0] == [0.03, 0.04]


# --------------------------------------------------------------------------- #
# The optimizer
# --------------------------------------------------------------------------- #
def test_adamw_walks_a_quadratic_downhill():
    from autograd import Tensor

    p = Tensor([[5.0]])
    opt = train.AdamW([p], weight_decay=0.0)
    for _ in range(200):
        p.zero_grad()
        p.grad[0][0] = 2.0 * p.data[0][0]      # d/dx of x²
        opt.step(lr=0.1)
    assert abs(p.data[0][0]) < 0.1, f"expected to approach 0, got {p.data[0][0]}"


def test_weight_decay_shrinks_weight_matrices_but_leaves_biases_alone():
    """Decoupled decay applies to 2-D weight matrices only. Decaying biases and
    LayerNorm gains is a known way to make a small model worse."""
    from autograd import Tensor

    weight = Tensor([[1.0], [1.0]])   # a (2,1) weight matrix -> should decay
    bias = Tensor([[1.0, 1.0]])       # a (1,2) bias row      -> should not
    opt = train.AdamW([weight, bias], weight_decay=0.1)
    for _ in range(50):
        weight.zero_grad()
        bias.zero_grad()              # no gradient on either
        opt.step(lr=0.1)

    assert weight.data[0][0] < 1.0, "weight matrix should have been decayed"
    assert bias.data[0][0] == 1.0, "bias must not be decayed"


# --------------------------------------------------------------------------- #
# End-to-end training -- the acceptance criteria
# --------------------------------------------------------------------------- #
def test_training_starts_near_the_uniform_prior_loss():
    ds = dataset()
    result = train.train(micro_config(max_steps=2), ds, verbose=False)
    assert abs(result.losses[0] - math.log(ds.vocab_size)) < 0.5, result.losses[0]


def test_training_drives_the_loss_down():
    """The claim the whole post rests on. Compare averages, not raw endpoints,
    because a single mini-batch loss is noisy."""
    result = train.train(micro_config(max_steps=60), dataset(), verbose=False)
    n = len(result.losses)
    first_quarter = sum(result.losses[: n // 4]) / (n // 4)
    last_quarter = sum(result.losses[-(n // 4):]) / (n // 4)
    assert last_quarter < first_quarter, f"{first_quarter:.3f} -> {last_quarter:.3f}"


def test_a_fixed_seed_reproduces_the_final_loss_exactly():
    a = train.train(micro_config(), dataset(), verbose=False)
    b = train.train(micro_config(), dataset(), verbose=False)
    assert a.final_loss == b.final_loss
    assert a.losses == b.losses


def test_a_different_seed_gives_a_different_run():
    a = train.train(micro_config(seed=1), dataset(), verbose=False)
    b = train.train(micro_config(seed=2), dataset(), verbose=False)
    assert a.losses != b.losses


def test_samples_are_non_empty_and_decode_to_text():
    result = train.train(micro_config(), dataset(), verbose=False)
    assert len(result.samples) >= 2, "want samples from several checkpoints"
    for sample in result.samples:
        assert sample.text.strip(), "generated sample was empty"
        assert sample.step >= 0


def test_validation_loss_is_recorded():
    result = train.train(micro_config(), dataset(), verbose=False)
    assert result.val_losses, "no validation loss recorded"
    assert all(v > 0 for _, v in result.val_losses)


# --------------------------------------------------------------------------- #
# Checkpoints -- hashable and re-loadable (the supply-chain control)
# --------------------------------------------------------------------------- #
def test_checkpoint_round_trips_to_identical_weights_and_identical_loss():
    ds = dataset()
    cfg = GPTConfig(vocab_size=ds.vocab_size, block_size=8, n_layer=1, n_head=2, n_embd=16)
    model = GPT(cfg, seed=3)
    ids = ds.train_ids[:8]
    before = model.loss(ids, ds.train_ids[1:9]).data[0][0]

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ckpt.json")
        train.save_checkpoint(path, model, step=5, loss=before)
        restored = train.load_model(path)

    for p, q in zip(model.parameters(), restored.parameters()):
        assert p.data == q.data
    after = restored.loss(ids, ds.train_ids[1:9]).data[0][0]
    assert before == after


def test_checkpoint_hash_is_stable_for_identical_weights_and_changes_when_they_do():
    ds = dataset()
    cfg = GPTConfig(vocab_size=ds.vocab_size, block_size=8, n_layer=1, n_head=2, n_embd=16)

    with tempfile.TemporaryDirectory() as tmp:
        a = os.path.join(tmp, "a.json")
        b = os.path.join(tmp, "b.json")
        c = os.path.join(tmp, "c.json")
        h_a = train.save_checkpoint(a, GPT(cfg, seed=3), step=1, loss=1.0)
        h_b = train.save_checkpoint(b, GPT(cfg, seed=3), step=1, loss=1.0)
        h_c = train.save_checkpoint(c, GPT(cfg, seed=4), step=1, loss=1.0)

        assert h_a == h_b, "same weights must hash the same"
        assert h_a != h_c, "different weights must hash differently"
        assert len(h_a) == 64
        assert train.checkpoint_hash(a) == h_a, "hash must be recomputable from the file"


def test_tampering_with_a_checkpoint_changes_its_hash():
    """The Lab 11 forward-reference: this is how you notice a swapped model."""
    import json

    ds = dataset()
    cfg = GPTConfig(vocab_size=ds.vocab_size, block_size=8, n_layer=1, n_head=2, n_embd=16)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ckpt.json")
        original = train.save_checkpoint(path, GPT(cfg, seed=3), step=1, loss=1.0)

        with open(path) as fh:
            payload = json.load(fh)
        payload["params"][0][0][0] += 0.0001      # one weight, barely moved
        with open(path, "w") as fh:
            json.dump(payload, fh)

        assert train.checkpoint_hash(path) != original


def test_training_writes_checkpoints_that_can_be_reloaded():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = micro_config(out_dir=tmp)
        result = train.train(cfg, dataset(), verbose=False)
        assert result.checkpoints, "no checkpoints written"
        for ckpt in result.checkpoints:
            assert os.path.exists(ckpt.path)
            assert train.checkpoint_hash(ckpt.path) == ckpt.sha256
            train.load_model(ckpt.path)          # must not raise


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
            passed += 1
    print(f"\n{passed} passed")
