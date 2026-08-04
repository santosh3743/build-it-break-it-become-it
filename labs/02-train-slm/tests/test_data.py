"""
Tests for the Lab 01 -> Lab 02 handoff.

Lab 02 does not own a tokenizer or a cleaning step. It imports Lab 01's pipeline
and trains on the shards that come out. These tests assert that the link is real
-- that we are genuinely running Lab 01's code, and that its controls fired.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data  # noqa: E402


def test_lab01_pipeline_is_importable_and_is_the_real_thing():
    lab01 = data.load_lab01()
    for symbol in ("run_pipeline", "Tokenizer", "near_dedup", "scrub_pii"):
        assert hasattr(lab01, symbol), f"Lab 01 is missing {symbol}"


def test_dataset_is_big_enough_to_train_on():
    ds = data.build_dataset()
    assert len(ds.train_ids) >= 4000, f"only {len(ds.train_ids)} train tokens"
    assert len(ds.val_ids) >= 200


def test_every_token_id_is_inside_the_vocabulary():
    ds = data.build_dataset()
    assert max(ds.train_ids + ds.val_ids) < ds.vocab_size


def test_shards_respect_the_shard_size_and_concatenate_to_the_token_stream():
    ds = data.build_dataset(shard_size=256)
    assert all(len(s) <= 256 for s in ds.shards)
    flat = [t for shard in ds.shards for t in shard]
    assert flat == ds.train_ids + ds.val_ids


def test_lab01_controls_actually_fired_on_this_corpus():
    """If none of Lab 01's stages remove anything, the handoff is decorative."""
    ds = data.build_dataset()
    r = ds.report
    assert r.duplicates_removed > 0, "near-dedup removed nothing"
    assert r.quality_removed > 0, "quality filter removed nothing"
    assert r.contaminated_removed > 0, "decontamination removed nothing"
    assert r.pii_redactions > 0, "PII scrubber redacted nothing"


def test_the_eval_text_did_not_survive_into_the_training_tokens():
    """The whole point of decontamination: the eval string must not be trainable."""
    ds = data.build_dataset()
    from sample_corpus import EVAL_TEXTS

    needle = ds.tokenizer.encode(EVAL_TEXTS[0])[:12]
    haystack = ds.train_ids
    found = any(haystack[i:i + len(needle)] == needle for i in range(len(haystack)))
    assert not found, "eval text leaked into the training set"


def test_train_and_val_do_not_overlap():
    ds = data.build_dataset()
    assert len(ds.train_ids) + len(ds.val_ids) == len(ds.all_ids)


def test_batch_has_the_requested_shape_and_targets_are_inputs_shifted_by_one():
    import random

    ds = data.build_dataset()
    batch = data.get_batch(ds.train_ids, block_size=16, batch_size=4, rng=random.Random(0))
    assert len(batch) == 4
    for x, y in batch:
        assert len(x) == 16 and len(y) == 16
        assert x[1:] == y[:-1], "y must be x shifted one token to the left"


def test_batching_is_deterministic_for_a_fixed_seed():
    import random

    ds = data.build_dataset()
    a = data.get_batch(ds.train_ids, 16, 4, random.Random(3))
    b = data.get_batch(ds.train_ids, 16, 4, random.Random(3))
    assert a == b


def test_building_the_dataset_twice_gives_identical_tokens():
    """Reproducible data is a precondition for a reproducible training run."""
    assert data.build_dataset().train_ids == data.build_dataset().train_ids


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
            passed += 1
    print(f"\n{passed} passed")
