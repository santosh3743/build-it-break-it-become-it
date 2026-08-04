"""
Test suite for Lab 01. Run with:  python -m pytest -q   (or: python tests/test_pipeline.py)
Pure stdlib + pytest. Every control is asserted to do its one job.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import (  # noqa: E402
    MinHasher,
    Tokenizer,
    contamination_ratio,
    build_eval_index,
    near_dedup,
    normalize,
    quality_ok,
    run_pipeline,
    scrub_pii,
    shingles,
)
from sample_corpus import EVAL_TEXTS, RAW_TEXTS  # noqa: E402


def test_minhash_estimates_jaccard():
    """Estimated Jaccard from signatures should track the true Jaccard."""
    a = shingles("the quick brown fox jumps over the lazy dog by the river bank")
    b = shingles("the quick brown fox leaps over the lazy dog by the river bank")
    true_j = len(a & b) / len(a | b)
    h = MinHasher(num_perm=128, seed=7)
    est = MinHasher.estimated_jaccard(h.signature(a), h.signature(b))
    assert abs(est - true_j) < 0.15, (est, true_j)


def test_near_dedup_catches_near_duplicate():
    """Doc 2 is a near-dup of doc 1; exact hashing would miss it, MinHash must not."""
    docs = [normalize(t) for t in RAW_TEXTS]
    assert hash(docs[1]) != hash(docs[2])  # not byte-identical
    kept, removed = near_dedup(docs)
    # Exactly one of the {1,2} near-dup pair should be removed.
    assert len({1, 2} & set(removed)) == 1
    # Distinct clean docs must be kept.
    assert 0 in kept and 7 in kept


def test_quality_filter_rejects_junk():
    ok_short, reason = quality_ok("too short")
    assert not ok_short and reason == "too_short"
    ok_rep, reason = quality_ok(" ".join(["data"] * 40))
    assert not ok_rep and reason == "repetitive"
    ok_good, _ = quality_ok(RAW_TEXTS[0])
    assert ok_good


def test_decontamination_flags_eval_leak():
    idx = build_eval_index(EVAL_TEXTS)
    leaked = normalize(RAW_TEXTS[5])
    clean = normalize(RAW_TEXTS[0])
    assert contamination_ratio(leaked, idx) >= 0.30
    assert contamination_ratio(clean, idx) < 0.30


def test_pii_scrub_redacts_but_preserves_length_ish():
    scrubbed, n = scrub_pii(RAW_TEXTS[6])
    assert n >= 2  # at least the email and the phone number
    assert "@example.com" not in scrubbed
    assert "[EMAIL]" in scrubbed


def test_tokenizer_round_trips_known_tokens():
    tok = Tokenizer().build_vocab([RAW_TEXTS[0]])
    ids = tok.encode("the modern data pipeline")
    assert tok.decode(ids) == "the modern data pipeline"
    assert tok.vocab_size > 2


def test_full_pipeline_end_to_end():
    shards, tok, rep = run_pipeline(RAW_TEXTS, EVAL_TEXTS)
    assert rep.docs_in == len(RAW_TEXTS)
    assert rep.duplicates_removed == 1          # the one near-dup
    assert rep.quality_removed == 2             # the two junk docs
    assert rep.contaminated_removed == 1        # the eval leak
    assert rep.pii_redactions >= 2              # email + phone
    assert rep.docs_out == 4                    # 8 in - 1 dup - 2 junk - 1 contam
    assert rep.tokens_out > 0
    assert rep.shards == len(rep.shard_hashes)
    # Datasheet must be JSON-serializable and complete.
    ds = rep.datasheet()
    assert set(["docs_in", "docs_out", "shard_hashes"]).issubset(ds)


if __name__ == "__main__":
    # Allow running without pytest installed.
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failures}/{len(fns)} tests passed")
    sys.exit(1 if failures else 0)
