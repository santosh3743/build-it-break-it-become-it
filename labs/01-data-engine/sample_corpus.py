"""
A tiny hand-built corpus that exercises every stage of the pipeline.

It deliberately contains:
  - clean reference-quality documents (should survive)
  - a near-duplicate of one of them (should be caught by MinHash, not exact hash)
  - low-quality junk (should be quality-filtered)
  - a document that leaks an eval question (should be decontaminated)
  - a document containing PII (should be scrubbed, not dropped)
"""

# The held-out evaluation set. Nothing overlapping these should reach training.
EVAL_TEXTS = [
    "What is the capital of France and which river runs through it? "
    "The capital of France is Paris and the Seine river runs through it.",
]

RAW_TEXTS = [
    # 0 — clean reference doc
    "The modern data pipeline for language models is a sequence of controls. "
    "Each stage removes one nameable class of failure, from redundant documents "
    "to leaked evaluation data. Engineers who skip a stage do not save time; they "
    "defer a bug to a place where it is far more expensive to find and fix later.",

    # 1 — clean reference doc
    "Near-duplicate detection separates people who have built a data pipeline from "
    "people who have only read about one. Exact-duplicate removal is trivial, but the "
    "valuable problem is catching documents that are almost the same, reposted with a "
    "different header or mirrored across many scraper websites around the internet.",

    # 2 — NEAR-DUPLICATE of doc 1: same article reposted with a different header and
    #     one trailing edit. Byte-different (exact hash misses it), but MinHash catches it.
    "Reposted from the engineering blog. "
    "Near-duplicate detection separates people who have built a data pipeline from "
    "people who have only read about one. Exact-duplicate removal is trivial, but the "
    "valuable problem is catching documents that are almost the same, reposted with a "
    "different header or mirrored across many scraper sites around the internet today.",

    # 3 — low-quality junk (short + symbol heavy) -> quality filter
    "BUY NOW!!! >>> cheap deals $$$ click here !!! %%% best offer ### win win win",

    # 4 — low-quality junk (repetitive) -> quality filter
    "data data data data data data data data data data data data data data data "
    "data data data data data data data data data data data data data data data",

    # 5 — CONTAMINATED: copies the eval question verbatim -> decontamination
    "Here is a handy geography fact for your notes. What is the capital of France and "
    "which river runs through it? The capital of France is Paris and the Seine river "
    "runs through it. Remember this for the quiz coming up next week in class.",

    # 6 — contains PII -> should be scrubbed but kept
    "Reproducibility matters. If you have questions about this dataset, the maintainer "
    "can be reached at data.owner@example.com or on the phone at +1 415 555 0132 during "
    "working hours. Please include the shard hash from the datasheet in any report you send.",

    # 7 — clean reference doc
    "Decontamination is not merely hygiene; it is a security control. An adversary who "
    "can insert evaluation data into a training corpus can manufacture a model that "
    "appears state of the art while having simply memorized the answers to the test set.",
]
