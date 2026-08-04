"""
Run the full data pipeline on the sample corpus and print a before/after report.

    python run_demo.py

No arguments, no dependencies, no GPU. Finishes in well under a second.
"""

import json

from pipeline import run_pipeline
from sample_corpus import EVAL_TEXTS, RAW_TEXTS


def main() -> None:
    shards, tok, rep = run_pipeline(RAW_TEXTS, EVAL_TEXTS)

    print("=" * 60)
    print("  LAB 01 — THE DATA ENGINE  ·  before / after")
    print("=" * 60)
    print(f"  documents in .................. {rep.docs_in}")
    print(f"  - near-duplicates removed ..... {rep.duplicates_removed}")
    print(f"  - quality-filtered ............ {rep.quality_removed}  {rep.rejections}")
    print(f"  - contaminated removed ........ {rep.contaminated_removed}")
    print(f"  PII spans redacted ............ {rep.pii_redactions}")
    print("-" * 60)
    print(f"  documents out ................. {rep.docs_out}")
    print(f"  tokens out .................... {rep.tokens_out}")
    print(f"  vocab size ................... {rep.vocab_size}")
    print(f"  shards written ............... {rep.shards}  (size <= 256 tokens)")
    print("=" * 60)

    print("\nDATASHEET (this is what makes the dataset auditable):\n")
    print(json.dumps(rep.datasheet(), indent=2))

    print("\nSanity check — first shard decoded back to text (first 40 tokens):\n")
    if shards and shards[0]:
        print("  " + tok.decode(shards[0][:40]))

    print(
        "\nNotice: doc 2 (a near-duplicate of doc 1) was caught by MinHash even though "
        "its exact hash differs; the eval-leaking doc was decontaminated; the contact "
        "details were redacted but the doc was kept. Each control did its one job.\n"
    )


if __name__ == "__main__":
    main()
