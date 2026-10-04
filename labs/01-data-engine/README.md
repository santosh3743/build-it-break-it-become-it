# Lab 01 — The Data Engine

> Companion to **Build It, Break It, Become It**, Post 1: *The Data Engine — Where Every Model Is Actually Won or Lost.*

A minimal, dependency-free pretraining **data pipeline**. It takes a raw text corpus and runs it through the same sequence of controls a real lab uses — then prints a before/after report so you can watch each control do its one job.

**Runs on a laptop. CPU-only. No numpy, no torch. Standard library only.** Finishes in under a second.

---

## The pipeline

```
raw text
  → normalize            (strip boilerplate, standardize whitespace)
  → near-dedup           (MinHash + LSH — catches reposts an exact hash misses)
  → quality filter       (length, symbol ratio, repetition heuristics)
  → decontaminate        (remove anything overlapping the eval set)
  → PII scrub            (redact emails, phones, cards, keys)
  → tokenize             (build vocab, encode)
  → shard + datasheet    (fixed-size token shards + an auditable manifest)
```

Each stage removes one nameable class of failure. Skip a stage and you don't save time — you defer a bug to somewhere far more expensive to find.

## Run it

```bash
python run_demo.py        # full pipeline on the sample corpus + before/after report
python -m pytest -q       # or:  python tests/test_pipeline.py   (no pytest needed)
```

## What you'll see

The sample corpus (`sample_corpus.py`) is hand-built to exercise every stage: clean docs, one **near-duplicate** (a repost with a changed header — byte-different, so an exact hash misses it), two low-quality junk docs, one doc that **leaks an eval question**, and one doc containing **PII**. The demo shows 8 documents in → 1 near-dup removed, 2 quality-filtered, 1 decontaminated, PII redacted, 4 clean docs out — plus a **datasheet** with per-shard SHA-256 hashes.

## Files

| File | What it is |
|---|---|
| `pipeline.py` | The whole pipeline — each stage is a small, readable function. |
| `sample_corpus.py` | Tiny corpus + eval set that trips every control. |
| `run_demo.py` | Runs the pipeline, prints the before/after report + datasheet. |
| `tests/test_pipeline.py` | Asserts every control does its job (8 tests). |
| `exercises.md` | Go deeper — tune the LSH curve, add filters, try to beat decontamination. |
| `SECURITY.md` | Why this pipeline is also your first supply-chain defense. |

## The one idea

The model is downstream of the data. Everything you build in later labs — the trainer, the fine-tuner, the evals — sits on top of this pipeline and can be no better than the tokens it produces. And, as `SECURITY.md` explains, this is also the **first place an attacker can reach you**. We build these stages as hygiene now; in Lab 11 we come back and watch them stop a real poisoning attack.
