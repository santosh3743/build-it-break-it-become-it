# SECURITY.md — Lab 01

## Why a data pipeline is a security control

This lab looks like data hygiene. It is also the **first attack surface in the entire model stack** — the vulnerability that exists before a single weight does.

An attacker does not need access to your training run. They need access to a *source you ingest*. If they can put crafted documents into that source, they can:

- **Poison** the model — plant a backdoor or bias an output (demonstrated for real in Lab 11).
- **Contaminate** your evaluations — leak test answers into training so a weak model scores as state-of-the-art, and you ship it believing it's good.
- **Amplify** their influence — a poison document repeated across many mirrored pages carries far more weight than one.

The stages in `pipeline.py` are the defenses, viewed through a security lens:

- **Provenance + hashing** (the datasheet's per-shard SHA-256, and exercise 5's per-source hashes) is your **supply-chain control**. Treat the corpus like a signed software dependency: if you can't say where a document came from and prove it hasn't changed, you can't trust it.
- **Decontamination** protects the **integrity of measurement**. Poisoned evals mean you're flying blind.
- **Near-deduplication** limits attacker **leverage** by collapsing repeated poison.

## Responsible use

This lab is defensive and educational. It processes only the synthetic sample corpus shipped here. The "PII" and "eval leak" documents are fabricated for demonstration. If you point the pipeline at real data (exercise stretch goal), use only data you are permitted to process, and keep any redaction output private. Nothing here should be used to *construct* poisoned corpora against systems you do not own.
