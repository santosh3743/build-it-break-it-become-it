# Capstone — Memory (stub)

Filled by **Post 15 — Capstone II: Memory and Hands**.

Persistent, per-visitor memory of past conversations, reusing the Lab 09 `MemoryAgent` design with its defenses on from the start (provenance, trust scoring, namespacing, quarantine).

This folder is a placeholder in Post 13. It deliberately contains no code and
no tests yet: CI runs `pytest` in every `capstone/*/` folder that has tests,
so tests arrive together with the code they test.

What it must honour, already defined in Post 13:

- DATA rules DA03, DA06, DA07 in `../twin-charter.md`
- control C07 in `../twin_threat_model.py`
- threats T14 (cross-visitor leak) and T18 (memory poisoning)
