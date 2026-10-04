# Capstone — Corpus (stub)

Filled by **Post 14 — Capstone I: Your Corpus and Your Voice**.

The twin's knowledge: the author's own published writing, run through the Lab 01 pipeline (clean, dedup, PII-aware, chunk), with a source URL and content hash on every chunk, and an offline retrieval index (stdlib TF-IDF path).

This folder is a placeholder in Post 13. It deliberately contains no code and
no tests yet: CI runs `pytest` in every `capstone/*/` folder that has tests,
so tests arrive together with the code they test.

What it must honour, already defined in Post 13:

- DATA rules DA01, DA04, DA05 in `../twin-charter.md`
- controls C08, C09, C16, C20 in `../twin_threat_model.py`
- threats T16 (corpus poisoning) and T17 (indirect injection via a chunk)
