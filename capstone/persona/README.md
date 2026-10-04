# Capstone — Persona (stub)

Filled by **Post 14 — Capstone I: Your Corpus and Your Voice**.

The twin's voice: a system contract generated from the charter's MAY SAY / MUST REFUSE sections, plus few-shot voice exemplars taken from the author's published writing.

This folder is a placeholder in Post 13. It deliberately contains no code and
no tests yet: CI runs `pytest` in every `capstone/*/` folder that has tests,
so tests arrive together with the code they test.

What it must honour, already defined in Post 13:

- MAY SAY and MUST REFUSE in `../twin-charter.md` (parse them with `charter.load()`)
- controls C12, C24 in `../twin_threat_model.py`
- threats T15 (misrepresentation), T25 (prompt extraction), T28 (persona tampering)
