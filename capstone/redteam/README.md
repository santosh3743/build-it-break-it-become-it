# Capstone — Red team (stub)

Filled by **Post 17 — Capstone IV: Red-Teaming Your Own Twin**.

The Act II attacks (Labs 08, 09, 10, 12) run against the twin, a before/after attack-success table on the sensitive paths, the red-team report and `security-card.md`.

This folder is a placeholder in Post 13. It deliberately contains no code and
no tests yet: CI runs `pytest` in every `capstone/*/` folder that has tests,
so tests arrive together with the code they test.

What it must honour, already defined in Post 13:

- the MUST REFUSE tags in `../twin-charter.md` are the attack classes
- `twin_threat_model.checklist()` is the control checklist to grade (Lab 07's `checklist.py`)
- accepted threat T38 (off-platform impersonation) is re-tested here
