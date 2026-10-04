# Capstone — Hands (stub)

Filled by **Post 15 — Capstone II: Memory and Hands**.

The twin's tools, and only these: `faq_answer` (read-only), `draft_reply` (into the author's review queue) and `book_call` (a pending request). The tool list and approval levels are read from the charter's ACTIONS section, not hard-coded.

This folder is a placeholder in Post 13. It deliberately contains no code and
no tests yet: CI runs `pytest` in every `capstone/*/` folder that has tests,
so tests arrive together with the code they test.

What it must honour, already defined in Post 13:

- ACTIONS in `../twin-charter.md` (`charter.load().allowed_tools()`)
- controls C02, C04, C11 in `../twin_threat_model.py`
- threats T19, T20, T21, T22
