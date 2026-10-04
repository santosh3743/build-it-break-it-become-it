# Capstone — Guardrails (stub)

Filled by **Post 16 — Capstone III: Productionizing Your Twin**.

The production layer: the guardrail gateway (charter checks, escalation routing), eval scorecard for faithfulness and voice (Lab 04), rate and cost limits (Lab 05), tracing and the kill-switch (Lab 06), the Dockerfile and the ask-my-twin endpoint.

This folder is a placeholder in Post 13. It deliberately contains no code and
no tests yet: CI runs `pytest` in every `capstone/*/` folder that has tests,
so tests arrive together with the code they test.

What it must honour, already defined in Post 13:

- MUST ESCALATE and DISCLOSURE in `../twin-charter.md`
- controls C05, C13, C14, C15, C18, C19, C21, C22, C23 in `../twin_threat_model.py`
- threats T10, T11, T26, T27, T30, T31
