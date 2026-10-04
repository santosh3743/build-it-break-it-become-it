# Lab 06 — Security Notes

This is a Build-It lab, but it's the one Act II leans on most: its traces, cost ledger, redacting logger and guardrail layer are the detection and enforcement layer the Break-It labs attack and defend. This note covers what the layer protects, what it doesn't, and the new risks that telemetry itself creates.

---

## Threat framing

An LLM app in production has four failure modes this lab instruments:

| Threat | What it looks like | Control in this lab | Measured in the demo |
|---|---|---|---|
| **Prompt injection / jailbreak** (OWASP LLM01:2025) | User text that tries to override the system prompt or extract it | `InputFilter` (markers + normalization), backed by `OutputFilter` | 3 of 4 injections blocked before the model; the 4th blocked on output |
| **Sensitive information disclosure** (LLM02, LLM07) | The model repeats a secret or PII from its system prompt or context | `OutputFilter`: secrets block the response, PII is redacted | secret leaks 5/7 requests → 0/7 |
| **Excessive agency** (LLM06) | The model proposes a destructive tool call | `ToolAllowlist`, default deny | destructive tool runs 1 → 0 |
| **Unbounded consumption** (LLM10) | A flood or a loop runs the bill past budget | `BudgetBreaker` with worst-case reservation | $0.001560 → $0.000832 against a $0.0010 cap |

Plus one operational control with no OWASP number: the **kill-switch**, which stops every response before any other work happens (3/3 served → 0/3).

## The risk telemetry creates: logs as a data store

Observability means copying user input and model output into a second system with different access controls and a longer retention period. Without care, your log platform becomes the largest store of customer PII you own, and the least protected. In this lab's demo, the unprotected app's logs contain **15** secrets/PII items after seven risky requests and one normal one. The redacting logger brings that to **0**.

Controls in `logging_redact.py`:

1. **Redact before write.** Strings are scrubbed in `JsonLogger.log()` before a line exists anywhere. There's no "scrub later" step to forget.
2. **Secrets first, then Lab 01's PII scrubber.** Running Lab 01's `scrub_pii` alone leaves fragments of some key formats in the log (`sk-FAKE[PHONE]abcdEF`) and misses `sk-test-...` keys entirely. The ordering is pinned by a test.
3. **Log the decision, not the payload.** A `guardrail.block` line records the guard, the reason and a *redacted* snippet. It never records the secret that was blocked.
4. **Don't over-redact the join keys.** `trace_id`, `request_id` and numeric fields are left alone so incidents can still be reconstructed.

## What this lab does *not* protect against

Being explicit, because a control list that overstates itself is worse than none:

- **Paraphrased or encoded injections** get past the regex input filter. The output filter only stops the *harms* it has patterns for. A semantic classifier (exercise 5) narrows the gap but doesn't close it.
- **Secrets in unknown formats** and **PII that isn't pattern-shaped** (names, street addresses) get past redaction. "Priya Nair" survives the output filter in the demo.
- **Attacks via retrieved content or tool results** (indirect prompt injection) aren't modeled here: the only untrusted input is the user's message. The planned Lab 08 adds that path.
- **Per-user authorization.** The tool allowlist is per app. It doesn't know who is asking, or check tool arguments. The planned Lab 10 adds least privilege.
- **Tampering with the telemetry itself.** Logs and traces live in memory. In production they need integrity protection and access control of their own, or an attacker who gets in will edit the evidence.
- **The mock model is scripted.** Its gullibility is fixed, so the numbers measure the guardrail layer, not any real model's robustness.

## Responsible use

Everything here is fictional: the company (`ACME`), the key (`sk-test-FAKE0000ACME0000KEY1`), the people and addresses (`priya.nair@acme.example`, `riya.k@example.com`, reserved under RFC 2606), and the tools, which don't touch the file system or network. The "destructive" tool only reports what it would have done.

If you adapt the logger to real traffic, check what your jurisdiction and your users' agreements say about logging personal data, set a retention period, and remember that redaction patterns have false negatives. Test them against your own data before trusting them.

---

Found a problem in this lab's code? Open an issue.
