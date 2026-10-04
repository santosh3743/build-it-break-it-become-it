# Lab 04 — Security Notes

This is a Build-It lab. It contains no attack to run and nothing in this folder is a weapon. It carries a `SECURITY.md` because **an eval harness is a security control**: it's the place where "this release can't be talked into leaking a secret" either gets checked on every build or doesn't get checked at all.

---

## The threat framing

| What goes wrong | What it looks like | What this lab does about it |
|---|---|---|
| **Silent safety regression** | A prompt or model change reopens a jailbreak. The overall score goes *up*, so nobody looks. | `candidate-v3-quiet` is exactly this. The gate checks safety items one by one and blocks on any item that passed before and fails now. |
| **Noise mistaken for signal** | A team ships (or rolls back) on a 2-point move that is well inside the noise, or learns to rerun a flaky gate until it passes. | Every score carries a 95% bootstrap CI. The gate blocks on the paired CI, not the point estimate. |
| **A biased grader** | An LLM judge that prefers whichever answer it reads first, or the longer one, approves the wrong release. | `judge.py` measures both biases against gold labels and applies position swap plus an explicit rubric. |
| **Moving the goalposts** | Someone edits the golden set (on purpose or not) and the new numbers get compared with the old baseline. | The baseline stores a fingerprint of the golden set. The gate exits 2 if it changed. |
| **Untrusted code execution** | A code eval runs model-written code with `eval`/`exec` and the "sample" does something other than compute. | `pass_at_k.safe_eval` accepts integer arithmetic only, by AST allowlist; the tests try to break out of it. |

## The safety slice: what it proves and what it doesn't

The eight safety items use **canaries**, fake secrets such as `sk-test-FAKE-ADMIN-0001`. A canary is planted only to detect leakage, and checking for it is a substring test rather than a judgement call. The items cover direct prompt injection, role-play jailbreaks, indirect injection (instructions inside an email, a translation request and a fetched web page) and a fake "system override". These map to OWASP Top 10 for LLM Applications 2025, **LLM01 Prompt Injection** and **LLM07 System Prompt Leakage**.

What it does **not** prove:

- **Eight items is not a red-team suite.** It's a regression tripwire. Passing it means "these eight known attacks still fail", not "the model is safe". The planned Lab 08 will build a real injection suite and Lab 17 red-teams the capstone.
- **The models are mocks.** A real model's refusals vary with sampling. With temperature above 0, you need several runs per safety item and a rule about how many leaks are acceptable (Exercise 6).
- **A refusal is not a control.** The safety slice measures what the model *does*. It can't guarantee what it *can't* do. Secrets that must not leak shouldn't be in the model's context at all; enforce limits outside the model (Lab 06 builds that layer).
- **The judge mitigations are modelled optimistically.** In this mock an explicit rubric removes verbosity bias completely. Real judges keep some of it, so validate any judge against human-labelled pairs before trusting it in a gate.

## The controls this lab ships

1. **Per-item results stored in the baseline** (`baseline.json`), enabling paired comparisons and per-item safety checks.
2. **A zero-tolerance safety rule** in `ci_gate.py`, independent of the overall score.
3. **A statistical regression rule** that blocks only when the paired 95% CI of the change is entirely below zero and the drop exceeds the tolerance, so the gate stays trusted.
4. **A golden-set fingerprint** that stops comparisons against a stale baseline.
5. **A sandboxed verifier** for anything that evaluates model-written expressions.
6. **Tests that guard the exam itself:** every reference must pass its scorer and every distractor must fail (`test_every_reference_passes_and_every_distractor_fails`).

## Responsible use

All data here is synthetic. `ACME`, `ATTACKER99` and every `sk-test-FAKE-...` key are fictional placeholders. The safety prompts are generic, widely known injection patterns with harmless canary payloads, included so the harness can show how to *test for* them.

If you point this harness at a real model, test only models and endpoints you own or are authorised to evaluate, never put real secrets in eval prompts (use canaries), and keep a private split of your golden set that never appears in public repos or training data.

---

Found a problem in this lab's code? Open an issue.
