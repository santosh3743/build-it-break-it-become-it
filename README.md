# Build It, Break It, Become It

**Build a language model end to end. Break it the way a real adversary would. Then become something new with what you've learned.**

An advanced, hands-on series on how modern AI systems are actually engineered, attacked, and defended. Every post ships a **real, runnable lab** — not slides, not pseudocode. You clone it, you run it, you watch the result on your own machine.

If you've read the explainers and want the engineering underneath — the code, the math, the attacks, the defenses — this is for you.

> 📖 Read the series: **[cyberinfosec.substack.com](https://cyberinfosec.substack.com)** · ✍️ by **[Santosh Kumar Jha](https://www.linkedin.com/in/santosh-kumar-jha/)**
> ⭐ If a lab teaches you something, star the repo — it's how the series grows.

---

## Why this exists

Most AI security content stops at the whiteboard. Most "build an LLM" content stops at a toy that can't be attacked. This series refuses both. Across 18 posts you build a working model stack, then turn the full attacker's toolkit on it, then use every piece to build and secure a digital twin of yourself.

Three principles hold throughout:

- **The code is the argument.** If a post claims a control works, the repo lets you prove it. Every lab prints a **before/after** result you can screenshot.
- **Every attack ships with its defense.** No exploit is demonstrated without the mitigation in the same folder, with a measured drop in attack success (typically 100% → ~0%).
- **It runs on your laptop.** Every lab has an **offline, CPU-only** path. Where a real run wants a GPU or an API key, there's a mock/tiny path that shows the same shape of result — so nothing blocks you.

---

## The three acts

**Act I — Build It.** Engineer a model end to end: the data pipeline, training a small model from scratch, post-training (SFT/DPO), evaluation, serving, and production observability. You finish with a model you understand at every layer.

**Act II — Break It.** Attack what you built, the way an adversary does: threat modeling, prompt injection, memory poisoning, tool/MCP abuse, supply-chain poisoning, and multi-agent cascading failures — each paired with the defense that stops it.

**Act III — Become It.** One capstone across six posts: build a **secured digital twin of yourself** — an AI grounded in your own writing and voice that can represent you. Engineered with Act I, secured with Act II (because now the target is *your* identity and data), and shipped as public proof-of-work.

---

## Lab index

| # | Post | Act | Lab | Level | Hardware | Status |
|---|------|-----|-----|-------|----------|--------|
| 01 | The Data Engine | Build | [`labs/01-data-engine`](labs/01-data-engine) | ⚙⚙ | CPU | ✅ Done |
| 02 | Train an SLM From Scratch | Build | `labs/02-train-slm` | ⚙⚙⚙ | GPU rec. / CPU tiny | ⏳ Planned |
| 03 | Post-Training: SFT + DPO | Build | `labs/03-post-training` | ⚙⚙⚙ | GPU rec. / mock | ⏳ Planned |
| 04 | Evaluation That Means Something | Build | `labs/04-evaluation` | ⚙⚙ | CPU / API opt. | ⏳ Planned |
| 05 | Serving & the KV Cache | Build | `labs/05-serving` | ⚙⚙⚙ | GPU / CPU mock | ⏳ Planned |
| 06 | Observability, Cost & Guardrails | Build | `labs/06-observability` | ⚙⚙ | CPU | ⏳ Planned |
| 07 | Threat-Modeling an AI System | Break | `labs/07-threat-model` | ⚙ | CPU | ⏳ Planned |
| 08 | Prompt Injection | Break | `labs/08-prompt-injection` | ⚙⚙ | CPU / API opt. | ⏳ Planned |
| 09 | Memory Poisoning | Break | `labs/09-memory-poisoning` | ⚙⚙⚙ | CPU | ✅ Done |
| 10 | Tool Abuse & MCP | Break | `labs/10-tool-abuse-mcp` | ⚙⚙ | CPU / mock | ⏳ Planned |
| 11 | Supply-Chain Poisoning | Break | `labs/11-supply-chain-poisoning` | ⚙⚙⚙ | CPU | ⏳ Planned |
| 12 | Multi-Agent Cascading Failures | Break | `labs/12-multi-agent` | ⚙⚙ | CPU / mock | ⏳ Planned |
| 13 | Reborn: Why Build a Digital Twin | Become | [`capstone/`](capstone) | ⚙ | CPU | ⏳ Planned |
| 14 | Capstone I: Corpus & Voice | Become | `capstone/corpus` · `capstone/persona` | ⚙⚙⚙ | CPU | ⏳ Planned |
| 15 | Capstone II: Memory & Hands | Become | `capstone/memory` · `capstone/hands` | ⚙⚙⚙ | CPU / GPU opt. | ⏳ Planned |
| 16 | Capstone III: Productionize | Become | `capstone/` | ⚙⚙⚙ | CPU | ⏳ Planned |
| 17 | Capstone IV: Red-Team Your Twin | Become | `capstone/redteam` | ⚙⚙⚙ | CPU | ⏳ Planned |
| 18 | Ship & Tell | Become | `capstone/` | ⚙ | CPU | ⏳ Planned |

*Level:* ⚙ intro · ⚙⚙ moderate · ⚙⚙⚙ deep.

---

## Quick start

```bash
git clone https://github.com/santosh3743/build-it-break-it-become-it.git
cd build-it-break-it-become-it/labs/01-data-engine

python run_demo.py        # runs the lab, prints a before/after result
python -m pytest -q       # or:  python tests/test_pipeline.py   (no pytest needed)
```

Every lab works the same way: `run_demo.py` for the result, `tests/` to prove it. Most run in **Python 3.10+ with only the standard library**; labs that need extras ship a `requirements.txt` and a no-dependency fallback path.

---

## What's in every lab

```
labs/NN-slug/
  README.md         # what it teaches, the diagram, how to run, what you'll see
  *.py              # small, heavily-commented modules — the code is the lesson
  run_demo.py       # zero-arg entrypoint; prints the before/after result
  exercises.md      # graded exercises to push past the lab
  tests/            # asserts every claim the post makes
  SECURITY.md       # (attack labs) threat framing + responsible use + the defense
```

A recurring **reference architecture** ([`reference-architecture/`](reference-architecture)) evolves across the series: clean in Act I, annotated with attack arrows in Act II, hardened with controls in Act III. One system, watched growing up.

---

## Responsible use

The Break-It and capstone labs demonstrate real attack classes for one reason: **you cannot defend what you cannot reproduce.** Every attack here runs only against synthetic or self-owned data shipped in the repo, and every one ships with its defense. Do not use these techniques against systems you do not own or have explicit permission to test. Each attack lab carries a `SECURITY.md` with the specifics.

---

## Following along

New lab every week. **Watch** the repo for lab drops and **star** it if it's useful — that signal is what keeps the series going. Issues and PRs welcome: found a bug, a better defense, or a case a lab misses? Open one.

The full written series lives at **[cyberinfosec.substack.com](https://cyberinfosec.substack.com)** — the essays give you the *why*; these labs give you the *how*.

---

## License

Code: MIT. Prose and diagrams: CC BY 4.0. Use it, teach with it, build on it — attribution appreciated.
