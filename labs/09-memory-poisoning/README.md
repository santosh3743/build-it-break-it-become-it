# Lab 09 — Memory Poisoning: Compromising an Agent That Remembers

**Series:** Build It, Break It, Become It · **Act II — Break It**
**Difficulty:** ⚙⚙⚙ · **Hands-on time:** ~60 min · **Needs:** Python 3.10+ only. No GPU, no API key, no network.

> An agent with memory is an agent with an attack surface that persists after you log off. In 2025, researchers showed you can poison that memory **without ever touching the database** — just by talking to it.

---

## What you'll break today

A banking-assistant agent that decides what to do by **retrieving similar past experiences** from its memory and reusing them. You will:

1. Run a **query-only** poisoning attack (the attacker never writes to the memory store — it only sends normal-looking messages).
2. Watch a benign victim request ("pay my rent to my saved payee") get **silently rerouted to the attacker's account**.
3. Build a three-layer defense and drive attack success from **100% → 0%**.

```bash
python run_demo.py     # attack, then detection, then defense — with a before/after table
python test_lab.py     # proves: attack works, and each defense layer independently stops it
```

Actual output from `run_demo.py`:

```
Attack success (no defense): 100.0%   (5/5 requests hijacked)
Attack success (defended):     0.0%   (0/5 requests hijacked)
```

---

## The concept, at depth

**Persistent agent memory** is what makes modern agents useful across sessions: they store past interactions, "successful experiences," and retrieved documents, then reuse them. Retrieval-augmented generation (RAG), episodic/experience memory, and reflection loops all write things into a store that later steers decisions.

That write path is the attack surface. **OWASP Top 10 for Agentic Applications (2026)** lists this as **ASI06 — Memory & Context Poisoning**: attackers corrupt an agent's memory, embeddings, or RAG store to manipulate its decisions across sessions.

The scary part, shown in 2025 research, is that you don't need database access:

- **MINJA** (Dong et al., 2025, *Memory Injection Attacks on LLM Agents via Query-Only Interaction*, first titled *A Practical Memory Injection Attack against LLM Agents*) plants malicious records using **query-only interaction**: normal-looking messages, with no write access to the memory store. A 2026 follow-up evaluation reports over 95% injection success and about 70% attack success under idealised conditions, and much lower effectiveness when the store already holds many legitimate memories. It works by getting the agent to write a **bridging record** that connects a future victim query to an attacker-chosen behavior.
- **MemoryGraft** (2025) implants malicious "successful experiences" so the agent later *retrieves and repeats* them.
- A systematic 2026 study of memory poisoning finds that existing prompt-injection defences do not cover it: the poison arrives as plausible, harmless-looking records written through the agent's normal path.

### How the attack works in this lab

Our agent has the vulnerability that makes memory poisoning bite:

1. **It binds parameters from untrusted text.** For a transfer request it reads an account number straight out of the user's message (`...to account X`) with no authorization check. *(see `agent.py: _parse_action`)*
2. **It learns from its own interactions.** After every turn it stores `(situation, action)` as a new experience. *(see `agent.py: MemoryAgent.act`)*

So the attacker sends a handful of ordinary-looking messages that (a) closely match how the victim will *later* phrase their rent payment, and (b) carry an inline `to account ATTACKER99` directive. The agent executes them and **memorizes** the poisoned `(victim-like situation → transfer to ATTACKER99)` pairs. When the real victim asks their normal question, the nearest neighbour in memory is now the poisoned record — and the agent reuses its action.

No database write. No malicious document. Just conversation.

---

## The lab walkthrough

| File | What it is |
|---|---|
| `embeddings.py` | Deterministic hashing bag-of-words embeddings + cosine. Offline stand-in for a real embedder — swap in a sentence-transformer and nothing else changes. |
| `agent.py` | `MemoryAgent`: retrieve nearest experience → act → learn. Contains the two deliberate weaknesses (untrusted parameter binding, self-learning writes). |
| `attack.py` | `poison_memory()`: the MINJA-style query-only attack. |
| `defense.py` | `Defense`: three independently-toggleable controls + a post-hoc `audit()`. |
| `scenario.py` | The seeded agent, the victim's benign requests, and the success metric. |
| `run_demo.py` | Baseline → detection → defended, with the before/after table. |
| `test_lab.py` | Proves the claims (and that *any single* control stops the attack). |

### The defense (defense in depth)

Any **one** of these drops attack success to 0% — together they are layered:

1. **Provenance write-guard** — never persist an irreversible, parameter-bound action (a transfer to a specific account) learned from an *untrusted* turn. Treat a memory write like a code commit.
2. **Trust-filtered retrieval** — never let a low-trust, learned "transfer to account X" record influence a decision, even if it got stored.
3. **Decision authorization (fail closed)** — a transfer to a **non-approved** account is escalated to a human, never executed silently.

### Reading the defended output honestly

In the defended run, the victim's requests return `escalate_to_human` rather than a clean `transfer_to_saved_payee`. That is the fail-closed property: once this request pattern has been targeted, the system routes it to human review instead of moving money. **Zero requests reach the attacker.** Note the cost: the attacker can no longer steal, but can still force every victim request into manual review, which is a denial of service. Quarantining the flagged records is what restores normal service. A production system would additionally quarantine the flagged records (see exercises) so the pattern returns to normal after review.

---

## Deeper Dive

- **Why memory poisoning is stealthier than prompt injection:** prompt injection lives in a single turn; memory poisoning **survives the session** and fires later, against a *different* (benign) input. There's no malicious text in the victim's request to filter.
- **The bridging step:** MINJA's core trick is engineering the stored record to sit *between* the attacker's setup and the victim's future query in embedding space. Our `rounds` parameter in `attack.py` reinforces that neighbourhood — increase it and watch robustness rise.
- **Taxonomy:** the 2026 systematic study frames poisoning at three levels — **model** (what the agent trusts), **prompt** (how context is assembled), **system** (how memory is written/retrieved). Our three defenses map onto system-level controls; real deployments need all three levels.

---

## Defender's note

The single most important habit: **memory writes deserve the same review discipline as code commits.** Provenance tags, trust scores (this lab records them; using them in the filter is an exercise), TTLs on learned records, least privilege on the actions memory is allowed to encode, and anomaly detection on the *write* path — not just the read path. Detection alone is reactive; the win is failing closed on sensitive, low-provenance actions.

---

## Your turn

See [`exercises.md`](./exercises.md) — five graded exercises from "make the attack more robust" to "add memory quarantine so the legitimate flow recovers."

---

## References

- OWASP GenAI Security Project — *Top 10 for Agentic Applications (2026)*, ASI06 Memory & Context Poisoning.
- Dong et al., *Memory Injection Attacks on LLM Agents via Query-Only Interaction* (MINJA; earlier titled *A Practical Memory Injection Attack against LLM Agents*), 2025 — arXiv:2503.03704.
- Devarangadi Sunil et al., *Memory Poisoning Attack and Defense on Memory Based LLM-Agents*, 2026 — arXiv:2601.05504 (evaluation of MINJA under realistic conditions).
- Srivastava & He, *MemoryGraft: Persistent Compromise of LLM Agents via Poisoned Experience Retrieval*, 2025 — arXiv:2512.16962.
- *From Untrusted Input to Trusted Memory: A Systematic Study of Memory Poisoning Attacks in LLM Agents*, 2026 — arXiv:2606.04329.
- OWASP GenAI Security Project — *Top 10 for LLM Applications (2025)*, LLM08 Vector & Embedding Weaknesses.

*Ethics: every target here is this repo's own toy agent. The attack ships with its defense. For research and defense only.*
