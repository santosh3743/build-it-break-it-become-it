# Security & responsible use — Lab 09

**Threat:** persistent agent memory (OWASP Agentic ASI06 Memory & Context Poisoning; OWASP LLM04 Data and Model Poisoning, LLM08 Vector and Embedding Weaknesses). An attacker who can only talk to an agent can get it to store records that later steer other users' requests.

**Scope of this lab:** a toy banking assistant with fictional accounts (`ATTACKER99` and friends). The attack runs only against this repository's own sample agent. Nothing here targets a real product, model or service.

**The defence ships with the attack:**
1. Provenance-gated memory writes (records learned from untrusted turns cannot encode sensitive actions).
2. Trust-filtered retrieval (low-provenance records are not reused for sensitive decisions).
3. Decision-time authorisation (sensitive parameters must come from verified state, never from retrieved text); fails closed by escalating to a human.
Each layer alone drops attack success from 100% to 0% in the tests. Fail-closed escalation can itself be abused to force manual review, so production systems also quarantine flagged records.

**Responsible use:** use these techniques only on systems you own or are explicitly authorised to test. See the repository-level `SECURITY.md`.
