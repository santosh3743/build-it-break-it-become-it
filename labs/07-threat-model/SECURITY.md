# Lab 07 — Security notes

## Threat framing

This lab threat-models the series' reference agent architecture: a user-facing agent with an LLM, tools (including MCP servers), long-term memory, a RAG corpus, a customer database, a data and training pipeline, a model registry, and logs. The architecture is generic and fictional. The threats, likelihoods and impacts are **illustrative**: one reasonable reading of a customer-support agent, written to teach the method. They are not an assessment of any real product.

The attack tree describes attacker goals and steps at the level of a design review ("plant instructions in a web page a tool fetches"). It contains no payloads, no exploit code and no targeting information. The concrete attacks, each with its defense and a measured before/after, live in Labs 08 to 12.

## The defense that ships with this lab

There is no exploit here, so the defense is a process control: `validate.py`. Run it in CI next to your agent code and an incomplete threat model fails the build. Specifically, it fails when:

- a component has no trust zone,
- a flow that crosses a trust boundary has no STRIDE review for tampering, information disclosure or denial of service (or a written reason why one does not apply),
- an OWASP LLM (2025) or Agentic (2026) item is neither treated by a control nor explicitly accepted with a reason,
- a threat has no control and no acceptance,
- a node of `reference-architecture/architecture.mmd` was never modeled.

## Limits you should know about

- **A passing validator means complete, not correct.** It checks that every box has an answer, not that the answer is good or that the control works.
- **It cannot see what was never written down.** The cross-check against the architecture diagram helps, but if both forget a component, nothing catches it.
- **Attack-tree "blocked" means a control is listed, not that it is effective.** Treat every control in `out/controls.md` as unverified until an Act II lab or the Post 17 red-team has tested it.
- **Risk scores are ordinal guesses.** Likelihood x impact orders a backlog. It is not a measurement.

## Responsible use

Use this method on systems you own or are authorized to assess. A threat model of a real system is sensitive: it lists that system's weaknesses in priority order. Keep real threat models in access-controlled repositories, and do not publish one for a system you do not own. See the repository-level [SECURITY.md](../../SECURITY.md).
