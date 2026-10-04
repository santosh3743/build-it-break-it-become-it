# Capstone — Security and ethics notes

## Threat framing: the asset is a person

Every earlier lab protected a system. The capstone protects a person. A digital twin speaks in someone's name, in their voice, to people who may not know them, so the three failures that matter are about identity, not uptime:

- **Impersonation.** In one direction, visitors believing they are talking to the real person. In the other, attackers claiming to be the person to change the twin's rules, unlock its tools, or hit its kill-switch. The threat model treats both (T11, T12, T13) and the charter refuses and escalates both (R01, E05).
- **Leakage.** The twin revealing private information: the author's private life, other visitors' conversations, memory contents, logs, configuration (R03, T14, T33). The strongest control is structural: the twin is never given private data in the first place (DATA rules DA04 to DA06).
- **Misrepresentation.** The twin attributing to the author an opinion, position or experience he never published (R02, T15), or being manipulated through memory into doing so later (T18). Grounding with citations, refusal on unknown, a faithfulness eval, and corrections logged for the author rather than written into memory as fact (E06).

One threat is accepted rather than treated, and written down as such: twin output clipped and passed off as the author off-platform, or his voice cloned from it (T38). No control inside this system can stop a screenshot. Labelling reduces it; the red-team in Post 17 re-tests it.

The likelihoods and impacts in `twin_threat_model.py` are illustrative, as in Lab 07: one reasonable reading of a public, disclosed twin, not an assessment.

## Consent and disclosure first

- **The person decides.** A twin is built only of a person, by that person or with their explicit consent. The charter is theirs: what the twin may say, must refuse, must escalate, and may do. The worked example is the series author's own twin, built by him from his own published writing.
- **Visitors are always told.** DISCLOSURE rules are `(locked)`. The twin opens every conversation with the charter's `disclosure-text`, answers "are you human?" and "are you him?" truthfully every time, and labels its drafts as AI-drafted. Removing disclosure fails `charter.py` and the capstone tests.
- **Published data only.** The corpus is the author's own published writing. Private messages, email, calendars, documents and third parties' personal data are denied. Visitor conversations are not used for training without separate, explicit consent (DA07).
- **No action without the person.** The twin's only unapproved action is reading a published FAQ. Drafts and call requests queue for the author's approval. The kill-switch is reachable only through an authenticated owner channel, never through chat.
- **Hand off to the real person.** Commitments, security reports, legal or press questions, and anyone who appears to be in distress go to the author (MUST ESCALATE), not to the model.

## The defense that ships with this post

There is no attack in Post 13, so the defense is a set of design gates that fail CI:

- `charter.py` fails a charter that is missing a section, has removable disclosure, has no escalation path, runs a side-effecting tool with `approval: none`, or lacks a default-deny for unlisted actions.
- Lab 07's validator, run by `twin_threat_model.py` against `architecture.mmd`, fails an unzoned component, an unreviewed boundary-crossing flow, an OWASP LLM 2025 or Agentic 2026 item with no control and no acceptance, an untreated threat, or a diagram node with no component.
- `charter_gaps()` fails when the charter and the threat model disagree (a tool in one and not the other, a disclosure or escalation policy with no control behind it).
- `component_controls()` fails when any component has no control.

## Limits you should know about

- **Complete is not correct.** A passing validator means every box has an answer. It does not mean the controls work; every one of them is unverified until Posts 15 to 17 test it against the running twin.
- **The charter is policy, not enforcement.** Post 16's guardrail gateway enforces it. Until then, it is a contract the code is built against.
- **Tag coverage is shallow.** `charter.py` checks that MUST REFUSE mentions impersonation; it cannot check that the rule is well written. Read the charter, do not just validate it.

## Responsible use

Build a twin only of yourself, or of someone who has given explicit, informed consent and keeps control of its charter and kill-switch. Do not build a twin of a real person from their writing without their consent, do not present a twin as the person, and do not use the Post 17 red-team techniques against anyone else's twin or AI system without permission. See the repository-level [SECURITY.md](../SECURITY.md).
