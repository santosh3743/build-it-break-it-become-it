# Lab 07 — Threat-Modeling an AI System Like You Mean It

**Series:** Build It, Break It, Become It · **Act II — Break It** (opener)
**Difficulty:** ⚙ · **Hands-on time:** ~40 min · **Needs:** Python 3.10+ only. No GPU, no API key, no network, no dependencies.

> You cannot defend a system you can't draw. This lab draws the series' reference agent architecture as code, not slides, and then makes a validator refuse to call the drawing finished until every component has a trust boundary, every boundary crossing has been reviewed, and every OWASP item has an answer.

---

## What you'll build today

Act I built a model and the harness around it. Act II attacks it. Before you attack anything, you need a map, and the map has to be one that cannot quietly go stale. You will:

1. Write the **threat model as a data structure** (`model.py`, `reference_model.py`): 14 components, 20 data flows, 7 trust zones, 29 threats tagged with STRIDE, 17 controls.
2. Build a **completeness validator** (`validate.py`) that fails if any component lacks a trust boundary, any boundary-crossing flow lacks a STRIDE review, any OWASP LLM or Agentic item is neither treated nor explicitly accepted, any threat is untreated, or any node of the reference architecture diagram was never modeled.
3. Build an **attack tree** (`attack_tree.py`) for the goal "exfiltrate customer data via the agent" and count how many distinct attack paths each set of controls leaves open.
4. **Risk-rank** the controls (`risk.py`) and render everything (`render.py`): a Mermaid data-flow diagram with trust-boundary subgraphs and attacker entry points, the attack tree as text and Mermaid, and a markdown control checklist that the Post 17 capstone red-team grades itself against (`checklist.py`).

The demo runs the validator on a deliberately incomplete **first draft** (BEFORE) and on the **complete** model (AFTER).

```bash
python run_demo.py              # BEFORE/AFTER validator, attack tree, top controls; writes out/
python validate.py              # validate the complete model (exit 0)
python validate.py --draft      # validate the first draft (exit 1, lists every gap)
python -m pytest -q             # or:  python tests/test_threat_model.py
```

---

## Actual output of `run_demo.py`

```
==============================================================================
Lab 07 -- Threat-Modeling an AI System Like You Mean It
==============================================================================
Reference agent architecture: 14 components, 20 data flows, 7 trust zones, 17 flows cross a trust boundary.

[1] BEFORE -- validator on the first-draft threat model
  [boundary ] component memory (datastore) has no trust boundary
  [boundary ] component rag_corpus (datastore) has no trust boundary
  [boundary ] component model_registry (datastore) has no trust boundary
  [boundary ] component logs (datastore) has no trust boundary
  [stride   ] flow F11 agent_loop->memory crosses a boundary but has no STRIDE review for T, I, D
  [stride   ] flow F12 memory->agent_loop crosses a boundary but has no STRIDE review for T, I, D
  [stride   ] flow F13 rag_corpus->agent_loop crosses a boundary but has no STRIDE review for T, I, D
  [stride   ] flow F18 model_registry->llm crosses a boundary but has no STRIDE review for T, I, D
  [stride   ] flow F19 agent_loop->logs crosses a boundary but has no STRIDE review for T, I, D
  [stride   ] flow F20 gateway->logs crosses a boundary but has no STRIDE review for T, I, D
  [owasp    ] LLM03 Supply Chain is neither mapped to a control nor explicitly accepted
  [owasp    ] LLM04 Data and Model Poisoning is neither mapped to a control nor explicitly accepted
  [owasp    ] LLM06 Excessive Agency is neither mapped to a control nor explicitly accepted
  [owasp    ] LLM07 System Prompt Leakage is neither mapped to a control nor explicitly accepted
  [owasp    ] LLM08 Vector and Embedding Weaknesses is neither mapped to a control nor explicitly accepted
  [owasp    ] LLM09 Misinformation is neither mapped to a control nor explicitly accepted
  [owasp    ] ASI04 Agentic Supply Chain Vulnerabilities is neither mapped to a control nor explicitly accepted
  [owasp    ] ASI06 Memory and Context Poisoning is neither mapped to a control nor explicitly accepted
  [owasp    ] ASI07 Insecure Inter-Agent Communication is neither mapped to a control nor explicitly accepted
  [owasp    ] ASI09 Human-Agent Trust Exploitation is neither mapped to a control nor explicitly accepted
  [treatment] threat T10 'Direct prompt injection / jailbreak from the user' has no control and no acceptance
  [treatment] threat T17 'Bulk read of customer records through a tool' has no control and no acceptance
  [treatment] threat T19 'Agent granted more tools and permissions than the task needs' has no control and no acceptance
  [treatment] threat T20 'Model output rendered unsafely (markdown image leaks data)' has no control and no acceptance
  [treatment] threat T24 'Model regurgitates PII memorized from training data' has no control and no acceptance
  [treatment] threat T25 'Compromised or malicious tool / MCP server' has no control and no acceptance
  [treatment] threat T26 'System prompt extracted, exposing secrets and rules' has no control and no acceptance
  [treatment] threat T28 'One bad tool result cascades through later steps' has no control and no acceptance
  [treatment] threat T29 'Hallucinated facts presented as authoritative' has no control and no acceptance
  [treatment] threat T30 'Persuasive agent output talks the user into approving harm' has no control and no acceptance
  [reference] reference-architecture node PIPE has no component
  [reference] reference-architecture node SRC has no component
  [reference] reference-architecture node TOK has no component
  [reference] reference-architecture node TRAIN has no component
  RESULT: FAIL -- 34 gap(s)

[2] AFTER -- validator on the complete threat model
  RESULT: PASS -- threat model is complete

[3] BEFORE / AFTER
                                                     BEFORE             AFTER
                                                      draft          complete
  --------------------------------------------------------------------------
  Components on the diagram                              11                14
  Components with no trust boundary                       4                 0
  Boundary-crossing flows not STRIDE-reviewed             6                 0
  OWASP LLM 2025 items unmapped (of 10)                   6                 0
  OWASP Agentic 2026 items unmapped (of 10)               4    0 (1 accepted)
  Threats with no control                                10                 0
  Reference-architecture nodes not modeled                4                 0
  Controls identified                                     6                17
  Attack paths to the goal still open (of 15)             5                 0
  Validator                                            FAIL              PASS

[4] Attack tree -- goal: exfiltrate customer data via the agent
  (OR = any child, AND = all children)                             draft  complete
  --------------------------------------------------------------------------------
  OR  G Exfiltrate customer data via the agent                     REACH   blocked
    AND A Hijack the agent and walk the data out                   REACH   blocked
      OR  A1 Get attacker instructions into the context            REACH   blocked
         -  L1 Plant instructions in a web page a tool fetches   blocked   blocked
         -  L2 Plant a poisoned document in the RAG corpus       blocked   blocked
         -  L3 Poison long-term memory via a normal-looking chat    open   blocked
      OR  A2 Make the agent read records it should not             REACH   blocked
         -  L4 Tool runs as an all-tables service account           open   blocked
         -  L5 Retrieval returns other tenants' chunks              open   blocked
      OR  A3 Get the data out                                      REACH   blocked
         -  L6 Mail or POST it to an attacker address            blocked   blocked
         -  L7 Emit a markdown image whose URL carries the data     open   blocked
    AND B Ship a malicious tool that reads the data itself         REACH   blocked
       -  L8 Get a malicious MCP server installed                   open   blocked
       -  L9 The tool server holds broad database credentials       open   blocked
    AND C Extract data the model memorized                         REACH   blocked
       -  L10 Customer PII was in the training data                 open   blocked
       -  L11 Query the model with extraction prompts               open      open
    AND D Steal credentials from the system prompt                 REACH   blocked
       -  L12 System prompt holds a DB connection string            open   blocked
       -  L13 Talk the model into revealing its system prompt       open      open

  Paths still open with the draft's controls (5):
    L3 + L4 + L7
    L3 + L5 + L7
    L8 + L9
    L10 + L11
    L12 + L13
  With the complete model every path is blocked ON PAPER: each needs a step
  some listed control stops. Act II tests whether those controls actually work.

[5] Top 10 controls, risk-ranked (score = max likelihood x impact treated;
    illustrative 1-25 scale)
   #  id   control                                        score  verify in
  ----------------------------------------------------------------------------
   1  C04  Human approval for high-impact actions            25  Lab 10
   2  C02  Least-privilege tool scopes                       25  Lab 10
   3  C01  Quarantine untrusted content                      25  Lab 08
   4  C05  Output handling                                   20  Lab 08
   5  C08  Permission-aware RAG with ingestion provenance    20  Lab 08, Lab 11
   6  C07  Memory write gating                               20  Lab 09
   7  C17  Service identity and delegated authority          20  Lab 12
   8  C03  Egress allowlist                                  20  Lab 08
   9  C09  PII scrub and dedup at ingestion                  15  Lab 01
  10  C10  Signed, hash-pinned model artifacts               15  Lab 02, Lab 11

[6] Artifacts written (complete model; '-draft' versions alongside):
  out/architecture-threats.mmd
  out/attack-tree.mmd
  out/attack-tree.txt
  out/controls.md
  out/threats.md
  out/controls.json
==============================================================================
```

The headline: the first draft has **34 gaps** (4 components with no trust boundary, 6 unreviewed boundary crossings, 10 OWASP items with no answer, 10 untreated threats, 4 reference-architecture nodes never modeled) and leaves **5 of 15** attack paths to customer data open. The complete model passes with 0 gaps and 0 open paths.

Read that last number carefully. **Zero open paths is a claim on paper.** It means every path needs at least one step that some listed control is supposed to stop. Whether the control actually stops it is what Labs 08 to 12 measure.

---

## The concept, at depth

### Four questions, made executable

Adam Shostack's four-question frame for threat modeling is: *What are we working on? What can go wrong? What are we going to do about it? Did we do a good job?* This lab maps one file to each:

| Question | Artifact | File |
|---|---|---|
| What are we working on? | Components, flows, trust zones (the DFD) | `reference_model.py`, `out/architecture-threats.mmd` |
| What can go wrong? | STRIDE threats per element, the attack tree | `reference_model.py`, `attack_tree.py` |
| What are we going to do about it? | Risk-ranked controls, OWASP mapping | `risk.py`, `out/controls.md` |
| Did we do a good job? | The completeness validator | `validate.py` |

The fourth question is the one teams skip, and it is the one code can answer best. A threat model in a slide deck is "reviewed" forever. A threat model in the repo fails CI the day someone adds a tool without modeling its flow.

### Trust boundaries are where the attacks are

A **trust boundary** is any line across which data moves between parties who trust each other differently. Here they are zones: `internet` (trust 0), `sandbox` (1), `app` and `runtime` (2), `data`, `build` and `ops` (3). A flow whose two ends sit in different zones **crosses a boundary**, and 17 of the 20 flows do.

The rule the validator enforces: a component with no zone is treated as untrusted, so **every** flow touching it counts as a crossing that needs review. That is why, in the draft, forgetting to place `memory` in a zone immediately surfaces two unreviewed flows (memory write and memory recall). Lab 09 attacks exactly those two flows.

### STRIDE per element

STRIDE names six threat categories, each the violation of one property: **S**poofing (authentication), **T**ampering (integrity), **R**epudiation (non-repudiation), **I**nformation disclosure (confidentiality), **D**enial of service (availability), **E**levation of privilege (authorization). Not every letter applies to every element. The standard STRIDE-per-element chart, encoded in `model.STRIDE_PER_ELEMENT`:

| Element | Applicable |
|---|---|
| External entity (user, data sources) | S, R |
| Process (gateway, agent loop, LLM, tools, pipelines) | S, T, R, I, D, E |
| Data store (memory, RAG corpus, registry, logs) | T, R, I, D |
| Data flow | T, I, D |

The validator requires each boundary-crossing flow to have T, I and D each covered by a threat **or** explicitly ruled out with a written reason (`stride_na`). The model rules out exactly one: information disclosure on the model-weights flow, because this is an open-weight model and its weights are not secret. "Considered and ruled out" is a review. "Never thought about it" is a gap. The validator can tell them apart.

It also rejects threats with a letter that cannot apply, such as "spoofing a data flow".

### Where STRIDE strains on AI systems

Two of the OWASP items did not fit STRIDE cleanly, and the model says so rather than hiding it:

- **LLM09 Misinformation** is not an attacker violating a property. The closest fit is Tampering with the integrity of the LLM's output (T29), but nobody tampered with anything. The model was just wrong.
- **ASI09 Human-Agent Trust Exploitation** is filed as Spoofing against the user (T30): persuasive output impersonating a trustworthy authority. It is really a human-factors problem.

STRIDE was designed for software that does what its code says. A model that produces plausible wrong answers under no attack at all needs categories STRIDE does not have, which is one reason the OWASP lists exist.

### Attack trees: AND, OR and counting paths

An **attack tree** puts the attacker's goal at the root. An OR node is achieved if any child is; an AND node only if all are. Leaves are concrete steps, and each lists the controls that block it.

The tree in `attack_tree.py` has four branches under "exfiltrate customer data via the agent":

- **A** (AND): get instructions into the context (web page, RAG document, or poisoned memory) **and** make the agent read records it should not **and** get the data out (email/HTTP tool or a markdown image URL). That is 3 x 2 x 2 = 12 paths.
- **B** (AND): get a malicious MCP server installed **and** it holds broad database credentials.
- **C** (AND): customer PII was in the training data **and** someone asks the right extraction prompts.
- **D** (AND): the system prompt holds a database connection string **and** someone talks the model into revealing it.

15 paths with no controls. `attack_paths()` enumerates them. The draft's six controls leave five open; the complete model's seventeen leave none.

Two leaves have **no control at all**: "query the model with extraction prompts" and "talk the model into revealing its system prompt". You cannot stop people from asking. Those branches are only closed by their sibling step, which is the point of LLM07: assume the system prompt will be extracted, and keep secrets out of it.

### The scoring is a sorting aid, not a measurement

`risk.py` uses likelihood (1-5) x impact (1-5), and ranks each control by the highest-risk threat it treats, breaking ties by total risk treated. The numbers in `reference_model.py` are one reasonable reading of a generic customer-support agent. They are **illustrative**. Multiplying two ordinal guesses gives a third guess. Use it to order a backlog, and reach for a quantitative method if you need to defend a budget.

Even so, the ranking says something worth noticing: the top three controls are **human approval for high-impact actions**, **least-privilege tool scopes**, and **quarantining untrusted content**. Two of the three are not "AI security" at all. They are authorization design. The draft had neither.

---

## Deeper Dive: why agentic systems break the classic threat model

Classic threat modeling assumes three things that an LLM agent quietly violates.

**1. Code decides; data is data.** In a web app, the request body cannot rewrite the handler. In an agent, the model reads instructions and data through one channel, so any data the agent reads (a web page, a RAG chunk, a memory, a tool description) can become an instruction. That is why the draft's habit of threat-modeling only the user's chat input misses most of the tree: two of the three ways to get instructions into the context (a fetched web page, a poisoned RAG document) do not come from the user at all, and the third arrives through a chat that looks completely normal.

**2. Authority is explicit.** An agent's tools usually run with the service's credentials, not the user's. Every tool call is the agent spending authority it holds on behalf of whoever wrote the text it just read. This is the **confused deputy** (Norm Hardy, 1988): a program with legitimate privileges tricked into using them for someone else. Threat T18 and leaves L4 and L9 are confused-deputy threats, and C02 and C17 (least-privilege, delegated identity) are the classic answers: give the deputy only the authority the current user and task need, so being confused costs less.

**3. Behavior is deterministic.** The same input to the same code does the same thing, so you can test a control once. A model's behavior is a distribution. A defense that blocks an injection in 99 trials out of 100 is still a defense with a hole, and "it refused when I tried it" is not evidence. This is why every Act II lab measures attack success over seeded trials instead of showing one screenshot.

Put together: the attack surface of an agent is **every input it reads times every authority it holds**, sampled non-deterministically. A threat model has to be explicit about both factors, and has to be re-checked whenever either changes. Hence: threat model as code, validated in CI.

---

## Word of the Week: Confused deputy

A program that holds legitimate authority and is tricked into exercising it on behalf of someone who does not have that authority. Hardy's original example was a compiler with write access to a billing file, tricked by a user into overwriting it. The modern example: an agent with a database tool, reading a web page that tells it to email the customer table somewhere. The agent is not compromised. It is doing its job, for the wrong principal.

---

## Mapping to OWASP and MITRE ATLAS

Every item on both OWASP lists is either treated by at least one control or explicitly accepted with a reason. `validate.py` enforces this and the tests assert it.

**OWASP Top 10 for LLM Applications (2025)** — all ten treated, none accepted:

| Item | Controls |
|---|---|
| LLM01 Prompt Injection | C01 |
| LLM02 Sensitive Information Disclosure | C02, C03, C09, C12, C14 |
| LLM03 Supply Chain | C10, C11 |
| LLM04 Data and Model Poisoning | C07, C08, C09, C10 |
| LLM05 Improper Output Handling | C05, C06 |
| LLM06 Excessive Agency | C02, C04 |
| LLM07 System Prompt Leakage | C12 |
| LLM08 Vector and Embedding Weaknesses | C08 |
| LLM09 Misinformation | C16 |
| LLM10 Unbounded Consumption | C13 |

**OWASP Top 10 for Agentic Applications (2026)** — nine treated, one accepted:

| Item | Controls |
|---|---|
| ASI01 Agent Goal Hijack | C01 |
| ASI02 Tool Misuse and Exploitation | C02, C03, C04, C11 |
| ASI03 Identity and Privilege Abuse | C02, C13, C17 |
| ASI04 Agentic Supply Chain Vulnerabilities | C10, C11 |
| ASI05 Unexpected Code Execution | C06 |
| ASI06 Memory and Context Poisoning | C07, C08 |
| ASI07 Insecure Inter-Agent Communication | **Accepted:** single-agent architecture, no inter-agent channel yet. Re-open in Lab 12. |
| ASI08 Cascading Failures | C13, C15 |
| ASI09 Human-Agent Trust Exploitation | C04, C16 |
| ASI10 Rogue Agents | C14, C15 |

The Agentic item ids and names are in one place (`owasp.py`). If OWASP revises the list, edit that file and the validator will flag every mapping that no longer lines up.

**MITRE ATLAS.** Threats are tagged with ATLAS tactic names (Initial Access, Persistence, Collection, Exfiltration, Privilege Escalation, Execution, Credential Access, Resource Development, Defense Evasion, Impact) in `out/threats.md`. Two technique ids are worth knowing by heart: **AML.T0051 LLM Prompt Injection** (threats T10, T11, T12) and **AML.T0020 Poison Training Data** (T23).

---

## The lab walkthrough

| File | What it does |
|---|---|
| `model.py` | The data model: `Zone`, `Component`, `Flow`, `Threat`, `Control`, `ThreatModel`. STRIDE and the STRIDE-per-element chart. `crosses_boundary()` treats an unzoned component as untrusted. |
| `owasp.py` | The OWASP LLM 2025 and Agentic 2026 catalogs, plus the ATLAS tactic names used for tags. |
| `reference_model.py` | The reference architecture threat-modeled: zones, components (with `ref_ids` back to `reference-architecture/architecture.mmd`), flows, 17 controls, 29 threats, the one accepted OWASP item, the one STRIDE ruling-out. `build_draft_model()` derives the first draft by deleting things. |
| `validate.py` | The completeness validator: six gap types (structure, boundary, stride, owasp, treatment, reference). Parses the real reference-architecture Mermaid file to find nodes nobody modeled. CLI exit code 0/1 for CI. |
| `attack_tree.py` | AND/OR attack trees, `achievable()`, `attack_paths()` (every minimal leaf set that reaches the goal), the exfiltration tree, text and Mermaid renderers. |
| `risk.py` | Likelihood x impact per threat; controls ranked by max then total risk treated. |
| `render.py` | Mermaid DFD (one subgraph per zone, boundary crossings in red, attacker entry points dashed), markdown threat register with a STRIDE-per-element matrix, the control checklist, and `lint_mermaid()`, a structural check that runs without a Mermaid install. |
| `checklist.py` | The control checklist as an importable API for Post 17: `load()`, `parse_controls_md()` (reads ticked boxes back), `coverage()`. |
| `run_demo.py` | BEFORE (draft) vs AFTER (complete): validator output, before/after table, side-by-side attack tree, top 10 controls; writes `out/`. |
| `out/` | Generated artifacts: `architecture-threats.mmd`, `attack-tree.mmd`, `attack-tree.txt`, `controls.md`, `controls.json`, `threats.md`, and `-draft` versions of each. |
| `tests/test_threat_model.py` | 31 tests: each gap type caught in isolation, complete model passes, draft fails on the expected counts, every LLM01-LLM10 item mapped to a control, attack-path counts, Mermaid plausibility, checklist round trip, CLI exit codes. |

### The defense this lab ships

This is a Break-It lab with no exploit. Its defense is the process control: a validator that makes an incomplete threat model fail the build. Drop it into CI next to your agent code and the rule becomes mechanical: **no new component, flow or tool merges without a trust zone, a STRIDE review and a control.**

### Reading the result honestly

- **Completeness is not correctness.** The validator proves every box has an answer, not that the answers are right. The hygiene rows (T01-T03: tampering, eavesdropping and flooding on every hop, all treated by mTLS and rate limits) are what make every crossing "reviewed". They are real and boring, which is exactly why real threat models are full of them. The interesting work is in T10-T35.
- **The validator cannot see what is not in the model.** That is why it also parses `reference-architecture/architecture.mmd`: the draft left out the whole data plane, and only the cross-check against the architecture diagram caught it. If both the model and the diagram forget a component, nothing catches it. A human review still has to ask "what's missing?"
- **The attack tree's "blocked" is optimistic.** A leaf counts as blocked if any listed control exists. Leaf L2 (poisoned RAG document) is blocked in the draft because the draft has C01 (quarantine untrusted content). Whether a quarantine pattern really neutralizes a poisoned document is Lab 08's question.

### Rendering the diagrams

GitHub renders `.mmd` files and ` ```mermaid ` blocks automatically. To check locally, paste `out/architecture-threats.mmd` into the Mermaid live editor or run it through `@mermaid-js/mermaid-cli`. While building the lab, all four generated diagrams were also checked with the parser in the `mermaid` npm package (version 12.1.0); that check is not part of the test suite because the lab has no Node dependency. The tests use `render.lint_mermaid()` instead.

`out/architecture-threats.mmd` is the Act II variant of the series' reference architecture: same components, now with trust zones, red boundary crossings, and attacker entry points.

---

## Defender's note

This threat model is the checklist the rest of Act II works through. Every control in `out/controls.md` names the lab that tests it: C01, C03, C05 and C12 in Lab 08 (prompt injection); C07 in Lab 09 (memory poisoning); C02, C04, C06 and C11 in Lab 10 (tool abuse); C08 and C10 in Lab 11 (supply chain); C15 and C17 in Lab 12 (multi-agent). In Post 17 the capstone red-team imports `checklist.py`, runs the attacks against the digital twin, and ticks a box only when the attack that targets that control has been driven to roughly zero success. Straight out of this lab, coverage is 0 of 17. That is honest: naming a control is not the same as having one that works.

---

## Your turn

See [`exercises.md`](exercises.md): add a new tool and watch CI fail until you model it, add the orchestrator from Lab 12 and close the accepted ASI07 item, replace max-risk ranking with a cost-aware one, and find the minimal set of controls that blocks every attack path.

---

## References

- OWASP GenAI Security Project — *OWASP Top 10 for LLM Applications 2025* (November 2024). genai.owasp.org
- OWASP GenAI Security Project, Agentic Security Initiative — *OWASP Top 10 for Agentic Applications* (2026 edition, December 2025). genai.owasp.org
- MITRE ATLAS — Adversarial Threat Landscape for Artificial-Intelligence Systems. atlas.mitre.org (techniques AML.T0051 LLM Prompt Injection, AML.T0020 Poison Training Data)
- Loren Kohnfelder and Praerit Garg — *The threats to our products*, Microsoft, 1999 (the origin of STRIDE).
- Adam Shostack — *Threat Modeling: Designing for Security*, Wiley, 2014 (STRIDE per element; the four-question frame).
- Bruce Schneier — "Attack Trees", *Dr. Dobb's Journal*, December 1999.
- Norm Hardy — "The Confused Deputy (or why capabilities might have been invented)", *ACM SIGOPS Operating Systems Review*, 1988.
- Simon Willison — "The Dual LLM pattern for building AI assistants that can resist prompt injection", 2023 (the pattern behind control C01).
