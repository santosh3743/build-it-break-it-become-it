# Lab 07 — Exercises

Six exercises, roughly in order of difficulty. Before you run anything, write down what you think the validator or the attack tree will say. Making that prediction is the real exercise.

Everything you change lives in `reference_model.py` unless the exercise says otherwise. `python validate.py` is your feedback loop; it runs in well under a second.

---

### 1. ⚙ Add a tool and watch the build fail

**Goal:** feel the validator acting as a CI gate.

Your product team wants the agent to read the user's calendar. Add a `calendar_api` external entity, a flow `tools -> calendar_api` ("calendar query") and a flow back ("calendar events"). Do not add any threats.

**Predict first:** which gap types fire, and how many messages each?

**Done when:** you have added the threats and controls needed for `python validate.py` to pass again, and you can say which existing controls you reused and why. Hint: the hygiene rows T01-T03 target a fixed tuple of flows (`CROSSING`). Extending them is legitimate, but ask yourself whether a calendar API deserves at least one specific threat too (whose calendar is it reading?).

---

### 2. ⚙ Close the accepted item: add the Lab 12 orchestrator

**Goal:** see an acceptance decision expire.

ASI07 Insecure Inter-Agent Communication is accepted because the architecture has one agent. Add an `orchestrator` process and a `worker_agent` process in the `app` zone, with flows between them and the existing agent loop. Then delete the ASI07 acceptance.

**Predict first:** the orchestrator and worker are in the same zone. Will the validator demand a STRIDE review of their flows? Should it?

**Done when:** ASI07 is mapped to a control you wrote (message provenance and schema validation between agents is a good start), and you have decided whether agents should share a trust zone at all. Hint: Lab 12's whole argument is that one compromised worker should not be trusted by its peers. Try putting each agent in its own zone and see what the validator asks for.

---

### 3. ⚙⚙ Make the validator catch shallow reviews

**Goal:** tighten what "reviewed" means.

Right now one generic hygiene threat can satisfy the STRIDE check for a flow. Add a seventh gap type, `depth`, to `validate.py`: every flow that crosses **into a higher-trust zone from a zone of trust 1 or lower** must have at least one threat that is not a hygiene row (T01-T03).

**Predict first:** which flows fail on the complete model?

**Done when:** the new check runs, you have fixed or justified every flow it flags, and there is a test for it in `tests/test_threat_model.py`. Hint: `Zone.trust` is already on every zone; compare `m.zone(src.zone).trust` with `m.zone(dst.zone).trust`.

---

### 4. ⚙⚙ Find the minimal control set

**Goal:** use the attack tree to answer a budget question.

With no controls there are 15 attack paths. Write `minimal_cut(tree, candidates)` in `attack_tree.py`: the smallest set of controls such that `attack_paths(tree, cut)` is empty. Brute force over combinations is fine at this size.

**Predict first:** how many controls do you need, and which ones? Is C02 (least privilege) in every minimal set?

**Done when:** the function returns a minimal set, you have checked whether the answer is unique, and you can explain why branches C and D force specific controls. Hint: two leaves have no controls at all, so their AND siblings must be blocked.

---

### 5. ⚙⚙ Replace the ranking with a cost-aware one

**Goal:** see how much the "top controls" list depends on the scoring rule.

Add an illustrative `cost` (1-5) to each `Control` and rank by risk reduced per unit of cost instead of maximum risk treated. Define "risk reduced" as the sum of risk of the threats for which this control is the **only** control.

**Predict first:** does human approval (C04) stay near the top once you account for how many threats share it with another control?

**Done when:** you can print both rankings side by side and explain the biggest mover. Hint: C04 appears on eight threats but is never the only control on any of them.

---

### 6. ⚙⚙⚙ Generate the threat model from the architecture

**Goal:** close the "nobody drew it" gap from the other side.

Today `ref_ids` connects components to `reference-architecture/architecture.mmd` by hand. Write `bootstrap.py`, which parses the Mermaid file (nodes, edges, subgraphs) and produces a skeleton `ThreatModel`: one component per node, one flow per edge, the `kind` guessed from the node shape (`[(...)]` is a data store, `([...])` an external entity), and zones left as `None`.

**Predict first:** run the validator on the skeleton. How many gaps? Which type dominates?

**Done when:** the skeleton validates (structurally) and the remaining gaps are exactly the decisions a human has to make: zones, threats and controls. Hint: `validate.reference_node_ids()` already has the node regex; you need edges too (`A --> B`, with an optional `|label|`).
