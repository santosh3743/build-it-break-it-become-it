# Capstone kickoff — Exercises

Six exercises, roughly in order of difficulty. Before you run anything, write down what you think the validators will say. Making that prediction is the real exercise.

Your feedback loop is `python run_demo.py` (or `python charter.py` and `python twin_threat_model.py` separately). Each runs in well under a second.

---

### 1. ⚙ Give the twin a new tool and watch both validators fail

**Goal:** feel the charter and the threat model acting as one gate.

Add an action to `twin-charter.md`: `- [A06] tool: post_social | approval: author | scope: post a short update to the author's social accounts.` Do not touch the threat model.

**Predict first:** does `charter.py` pass? Does `charter_gaps()`? What about Lab 07's validator?

**Done when:** you have added a `tool_post_social` component, its flows, its threats and controls, and a `POST` node in `architecture.mmd`, and everything passes again. Hint: the charter passes on its own (approval is `author`), the cross-check fails, and Lab 07's validator only notices once you add the flows. Then decide: does posting in the author's name belong in a twin at all? Removing the action is also a valid answer.

---

### 2. ⚙ Break disclosure three ways

**Goal:** make sure "non-removable" is tested, not promised.

Try three edits, one at a time: remove `(locked)` from D02; delete the `disclosure-text:` header; rewrite `disclosure-text` so it never mentions AI ("You are talking to Santosh's assistant.").

**Done when:** you know which check catches each, and you have found the one that the current validator does NOT catch as well as it should. Hint: look at how `validate()` decides whether the text "says AI". Write a stricter check and a test for it.

---

### 3. ⚙⚙ Write the escalation router's contract

**Goal:** turn MUST ESCALATE into something Post 16 can implement.

Write a function `route(message: str, charter) -> "answer" | "refuse" | "escalate"` that uses only the charter's tags plus a small keyword table you define per tag (`#commitment`, `#security-report`, `#safety`, `#press-or-legal`). Write ten test messages, including two that should escalate but use none of your keywords.

**Done when:** your tests pass and you can explain why the two keyword-free messages are the real design problem. Hint: this is exactly why C21 sits in the threat model as a control that Post 16 must verify with an eval, not a regex.

---

### 4. ⚙⚙ Tighten the side-effect tripwire

**Goal:** see the limits of a name-based check.

`charter.py` flags a side-effecting tool by the verbs in its name. Rename `book_call` to `calendar_helper`, set `approval: none`, and run the validator.

**Done when:** the charter fails again for the renamed tool. Hint: one option is a required `effect:` field on every action (`read`, `write`, `external`) with the rule "anything but `read` needs `approval: author`". Update the charter format notes at the top of `twin-charter.md`, the parser, and add a test for the rename.

---

### 5. ⚙⚙ Re-open ASI07 honestly

**Goal:** practise moving an accepted risk back into scope.

Post 16 or 17 may add Lab 12's verifier as a second agent that checks the twin's answers before they reach the gateway. Add a `verifier` component, the flows to and from it, and remove ASI07 from `OWASP_ACCEPTED`.

**Predict first:** how many gaps appear, and of which types?

**Done when:** the model passes with ASI07 treated by a control. Hint: Lab 07's C17 (service identity) is a start, but inter-agent messages also need provenance and schema validation (Lab 12's defense). Decide whether that is a new control or an extension of C19.

---

### 6. ⚙⚙⚙ Make the definition of done grade Post 18 in advance

**Goal:** write a check before the thing it checks exists.

Pick the Post 18 item "Disclosure is non-removable on the public path". Write a `check` function for it in `dod.py` that takes rendered HTML (a string) and passes only if `disclosure-text` appears verbatim, inside an element that is visible before any answer, and is not removable by a query parameter you choose to model (for example `?embed=1`). Feed it three hand-written HTML snippets: one good, one where the banner comes after the answer, one where `embed=1` hides it.

**Done when:** the check fails two of the three snippets for the right reasons, and the item still shows `[ ]` in `python dod.py` because no public page exists yet. Hint: the item should only flip to `[x]` when the check runs against the real page Post 18 serves, so wire it to read that page's output and return `False` if it is missing.
