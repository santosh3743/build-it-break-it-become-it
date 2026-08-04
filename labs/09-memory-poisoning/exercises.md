# Lab 09 — Exercises

Work these in order. Each builds on the last. Solutions sketch at the bottom.

### 1. Prove it's really query-only (⚙)
In `attack.py`, confirm the attacker never calls anything except `agent.act(..., source="user")`. Add an assertion or a counter that fails if the attack ever touches `agent.memory` directly. Re-run `run_demo.py`.

### 2. Make the attack more robust (⚙⚙)
Lower `rounds` in `poison_memory()` to `1`. Does the attack still hit 100%? Now change the victim phrasings in `scenario.py` to be *less* similar to the attacker's bridging queries. Find the point where the attack degrades — that's the embedding-neighbourhood boundary MINJA has to win.

### 3. Defeat detection-only defense (⚙⚙)
Suppose the only defense were `Defense.audit()` running nightly. Modify a bridging query so its stored action still routes money to the attacker but the record would **not** be flagged by the current `audit()` rule (hint: the audit keys on `is_sensitive(action) and source != "seed"`). What does this teach you about signature-based detection?

### 4. Add memory quarantine so the legitimate flow recovers (⚙⚙⚙)
Right now the defended agent returns `escalate_to_human` for the victim's benign requests. Add a fourth control to `Defense`: when a write is refused or a decision is escalated, **quarantine** the matching stored records (mark them inactive) so retrieval falls back to the trusted seed and the victim again gets a clean `transfer_to_saved_payee`. Add a test asserting the legitimate action returns after quarantine.

### 5. Swap in a real embedder (⚙⚙⚙)
Replace `embeddings.embed` with `sentence-transformers` (`all-MiniLM-L6-v2`). Nothing else should change. Re-run the tests. Does attack success move? This demonstrates the attack is a property of the *memory architecture*, not the embedding model.

---

## Solution sketches

1. Assertion passes — `poison_memory` only ever calls `act()`.
2. At `rounds=1` the attack typically still succeeds here because the bridging queries are near-verbatim; it degrades once victim phrasing diverges enough that a seed record becomes the nearest neighbour.
3. Any action string that moves money but isn't caught by the exact `is_sensitive` prefix (e.g. a differently-named transfer action) evades a signature rule — the lesson behind why the write/retrieval/decision guards, not `audit()`, are load-bearing.
4. Add `quarantined: bool` to `Experience`; in `on_write`/`on_decision`, set it; in `filter_candidates`, drop quarantined records; seed fallback then returns `transfer_to_saved_payee`.
5. Attack success stays high — memory poisoning is architecture-level, not embedder-level.
