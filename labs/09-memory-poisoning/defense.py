"""Defense-in-depth against memory poisoning.

Three independent controls — any ONE of them stops this attack; together they are
defense in depth. Each is a toggle so you can explore (in exercises.md) which layer
catches the poison and why detection-only approaches are not enough.

1. provenance_write_guard  — never persist an irreversible, parameter-bound action
   (a transfer to a specific account) that was learned from an untrusted turn.
   Memory writes get the same scrutiny as code commits.

2. trust_filtered_retrieval — never let a low-trust, learned "transfer to account X"
   experience influence a decision, even if it somehow got stored.

3. decision_authorization  — at decision time, a transfer to a non-approved account
   is escalated to a human. Sensitive actions fail closed.

Plus audit(): post-hoc anomaly detection that flags poisoned records (structural
signal: a sensitive, parameter-bound action learned from an untrusted source).
"""

from __future__ import annotations

from dataclasses import dataclass

from agent import Experience, account_of, is_sensitive, APPROVED_PAYEES


@dataclass
class Defense:
    provenance_write_guard: bool = True
    trust_filtered_retrieval: bool = True
    decision_authorization: bool = True

    def on_write(self, exp: Experience):
        if self.provenance_write_guard and exp.source != "seed" and is_sensitive(exp.action):
            return None  # refuse to memorize an untrusted irreversible action
        return exp

    def filter_candidates(self, candidates, query: str):
        if not self.trust_filtered_retrieval:
            return candidates
        return [
            e for e in candidates
            if not (is_sensitive(e.action) and e.source != "seed")
        ]

    def on_decision(self, query: str, exp, action: str) -> str:
        if not self.decision_authorization:
            return action
        acct = account_of(action)
        if acct is not None and acct not in APPROVED_PAYEES:
            return "escalate_to_human"  # fail closed on unapproved transfers
        return action

    @staticmethod
    def audit(memory) -> list[Experience]:
        """Flag records that look like planted poison. Detection is necessary but
        NOT sufficient — this is why the write/retrieval/decision guards exist."""
        return [e for e in memory if is_sensitive(e.action) and e.source != "seed"]
