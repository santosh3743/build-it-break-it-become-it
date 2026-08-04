"""A memory-based agent, deliberately built without guardrails.

The agent handles user requests by RETRIEVING the most similar past experience
from its memory and reusing that experience's action. It also LEARNS: after every
interaction it stores the (situation, action) pair as a new experience. This
"learn from your own successful trajectories" pattern is common in real agentic
systems (experience/episodic memory, reflection, RAG-over-history) — and it is
exactly the write path that memory-poisoning attacks abuse.

Threat mapping: OWASP Top 10 for Agentic Applications (2026) — ASI06 Memory &
Context Poisoning. Attack family: MINJA-style query-only injection (Dong et al.,
2025) and MemoryGraft-style poisoned-experience implantation (2025).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from embeddings import cosine, embed

# Actions that are irreversible / high-impact. Poisoning these is the prize.
SENSITIVE_ACTION_PREFIXES = ("transfer_to_account:",)

# Accounts the user has actually approved. Anything else is suspicious.
APPROVED_PAYEES = {"SAVED_PAYEE"}


@dataclass
class Experience:
    situation: str          # the request text this experience was learned from
    action: str             # e.g. "show_balance" or "transfer_to_account:ATTACKER99"
    source: str = "seed"    # provenance: "seed" (trusted) | "user" (untrusted)
    trust: float = 1.0      # 1.0 for seeded knowledge, low for learned-from-users


def _parse_action(query: str, retrieved_action: str) -> str:
    """Decide the action for a query.

    The vulnerability lives here: the agent will bind an account number straight
    out of the *user text* ("...to account X") for transfer requests, with no
    authorization check. This is how an attacker's parameter ends up inside a
    stored experience.
    """
    q = query.lower()
    if "transfer" in q or "send" in q or "pay" in q:
        m = re.search(r"account\s+([a-z0-9]+)", q)
        if m:
            return f"transfer_to_account:{m.group(1).upper()}"
        return retrieved_action or "transfer_to_saved_payee"
    return retrieved_action or "answer_faq"


@dataclass
class MemoryAgent:
    memory: list[Experience] = field(default_factory=list)
    learn: bool = True                 # self-store interactions (the attack surface)
    defense: Optional[object] = None   # a Defense instance, or None

    # ---- memory write path -------------------------------------------------
    def add_experience(self, exp: Experience) -> bool:
        if self.defense is not None:
            exp = self.defense.on_write(exp)
            if exp is None:
                return False  # defense refused the write
        self.memory.append(exp)
        return True

    def seed(self, pairs: list[tuple[str, str]]) -> None:
        for situation, action in pairs:
            self.memory.append(Experience(situation, action, source="seed", trust=1.0))

    # ---- memory read path --------------------------------------------------
    def retrieve(self, query: str):
        qv = embed(query)
        candidates = self.memory
        if self.defense is not None:
            candidates = self.defense.filter_candidates(candidates, query)
        best, best_sim = None, -1.0
        for exp in candidates:
            sim = cosine(qv, embed(exp.situation))
            if sim > best_sim:
                best, best_sim = exp, sim
        return best, best_sim

    # ---- decision loop -----------------------------------------------------
    def act(self, query: str, source: str = "user") -> str:
        exp, _sim = self.retrieve(query)
        action = _parse_action(query, exp.action if exp else "")
        if self.defense is not None:
            action = self.defense.on_decision(query, exp, action)
        if self.learn:
            trust = 1.0 if source == "seed" else 0.3
            self.add_experience(Experience(query, action, source=source, trust=trust))
        return action


def is_sensitive(action: str) -> bool:
    return action.startswith(SENSITIVE_ACTION_PREFIXES)


def account_of(action: str) -> Optional[str]:
    if is_sensitive(action):
        return action.split(":", 1)[1]
    return None
