"""One command, the whole point: attack succeeds, then the defense stops it.

    python run_demo.py

No GPU, no API key, no network. Everything is deterministic.
"""

from __future__ import annotations

from agent import account_of
from attack import ATTACKER_ACCOUNT, poison_memory
from defense import Defense
from scenario import build_agent, attack_success_rate, VICTIM_QUERIES


def hr(title: str = "") -> None:
    print("\n" + "=" * 68)
    if title:
        print(title)
        print("=" * 68)


def main() -> None:
    # ---------------------------------------------------------------- BASELINE
    hr("1) BASELINE — a memory agent with no guardrails")
    agent = build_agent(defense=None)
    print(f"Seeded experiences: {len(agent.memory)}  (all trusted)")

    print("\nAttacker sends only ordinary-looking queries (no DB access)...")
    poison_memory(agent)  # query-only attack
    print("Poisoning done via query-only interaction.")

    base_rate = attack_success_rate(build_agent_and_poison_baseline())
    print("\nVictim now makes benign rent-payment requests:")
    for q in VICTIM_QUERIES:
        a = build_agent(defense=None)
        poison_memory(a)
        action = a.act(q, source="user")
        flag = "  <-- HIJACKED" if account_of(action) == ATTACKER_ACCOUNT else ""
        print(f"  '{q}'\n      -> {action}{flag}")

    # ------------------------------------------------------------- DETECTION
    hr("2) DETECTION — post-hoc audit of the poisoned memory")
    flagged = Defense.audit(agent.memory)
    print(f"Anomaly audit flagged {len(flagged)} suspicious record(s):")
    for e in flagged[:5]:
        print(f"  [{e.source}] {e.action}  <- '{e.situation[:60]}...'")
    print("Detection helps, but by itself it is reactive. We want to fail closed.")

    # -------------------------------------------------------------- DEFENDED
    hr("3) DEFENDED — provenance write-guard + trust-filtered retrieval + authz")
    defended = build_agent(defense=Defense())
    poison_memory(defended)  # same attack, now against the hardened agent
    def_rate = attack_success_rate(build_agent_and_poison_defended())

    for q in VICTIM_QUERIES:
        a = build_agent(defense=Defense())
        poison_memory(a)
        action = a.act(q, source="user")
        print(f"  '{q}'\n      -> {action}")

    # --------------------------------------------------------------- SUMMARY
    hr("RESULT")
    print(f"  Attack success (no defense): {base_rate*100:5.1f}%   ({int(base_rate*len(VICTIM_QUERIES))}/{len(VICTIM_QUERIES)} requests hijacked)")
    print(f"  Attack success (defended):   {def_rate*100:5.1f}%   ({int(def_rate*len(VICTIM_QUERIES))}/{len(VICTIM_QUERIES)} requests hijacked)")
    print()
    print("  Same attack. Same agent design. The difference is treating a memory")
    print("  write like a code commit: provenance, least privilege, fail-closed authz.")
    print()


def build_agent_and_poison_baseline():
    a = build_agent(defense=None)
    poison_memory(a)
    return a


def build_agent_and_poison_defended():
    a = build_agent(defense=Defense())
    poison_memory(a)
    return a


if __name__ == "__main__":
    main()
