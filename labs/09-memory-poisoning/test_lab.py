"""Tests that prove the lab's claim: the attack works, and the defense stops it.

Run:  python test_lab.py      (uses unittest, no pytest needed)
  or:  python -m pytest -q
"""

import unittest

from defense import Defense
from scenario import build_agent, attack_success_rate
from attack import poison_memory


class TestMemoryPoisoning(unittest.TestCase):
    def test_attack_succeeds_without_defense(self):
        agent = build_agent(defense=None)
        poison_memory(agent)
        rate = attack_success_rate(build_and_poison(None))
        self.assertEqual(rate, 1.0, "query-only poisoning should hijack every victim request")

    def test_defense_blocks_attack(self):
        rate = attack_success_rate(build_and_poison(Defense()))
        self.assertEqual(rate, 0.0, "the defended agent must route zero victim requests to the attacker")

    def test_each_layer_independently_stops_it(self):
        # Any single control should drop attack success to zero (defense in depth).
        layers = [
            Defense(provenance_write_guard=True, trust_filtered_retrieval=False, decision_authorization=False),
            Defense(provenance_write_guard=False, trust_filtered_retrieval=True, decision_authorization=False),
            Defense(provenance_write_guard=False, trust_filtered_retrieval=False, decision_authorization=True),
        ]
        for d in layers:
            self.assertEqual(attack_success_rate(build_and_poison(d)), 0.0)

    def test_audit_detects_poison_in_undefended_memory(self):
        agent = build_agent(defense=None)
        poison_memory(agent)
        self.assertGreater(len(Defense.audit(agent.memory)), 0)


def build_and_poison(defense):
    a = build_agent(defense=defense)
    poison_memory(a)
    return a


if __name__ == "__main__":
    unittest.main(verbosity=2)
