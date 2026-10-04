"""
The control checklist as an importable API.

Post 17's capstone red-team is graded against this checklist: for every
control, run the Act II attack that targets it, and tick the box only when
the attack is driven to ~0% success. Two ways to consume it:

    # From Python (the capstone loads this lab by file path, like Lab 03 does):
    checklist = load()                      # list[ChecklistItem], risk-ranked
    open_items = [i for i in checklist if not i.verified]

    # From the generated markdown, after someone ticked boxes by hand:
    ticked = parse_controls_md("out/controls.md")   # {"C02": True, ...}
    checklist = load(verified=ticked)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from risk import rank_controls

HERE = os.path.dirname(os.path.abspath(__file__))
CONTROLS_MD = os.path.join(HERE, "out", "controls.md")


@dataclass
class ChecklistItem:
    rank: int
    id: str
    name: str
    description: str
    score: int
    threats: list[str]
    owasp: list[str]
    verify_in: list[str]
    verified: bool = False


def checklist_from_model(m, verified: dict[str, bool] | None = None) -> list[ChecklistItem]:
    verified = verified or {}
    items = []
    for r in rank_controls(m):
        c = m.control(r.id)
        items.append(ChecklistItem(r.rank, r.id, r.name, c.description, r.score,
                                   list(r.threats), list(r.owasp), list(r.verify_in),
                                   bool(verified.get(r.id, False))))
    return items


def load(verified: dict[str, bool] | None = None) -> list[ChecklistItem]:
    """The checklist for the complete reference threat model."""
    from reference_model import build_complete_model
    return checklist_from_model(build_complete_model(), verified)


_BOX = re.compile(r"^- \[( |x|X)\] \*\*(C\d+)\*\*")


def parse_controls_md(path: str = CONTROLS_MD) -> dict[str, bool]:
    """Read tick state back out of a controls.md checklist."""
    state: dict[str, bool] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            m = _BOX.match(line.strip())
            if m:
                state[m.group(2)] = m.group(1).lower() == "x"
    return state


def coverage(items: list[ChecklistItem]) -> float:
    """Fraction of controls verified by an attack. 0.0 straight out of Lab 07."""
    return sum(i.verified for i in items) / len(items) if items else 0.0
