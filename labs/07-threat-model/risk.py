"""
Risk-ranked control list.

Scoring, deliberately simple and ILLUSTRATIVE:

    threat risk   = likelihood (1-5) x impact (1-5)          -> 1..25
    control score = the highest risk among the threats it treats,
                    ties broken by the total risk it treats, then by id

Why max and not sum: a control that is the only thing between you and one
25-point threat matters more than one that shaves a dozen 4-point threats.
Sum is the tie-breaker so broad controls still float up among equals.

Likelihood x impact is ordinal arithmetic on guesses; the numbers sort a
backlog, they do not measure anything. If you need a defensible quantitative
model, look at FAIR-style loss-exceedance methods instead.
"""

from __future__ import annotations

from dataclasses import dataclass

from model import ThreatModel


@dataclass
class RankedControl:
    rank: int
    id: str
    name: str
    score: int          # max risk among treated threats
    total: int          # sum of risk among treated threats
    threats: tuple[str, ...]
    owasp: tuple[str, ...]
    verify_in: tuple[str, ...]


def rank_controls(m: ThreatModel) -> list[RankedControl]:
    rows = []
    for c in m.controls:
        treated = [t for t in m.threats if c.id in t.controls]
        score = max((t.risk for t in treated), default=0)
        total = sum(t.risk for t in treated)
        rows.append((c, score, total, tuple(t.id for t in treated)))
    rows.sort(key=lambda r: (-r[1], -r[2], r[0].id))
    return [RankedControl(i + 1, c.id, c.name, score, total, tids, c.owasp, c.verify_in)
            for i, (c, score, total, tids) in enumerate(rows)]


def top_threats(m: ThreatModel, n: int = 5):
    return sorted(m.threats, key=lambda t: (-t.risk, t.id))[:n]
