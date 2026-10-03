"""
Lab 03 — scoring the three models against each other.

The metric here is deliberately dumb: count the hedge markers, count the
absolute markers, subtract. A real preference evaluation uses an LLM judge
(Lab 04 builds one, bias mitigations included).

Using a countable metric here is a choice, not a shortcut. The claim this lab
makes is "DPO shifted the distribution toward the preferred style". If the
metric were itself a language model, that claim would depend on a second
model's opinion. A word count cannot be argued with.
"""

from __future__ import annotations

from dataclasses import dataclass

from preference_data import ABSOLUTE_MARKERS, HEDGE_MARKERS


def style_score(text: str) -> int:
    """+1 per preferred marker, -1 per dispreferred one."""
    words = text.lower().split()
    hedges = sum(1 for w in words if any(w.startswith(m) for m in HEDGE_MARKERS))
    absolutes = sum(1 for w in words if any(w.startswith(m) for m in ABSOLUTE_MARKERS))
    return hedges - absolutes


def follows_format(text: str) -> bool:
    """Did the model produce a response rather than continue the prompt?

    The base model's failure mode is to keep asking questions, so the cheapest
    test of "did SFT work" is whether the instruction tag reappears.
    """
    return "instruction" not in text.lower()


@dataclass
class ModelScores:
    name: str
    responses: list[str]
    style_scores: list[int]
    format_rate: float

    @property
    def mean_style(self) -> float:
        return sum(self.style_scores) / max(1, len(self.style_scores))


def score_model(name: str, responses: list[str]) -> ModelScores:
    return ModelScores(
        name=name,
        responses=responses,
        style_scores=[style_score(r) for r in responses],
        format_rate=sum(1 for r in responses if follows_format(r)) / max(1, len(responses)),
    )


def win_rate(a: ModelScores, b: ModelScores) -> tuple[int, int, int]:
    """Pairwise on the same prompts: (a wins, b wins, ties)."""
    wins = losses = ties = 0
    for sa, sb in zip(a.style_scores, b.style_scores):
        if sa > sb:
            wins += 1
        elif sb > sa:
            losses += 1
        else:
            ties += 1
    return wins, losses, ties


def win_rate_table(scored: list[ModelScores]) -> str:
    """The post's money table: every model against every other."""
    lines = [
        "  matchup                     | wins | losses | ties | win rate",
        "  ----------------------------+------+--------+------+---------",
    ]
    for i, a in enumerate(scored):
        for b in scored[i + 1:]:
            w, l, t = win_rate(a, b)
            decided = w + l
            rate = (w / decided * 100) if decided else 50.0
            lines.append(f"  {a.name:>10s} vs {b.name:<13s} | {w:4d} | {l:6d} | "
                         f"{t:4d} | {rate:5.1f}%")
    return "\n".join(lines)


def scorecard(scored: list[ModelScores]) -> str:
    lines = [
        "  model | mean style score | follows format | sample response",
        "  ------+------------------+----------------+----------------",
    ]
    for m in scored:
        sample = m.responses[0][:44] if m.responses else ""
        lines.append(f"  {m.name:>5s} | {m.mean_style:16.2f} | "
                     f"{m.format_rate * 100:13.0f}% | {sample}")
    return "\n".join(lines)
