"""
Token and cost accounting, plus a budget circuit breaker.

An LLM bill is a function of tokens, and tokens are a function of what users
type and what the model decides to say. Neither is under your control, which
is why OWASP lists "Unbounded Consumption" (LLM10:2025) as a top-10 risk for
LLM applications: a loop, a verbose prompt change or a hostile user can turn a
fixed budget into an open-ended one.

Three pieces:

* `count_tokens`  -- an offline approximation of a tokenizer.
* `CostLedger`    -- records every model call: who, which model, how many
                     tokens, how many dollars. Produces the dashboard line.
* `BudgetBreaker` -- refuses new calls once a spending cap is reached, and
                     refuses any call whose worst-case cost would cross it.

Money is held as `decimal.Decimal`, never float. Floats cannot represent 0.1
exactly; summing thousands of tiny per-call costs in float drifts, and a
breaker that compares a drifted float to a cap can trip one call early or one
call late. With Decimal the breaker trips exactly at the cap, and the tests
check that.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal

# --------------------------------------------------------------------------- #
# ILLUSTRATIVE price table -- not any vendor's real prices.
# --------------------------------------------------------------------------- #
# USD per 1,000,000 tokens, (input, output). The numbers are made up but have a
# realistic *shape*: output tokens cost more than input tokens, and a bigger
# model costs several times more than a small one. Replace with your provider's
# published prices before trusting any dollar figure this lab prints.
PRICES_PER_MTOK: dict[str, tuple[Decimal, Decimal]] = {
    "mock-small-v1": (Decimal("0.50"), Decimal("1.50")),
    "mock-small-v2": (Decimal("0.50"), Decimal("1.50")),
    "mock-large-v1": (Decimal("3.00"), Decimal("15.00")),
}

ONE_MILLION = Decimal(1_000_000)


def count_tokens(text: str) -> int:
    """Approximate token count: about 4 characters per token for English text.

    That ratio is a widely used rule of thumb for English with modern BPE
    tokenizers. It is an approximation: code, numbers and non-English text
    tokenize very differently. For billing-grade numbers use the provider's own
    tokenizer or the usage figures returned with each API response.
    """
    return max(1, math.ceil(len(text) / 4))


def price_of(model: str, input_tokens: int, output_tokens: int) -> Decimal:
    """Exact cost in USD of one call."""
    if model not in PRICES_PER_MTOK:
        raise KeyError(f"no price for model {model!r}; add it to PRICES_PER_MTOK")
    p_in, p_out = PRICES_PER_MTOK[model]
    return (Decimal(input_tokens) * p_in + Decimal(output_tokens) * p_out) / ONE_MILLION


@dataclass(frozen=True)
class Usage:
    request_id: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: Decimal


class CostLedger:
    """Every model call, priced. The source of truth for the cost dashboard."""

    def __init__(self) -> None:
        self.entries: list[Usage] = []

    def record(self, request_id: str, model: str, input_tokens: int,
               output_tokens: int) -> Usage:
        u = Usage(request_id, model, input_tokens, output_tokens,
                  price_of(model, input_tokens, output_tokens))
        self.entries.append(u)
        return u

    @property
    def total_usd(self) -> Decimal:
        return sum((u.cost_usd for u in self.entries), Decimal(0))

    @property
    def input_tokens(self) -> int:
        return sum(u.input_tokens for u in self.entries)

    @property
    def output_tokens(self) -> int:
        return sum(u.output_tokens for u in self.entries)

    def cost_of_request(self, request_id: str) -> Decimal:
        """A request can make several model calls (e.g. before and after a tool)."""
        return sum((u.cost_usd for u in self.entries if u.request_id == request_id), Decimal(0))

    def by_model(self) -> dict[str, Decimal]:
        out: dict[str, Decimal] = {}
        for u in self.entries:
            out[u.model] = out.get(u.model, Decimal(0)) + u.cost_usd
        return out

    def dashboard_line(self, cap_usd: Decimal | None = None, requests: int | None = None,
                       blocked: int | None = None) -> str:
        """One line you would put on a wall screen."""
        parts = []
        if requests is not None:
            parts.append(f"requests {requests}")
        if blocked is not None:
            parts.append(f"blocked {blocked}")
        parts.append(f"model calls {len(self.entries)}")
        parts.append(f"tokens in/out {self.input_tokens}/{self.output_tokens}")
        spend = f"spend ${self.total_usd:.6f}"
        if cap_usd is not None:
            pct = (self.total_usd / cap_usd * 100) if cap_usd else Decimal(0)
            spend += f" of ${cap_usd:.6f} cap ({pct:.1f}%)"
        parts.append(spend)
        return " | ".join(parts)


class BudgetBreaker:
    """A circuit breaker on spend.

    Two rules, checked before every model call:

    1. Tripped: once `spent >= cap`, every further call is refused until an
       operator calls `reset()`. Tripping is sticky on purpose -- a breaker that
       quietly re-closes invites a retry storm.
    2. Reservation: a call whose *worst-case* cost (its input tokens plus the
       maximum output tokens it is allowed) would push spend over the cap is
       refused, and that refusal trips the breaker too. This is what keeps
       spend from overshooting the cap by one expensive call.

    After the call, `charge()` adds the *actual* cost.
    """

    def __init__(self, cap_usd: Decimal | str | float):
        self.cap = Decimal(str(cap_usd))
        self.spent = Decimal(0)
        self.tripped = False

    def allow(self, estimate_usd: Decimal = Decimal(0)) -> tuple[bool, str]:
        if self.tripped:
            return False, f"budget breaker tripped: spent ${self.spent:.6f} of ${self.cap:.6f}"
        if self.spent + estimate_usd > self.cap:
            # Refusing on reservation also opens the breaker: the budget is,
            # for practical purposes, used up.
            self.tripped = True
            return False, (f"worst-case cost ${estimate_usd:.6f} would exceed cap "
                           f"(spent ${self.spent:.6f} of ${self.cap:.6f})")
        return True, "ok"

    def charge(self, cost_usd: Decimal) -> None:
        self.spent += cost_usd
        if self.spent >= self.cap:
            self.tripped = True

    def reset(self) -> None:
        self.spent = Decimal(0)
        self.tripped = False
