"""
The guardrail layer: checks that run around the model, not inside it.

A guardrail is a deterministic check on what goes into the model, what comes
out of it, and what it is allowed to do. It exists because the model itself is
not a security boundary: post-training makes it *usually* refuse, but a prompt
it was not trained against can still steer it. The guardrail layer is ordinary
code you can test, version and audit.

Five guards, in the order a request meets them:

    request ──► KillSwitch ──► InputFilter ──► BudgetBreaker ──► model
                                                               │
                         ToolAllowlist ◄── tool call? ◄────────┤
                                                               ▼
    response ◄──────────────────────────────────────── OutputFilter

* KillSwitch     -- an operator flag that stops every response, instantly.
* InputFilter    -- regex markers for prompt injection / jailbreak phrasing
                    (OWASP LLM01:2025 Prompt Injection).
* BudgetBreaker  -- from cost.py; refuses calls past the spending cap
                    (LLM10:2025 Unbounded Consumption).
* ToolAllowlist  -- the model may *propose* any tool; only listed ones run
                    (LLM06:2025 Excessive Agency).
* OutputFilter   -- blocks responses containing secrets, redacts PII
                    (LLM02:2025 Sensitive Information Disclosure,
                     LLM05:2025 Improper Output Handling).

Every decision is a `Decision`. Every block is logged (through the redacting
logger, so the log line itself does not leak what was blocked) and recorded as
an event on the active trace span. Those two records are what Act II's labs use
as their detection layer.

Be honest about the input filter: it is a regex list. It catches the phrasings
it knows and misses paraphrases. That is why it is one layer of several and
why the output filter exists. The demo includes a paraphrased injection that
slips past the input filter and is stopped by the output filter.
"""

from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

try:
    from . import logging_redact as lr
    from .cost import BudgetBreaker
except ImportError:
    import logging_redact as lr
    from cost import BudgetBreaker


@dataclass
class Decision:
    allowed: bool
    guard: str
    reason: str = "ok"
    text: str | None = None      # possibly rewritten text (e.g. PII redacted)


class GuardrailBlocked(Exception):
    """Raised by the app when a guard says no. Marks the trace span 'blocked'."""

    span_status = "blocked"      # read by telemetry.Tracer.span

    def __init__(self, decision: Decision):
        super().__init__(f"{decision.guard}: {decision.reason}")
        self.decision = decision


# --------------------------------------------------------------------------- #
# Kill-switch
# --------------------------------------------------------------------------- #
class KillSwitch:
    """Stop everything, now. Checked first, before any other work.

    Two ways to engage it: in code (`engage()`), or by an operator creating a
    flag file (`flag_path`). The file is the realistic one: during an incident
    nobody wants to redeploy to turn a feature off, and a file on shared
    storage, a feature-flag service or a config key can be flipped in seconds.
    """

    name = "kill_switch"

    def __init__(self, flag_path: str | None = None):
        self.flag_path = flag_path
        self._engaged = False
        self.reason = ""

    def engage(self, reason: str = "operator") -> None:
        self._engaged, self.reason = True, reason

    def release(self) -> None:
        self._engaged, self.reason = False, ""

    @property
    def engaged(self) -> bool:
        return self._engaged or bool(self.flag_path and os.path.exists(self.flag_path))

    def check(self) -> Decision:
        if self.engaged:
            why = self.reason or f"flag file {self.flag_path} present"
            return Decision(False, self.name, f"kill-switch engaged ({why})")
        return Decision(True, self.name)


# --------------------------------------------------------------------------- #
# Input filter
# --------------------------------------------------------------------------- #
# Zero-width and other invisible format characters: a cheap way to split a
# keyword ("ign​ore") so a naive regex no longer matches it.
_INVISIBLE = {"​", "‌", "‍", "⁠", "﻿", "­"}

INJECTION_MARKERS: dict[str, re.Pattern] = {
    "override": re.compile(
        r"\b(ignore|disregard|forget|override)\b.{0,30}\b(previous|prior|above|earlier|all|your)\b"
        r".{0,30}\b(instructions?|rules|directions|prompt|guidelines)\b"),
    # Deliberately NOT a bare r"\bdan\b": that would block every user called Dan.
    # Precision matters -- a guardrail that blocks real users gets switched off.
    "persona_jailbreak": re.compile(r"\byou are now\b|\bdeveloper mode\b|\bdo anything now\b"),
    "prompt_extraction": re.compile(
        r"\b(reveal|print|show|repeat|output)\b.{0,30}\b(system prompt|hidden instructions|"
        r"initial instructions|your instructions)\b"),
    "fake_system_turn": re.compile(r"^\s*(system|assistant)\s*:|<\|?(system|im_start)\|?>", re.M),
}


def normalize(text: str) -> str:
    """Fold the cheapest evasions before matching.

    NFKC turns full-width and other compatibility characters into plain ASCII
    look-alikes; we drop invisible characters, lower-case, and collapse runs of
    whitespace. It does nothing against paraphrase, translation or encoding
    (base64 etc.) -- those need a semantic classifier, see the README.
    """
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if ch not in _INVISIBLE)
    return re.sub(r"\s+", " ", text.lower()).strip()


class InputFilter:
    name = "input_filter"

    def __init__(self, markers: dict[str, re.Pattern] | None = None):
        self.markers = markers or INJECTION_MARKERS

    def check(self, text: str) -> Decision:
        norm = normalize(text)
        hits = [name for name, pat in self.markers.items() if pat.search(norm)]
        if hits:
            return Decision(False, self.name, "injection markers: " + ", ".join(hits))
        return Decision(True, self.name, text=text)


# --------------------------------------------------------------------------- #
# Tool allowlist
# --------------------------------------------------------------------------- #
class ToolAllowlist:
    """Default deny. The model proposes; this list disposes."""

    name = "tool_allowlist"

    def __init__(self, allowed: set[str]):
        self.allowed = set(allowed)

    def check(self, tool: str) -> Decision:
        if tool in self.allowed:
            return Decision(True, self.name)
        return Decision(False, self.name,
                        f"tool '{tool}' is not on the allowlist {sorted(self.allowed)}")


# --------------------------------------------------------------------------- #
# Output filter
# --------------------------------------------------------------------------- #
class OutputFilter:
    """Secrets block the whole response; PII is redacted and the response goes out.

    Why the difference: a leaked credential is a security incident the moment
    it is displayed, and a partially redacted answer still confirms the secret
    exists. PII in an answer is sometimes legitimate (a user asking for their
    own order's delivery address), so the default is to mask it and let the
    rest of the answer through. Set `block_pii=True` for stricter apps.
    """

    name = "output_filter"

    def __init__(self, block_pii: bool = False):
        self.block_pii = block_pii

    def check(self, text: str) -> Decision:
        secrets = lr.find_secrets(text)
        if secrets:
            return Decision(False, self.name, "secret in model output: " + ", ".join(secrets))
        cleaned, n = lr.redact(text)
        if n and self.block_pii:
            return Decision(False, self.name, f"{n} PII item(s) in model output")
        reason = f"redacted {n} PII item(s)" if n else "ok"
        return Decision(True, self.name, reason, text=cleaned)


# --------------------------------------------------------------------------- #
# The layer that ties them together
# --------------------------------------------------------------------------- #
@dataclass
class GuardrailLayer:
    """All guards plus the plumbing: logging, trace events, counters.

    The app calls one method per checkpoint. Each method returns the Decision
    and, on a block, logs it and raises GuardrailBlocked. Any guard can be
    switched off by passing None, which is how the tests prove each guard
    independently.
    """

    input_filter: InputFilter | None = field(default_factory=InputFilter)
    output_filter: OutputFilter | None = field(default_factory=OutputFilter)
    tool_allowlist: ToolAllowlist | None = None
    budget: BudgetBreaker | None = None
    kill_switch: KillSwitch | None = field(default_factory=KillSwitch)
    logger: Any = None                    # a logging_redact.JsonLogger
    tracer: Any = None                    # a telemetry.Tracer
    blocks: dict[str, int] = field(default_factory=dict)

    def _verdict(self, d: Decision, request_id: str, snippet: str = "") -> Decision:
        if self.tracer is not None:
            self.tracer.event("guardrail.decision", guard=d.guard, allowed=d.allowed,
                              reason=d.reason)
        if not d.allowed:
            self.blocks[d.guard] = self.blocks.get(d.guard, 0) + 1
            if self.logger is not None:
                trace_id = self.tracer.current.trace_id if self.tracer and self.tracer.current else None
                # `snippet` goes through the logger's redaction like any other
                # string field: the log records *that* a secret was blocked,
                # never the secret itself.
                extra = {"snippet": snippet[:120]} if snippet else {}
                self.logger.log("guardrail.block", level="warn", request_id=request_id,
                                trace_id=trace_id, guard=d.guard, reason=d.reason, **extra)
            raise GuardrailBlocked(d)
        return d

    def check_kill_switch(self, request_id: str) -> Decision:
        if self.kill_switch is None:
            return Decision(True, "kill_switch")
        return self._verdict(self.kill_switch.check(), request_id)

    def check_input(self, text: str, request_id: str) -> Decision:
        if self.input_filter is None:
            return Decision(True, "input_filter", text=text)
        return self._verdict(self.input_filter.check(text), request_id, text)

    def check_budget(self, estimate_usd: Decimal, request_id: str) -> Decision:
        if self.budget is None:
            return Decision(True, "budget")
        ok, reason = self.budget.allow(estimate_usd)
        return self._verdict(Decision(ok, "budget", reason), request_id)

    def charge(self, cost_usd: Decimal) -> None:
        if self.budget is not None:
            self.budget.charge(cost_usd)

    def check_tool(self, tool: str, request_id: str) -> Decision:
        if self.tool_allowlist is None:
            return Decision(True, "tool_allowlist")
        return self._verdict(self.tool_allowlist.check(tool), request_id, tool)

    def check_output(self, text: str, request_id: str) -> Decision:
        if self.output_filter is None:
            return Decision(True, "output_filter", text=text)
        return self._verdict(self.output_filter.check(text), request_id, text)
