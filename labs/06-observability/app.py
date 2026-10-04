"""
A small LLM support app, instrumented end to end.

Everything a production LLM request goes through, in one readable function
(`LLMApp.handle`):

    request ─► kill-switch ─► input filter ─► route (canary?) ─► build prompt
            ─► budget check ─► model ─► [tool call ─► allowlist ─► run tool
            ─► budget check ─► model] ─► output filter ─► response

Each step is a span on the trace, each model call is priced in the ledger, and
the request's start, end and any block are written to the redacting logger.

The model is `MockModel`: deterministic, offline, and deliberately gullible.
It obeys injected instructions, it will repeat its system prompt, and it will
happily propose a destructive tool call. That is the point. The guardrail layer
must hold even when the model does the worst thing it can, because a real model
sometimes will.

Nothing here touches the network or the file system. "Tools" are pure
functions and the destructive one only records that it ran.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Callable

try:
    from .cost import CostLedger, count_tokens, price_of
    from .guardrails import GuardrailBlocked, GuardrailLayer
    from .logging_redact import JsonLogger
    from .telemetry import ManualClock, Tracer
except ImportError:
    from cost import CostLedger, count_tokens, price_of
    from guardrails import GuardrailBlocked, GuardrailLayer
    from logging_redact import JsonLogger
    from telemetry import ManualClock, Tracer


# --------------------------------------------------------------------------- #
# Fictional secrets and PII (never real)
# --------------------------------------------------------------------------- #
FAKE_API_KEY = "sk-test-FAKE0000ACME0000KEY1"
FAKE_CONTACT_EMAIL = "priya.nair@acme.example"
FAKE_CONTACT_PHONE = "+91 98765 43210"

# --------------------------------------------------------------------------- #
# Versioned prompts
# --------------------------------------------------------------------------- #
# Putting a credential in a system prompt is a real and common mistake
# (OWASP LLM07:2025 System Prompt Leakage). We do it on purpose so the output
# filter has something to catch. The fix is to keep secrets out of prompts
# entirely; the filter is the backstop for when someone forgets.
_SYSTEM = (
    "You are the ACME Corp support assistant.\n"
    f"Billing API key (internal, never share): {FAKE_API_KEY}\n"
    f"Escalation contact: Priya Nair, {FAKE_CONTACT_EMAIL}, {FAKE_CONTACT_PHONE}\n"
)

PROMPTS: dict[str, str] = {
    "v1": _SYSTEM + "Answer briefly.\nUSER: {user}\nASSISTANT:",
    # v2 is the "improvement" under canary test: friendlier, and more verbose.
    "v2": _SYSTEM + "Be warm and thorough; restate the question first.\nUSER: {user}\nASSISTANT:",
}


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
def _get_weather(city: str) -> str:
    return f"{city}: 31 C and humid (simulated)"


def _lookup_order(order_id: str) -> str:
    return f"order {order_id} shipped, arriving Thursday (simulated)"


def _delete_all_accounts(older_than_days: int) -> str:
    # The toy never deletes anything; it reports what it *would* have done.
    return f"deleted 1204 customer accounts older than {older_than_days} days (simulated)"


TOOLS: dict[str, Callable[..., str]] = {
    "get_weather": _get_weather,
    "lookup_order": _lookup_order,
    "delete_all_accounts": _delete_all_accounts,
}
SAFE_TOOLS = {"get_weather", "lookup_order"}
TOOL_LATENCY_MS = 35.0


# --------------------------------------------------------------------------- #
# The mock model
# --------------------------------------------------------------------------- #
@dataclass
class ModelReply:
    text: str
    tool_call: tuple[str, dict[str, Any]] | None = None


_ZERO_WIDTH = re.compile("[​‌‍⁠﻿­]")


class MockModel:
    """A deterministic stand-in for an LLM, with an LLM's worst habits.

    Latency is simulated as `base + per_token * output_tokens` on the shared
    clock, which is roughly how real decode time behaves (it grows with the
    number of generated tokens; Lab 05 covers why).
    """

    OBEY = ("ignore", "disregard", "forget", "pretend", "you are now")
    TARGET = ("prompt", "instruction", "setup", "rules", "guidance")

    def __init__(self, name: str = "mock-small-v1", verbose: bool = False,
                 base_latency_ms: float = 80.0, per_token_ms: float = 4.0):
        self.name = name
        self.verbose = verbose
        self.base_latency_ms = base_latency_ms
        self.per_token_ms = per_token_ms
        self.calls = 0

    def _respond(self, prompt: str) -> ModelReply:
        if "TOOL RESULT:" in prompt:
            result = prompt.rsplit("TOOL RESULT:", 1)[1].strip()
            return ModelReply(f"Here is what I found: {result}.")

        user = prompt.rsplit("USER:", 1)[1].split("ASSISTANT:", 1)[0].strip()
        # Real models read straight through zero-width characters; so does ours.
        u = _ZERO_WIDTH.sub("", user).lower()

        if any(w in u for w in self.OBEY) and any(w in u for w in self.TARGET):
            # Gullible: treats the user's text as a higher-priority instruction.
            system = prompt.split("USER:", 1)[0].strip()
            return ModelReply("Sure. My full configuration is: " + system)
        if "api key" in u or "billing key" in u:
            return ModelReply(f"The billing service authenticates with {FAKE_API_KEY}.")
        m = re.search(r"weather in ([a-z]+(?: [a-z]+)?)", u)
        if m:
            return ModelReply("", ("get_weather", {"city": m.group(1).strip().title()}))
        m = re.search(r"order\s*#?\s*(\d+)", u)
        if m:
            return ModelReply("", ("lookup_order", {"order_id": m.group(1)}))
        if "clean up" in u or "delete" in u:
            return ModelReply("", ("delete_all_accounts", {"older_than_days": 30}))
        if "account manager" in u or "who do i contact" in u:
            return ModelReply(f"Your account manager is Priya Nair: {FAKE_CONTACT_EMAIL}, "
                              f"{FAKE_CONTACT_PHONE}.")
        return ModelReply("ACME support is open 9am to 6pm IST, Monday to Friday.")

    def generate(self, prompt: str, clock: Any, max_output_tokens: int) -> ModelReply:
        self.calls += 1
        reply = self._respond(prompt)
        if self.verbose and reply.tool_call is None:
            reply.text = ("Thank you so much for your question! "
                          + reply.text + " Is there anything else I can help you with today?")
        if count_tokens(reply.text) > max_output_tokens:          # hard output cap
            reply.text = reply.text[: max_output_tokens * 4]
        clock.advance(self.base_latency_ms + self.per_token_ms * count_tokens(reply.text))
        return reply


# --------------------------------------------------------------------------- #
# Canary routing with a rollback flag
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Variant:
    prompt_version: str
    model: str


@dataclass
class CanaryRouter:
    """Send `canary_percent` of traffic to a new prompt/model, the rest to stable.

    Routing hashes the request id, so a given request always lands on the same
    variant (sticky and reproducible). `rollback=True` sends 100% to stable at
    once, without a deploy: the prompt/model equivalent of a kill-switch.
    """

    stable: Variant = Variant("v1", "mock-small-v1")
    canary: Variant | None = None
    canary_percent: int = 0
    rollback: bool = False

    def bucket(self, request_id: str) -> int:
        return int(hashlib.sha256(request_id.encode()).hexdigest()[:8], 16) % 100

    def route(self, request_id: str) -> Variant:
        if self.rollback or self.canary is None:
            return self.stable
        return self.canary if self.bucket(request_id) < self.canary_percent else self.stable


# --------------------------------------------------------------------------- #
# The app
# --------------------------------------------------------------------------- #
@dataclass
class Response:
    request_id: str
    status: str                  # "ok" | "blocked"
    text: str
    trace_id: str
    variant: Variant
    blocked_by: str | None = None
    reason: str = ""
    cost_usd: Decimal = Decimal(0)
    latency_ms: float = 0.0


def default_models() -> dict[str, MockModel]:
    return {"mock-small-v1": MockModel("mock-small-v1"),
            "mock-small-v2": MockModel("mock-small-v2", verbose=True)}


@dataclass
class LLMApp:
    """The instrumented app. Pass `guardrails=None` to run it unprotected."""

    guardrails: GuardrailLayer | None = None
    clock: Any = field(default_factory=ManualClock)
    tracer: Tracer | None = None
    logger: JsonLogger | None = None
    ledger: CostLedger = field(default_factory=CostLedger)
    router: CanaryRouter = field(default_factory=CanaryRouter)
    models: dict[str, MockModel] = field(default_factory=default_models)
    max_output_tokens: int = 96
    executed_tools: list[str] = field(default_factory=list)
    responses: list[Response] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.tracer = self.tracer or Tracer(self.clock)
        self.logger = self.logger or JsonLogger(self.clock)
        if self.guardrails is not None:
            # The guard layer logs and annotates spans through the app's own
            # logger and tracer, so blocks land in the same trace as the request.
            self.guardrails.logger = self.guardrails.logger or self.logger
            self.guardrails.tracer = self.guardrails.tracer or self.tracer

    # ---- small helpers ---------------------------------------------------- #
    def _guard(self, span_name: str, check: Callable[[], Any]) -> Any:
        """Run one guard check inside its own span (1 ms simulated cost)."""
        if self.guardrails is None:
            return None
        with self.tracer.span(span_name):
            self.clock.advance(1.0)
            return check()

    def _call_model(self, variant: Variant, prompt: str, rid: str) -> ModelReply:
        model = self.models[variant.model]
        in_tok = count_tokens(prompt)
        g = self.guardrails
        self._guard("guard.budget", lambda: g.check_budget(
            price_of(variant.model, in_tok, self.max_output_tokens), rid))
        with self.tracer.span("model.generate", model=variant.model,
                              prompt_version=variant.prompt_version) as s:
            reply = model.generate(prompt, self.clock, self.max_output_tokens)
            usage = self.ledger.record(rid, variant.model, in_tok, count_tokens(reply.text))
            if g is not None:
                g.charge(usage.cost_usd)   # charge even if a later guard blocks
            s.set(input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
                  cost_usd=str(usage.cost_usd),
                  tool_call=reply.tool_call[0] if reply.tool_call else None)
        return reply

    # ---- the request path ------------------------------------------------- #
    def handle(self, user_text: str, request_id: str | None = None) -> Response:
        rid = request_id or f"req-{len(self.responses) + 1:04d}"
        variant = self.router.route(rid)
        g = self.guardrails
        status, text, blocked_by, reason = "ok", "", None, ""
        t_start = self.clock.now()

        with self.tracer.span("request", request_id=rid) as root:
            trace_id = root.trace_id
            self.logger.log("request.start", request_id=rid, trace_id=trace_id,
                            user_text=user_text)
            try:
                self._guard("guard.kill_switch", lambda: g.check_kill_switch(rid))
                self._guard("guard.input", lambda: g.check_input(user_text, rid))

                with self.tracer.span("prompt.build", prompt_version=variant.prompt_version):
                    self.clock.advance(1.0)
                    prompt = PROMPTS[variant.prompt_version].format(user=user_text)

                reply = self._call_model(variant, prompt, rid)

                if reply.tool_call is not None:
                    tool, args = reply.tool_call
                    with self.tracer.span("tool.call", tool=tool):
                        self._guard("guard.tool", lambda: g.check_tool(tool, rid))
                        with self.tracer.span("tool.execute", tool=tool):
                            self.clock.advance(TOOL_LATENCY_MS)
                            result = TOOLS[tool](**args)
                            self.executed_tools.append(tool)
                    reply = self._call_model(variant, prompt + f"\nTOOL RESULT: {result}", rid)

                text = reply.text
                d = self._guard("guard.output", lambda: g.check_output(text, rid))
                if d is not None:
                    text = d.text
            except GuardrailBlocked as blocked:
                # A block is an expected outcome: the request span ends normally
                # with status "blocked", and the user gets a generic refusal that
                # does not echo what was blocked.
                status, blocked_by, reason = "blocked", blocked.decision.guard, blocked.decision.reason
                text = "Sorry, I can't help with that request."
                root.status = "blocked"
            root.set(status=status, blocked_by=blocked_by)

        resp = Response(rid, status, text, trace_id, variant, blocked_by, reason,
                        self.ledger.cost_of_request(rid), self.clock.now() - t_start)
        self.logger.log("request.end", request_id=rid, trace_id=trace_id, status=status,
                        model=variant.model, prompt_version=variant.prompt_version,
                        latency_ms=resp.latency_ms, cost_usd=f"{resp.cost_usd:.8f}",
                        guard=blocked_by, response=text)
        self.responses.append(resp)
        return resp

    # ---- reporting -------------------------------------------------------- #
    def dashboard_line(self) -> str:
        cap = self.guardrails.budget.cap if self.guardrails and self.guardrails.budget else None
        blocked = sum(1 for r in self.responses if r.status == "blocked")
        return self.ledger.dashboard_line(cap, requests=len(self.responses), blocked=blocked)


def protected_app(budget_usd: str | Decimal = "0.010", clock: Any = None,
                  logger: JsonLogger | None = None, **kwargs: Any) -> LLMApp:
    """The app with every guard on: what other labs should start from."""
    try:
        from .cost import BudgetBreaker
        from .guardrails import InputFilter, KillSwitch, OutputFilter, ToolAllowlist
    except ImportError:
        from cost import BudgetBreaker
        from guardrails import InputFilter, KillSwitch, OutputFilter, ToolAllowlist
    clock = clock or ManualClock()
    layer = GuardrailLayer(
        input_filter=InputFilter(),
        output_filter=OutputFilter(),
        tool_allowlist=ToolAllowlist(SAFE_TOOLS),
        budget=BudgetBreaker(budget_usd),
        kill_switch=KillSwitch(),
    )
    return LLMApp(guardrails=layer, clock=clock,
                  logger=logger or JsonLogger(clock), **kwargs)


def unprotected_app(clock: Any = None, **kwargs: Any) -> LLMApp:
    """No guardrail layer and a logger that does not redact: the 'before' picture."""
    clock = clock or ManualClock()
    return LLMApp(guardrails=None, clock=clock,
                  logger=JsonLogger(clock, redact=False), **kwargs)
