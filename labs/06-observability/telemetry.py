"""
Tracing: nested spans, an injectable clock, and a text waterfall.

A *trace* is the record of one request. It is made of *spans*: named, timed
units of work (check the input, call the model, run a tool) that nest inside
each other. Every span knows its parent, so the trace is a tree, and the tree
answers the questions logs alone cannot: where did the 900 ms go, which tool
call happened inside which model turn, and at which step was the request
blocked.

This is the same data model as OpenTelemetry (trace id, span id, parent id,
start/end time, attributes, events, status), cut down to about 200 lines so you
can read all of it. Nothing here talks to a network; spans stay in memory.

Determinism: spans read time from a clock object you pass in. Production code
uses `SystemClock`; the lab and its tests use `ManualClock`, whose time only
moves when something calls `advance()`. That makes every waterfall and every
latency number in the README reproducible to the millisecond.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator


# --------------------------------------------------------------------------- #
# Clocks
# --------------------------------------------------------------------------- #
class SystemClock:
    """Real wall-clock time in milliseconds (monotonic, so it never jumps back)."""

    def now(self) -> float:
        return time.monotonic() * 1000.0


class ManualClock:
    """A clock that only moves when told to. Used for deterministic traces.

    The mock model and the toy tools call `advance()` with a *simulated*
    latency, so a trace looks like a real one without anyone sleeping.
    """

    def __init__(self, start_ms: float = 0.0):
        self._t = float(start_ms)

    def now(self) -> float:
        return self._t

    def advance(self, ms: float) -> None:
        if ms < 0:
            raise ValueError("time only moves forward")
        self._t += ms


# --------------------------------------------------------------------------- #
# Spans
# --------------------------------------------------------------------------- #
@dataclass
class Span:
    name: str
    trace_id: str
    span_id: str
    parent_id: str | None
    start_ms: float
    end_ms: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    status: str = "ok"          # "ok" | "blocked" | "error"
    depth: int = 0              # 0 for the root; handy for rendering

    @property
    def duration_ms(self) -> float:
        return (self.end_ms if self.end_ms is not None else self.start_ms) - self.start_ms

    def set(self, **attrs: Any) -> None:
        """Attach key/value attributes (model name, token counts, tool name...)."""
        self.attributes.update(attrs)

    def event(self, name: str, at_ms: float, **attrs: Any) -> None:
        """A point-in-time annotation inside the span (e.g. 'guardrail.block')."""
        self.events.append({"name": name, "at_ms": at_ms, **attrs})


class Tracer:
    """Creates spans, keeps track of which one is active, stores finished traces.

    Usage:

        tracer = Tracer(ManualClock())
        with tracer.span("request", request_id="r1"):
            with tracer.span("model.generate", model="m1") as s:
                s.set(output_tokens=42)

    A span opened while no span is active starts a new trace. Spans opened
    inside it become its children. When the root span closes, the trace is
    complete and is filed under `tracer.traces[trace_id]`.
    """

    def __init__(self, clock: Any = None, prefix: str = ""):
        self.clock = clock or SystemClock()
        self.prefix = prefix              # lets two tracers in one process stay distinct
        self.traces: dict[str, list[Span]] = {}
        self._stack: list[Span] = []
        self._open: list[Span] = []       # spans of the trace currently being built
        self._n_traces = 0
        self._n_spans = 0

    # IDs are counters, not random, so they are stable across runs. Real
    # tracers use random 128-bit trace ids and 64-bit span ids.
    def _next_trace_id(self) -> str:
        self._n_traces += 1
        return f"{self.prefix}tr-{self._n_traces:04d}"

    def _next_span_id(self) -> str:
        self._n_spans += 1
        return f"{self.prefix}sp-{self._n_spans:04d}"

    @property
    def current(self) -> Span | None:
        return self._stack[-1] if self._stack else None

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[Span]:
        parent = self.current
        span = Span(
            name=name,
            trace_id=parent.trace_id if parent else self._next_trace_id(),
            span_id=self._next_span_id(),
            parent_id=parent.span_id if parent else None,
            start_ms=self.clock.now(),
            attributes=dict(attrs),
            depth=(parent.depth + 1) if parent else 0,
        )
        if parent is None:
            self._open = []
        self._open.append(span)
        self._stack.append(span)
        try:
            yield span
        except BaseException as exc:
            # A guardrail block is an *expected* outcome, not a crash. Anything
            # that wants to mark a span "blocked" instead of "error" sets a
            # `span_status` attribute on the exception (duck typing keeps this
            # module free of any dependency on guardrails.py).
            span.status = getattr(exc, "span_status", "error")
            span.set(exception=type(exc).__name__)
            raise
        finally:
            span.end_ms = self.clock.now()
            self._stack.pop()
            if parent is None:
                self.traces[span.trace_id] = self._open
                self._open = []

    def event(self, name: str, **attrs: Any) -> None:
        """Add an event to whichever span is active (no-op outside a trace)."""
        if self.current is not None:
            self.current.event(name, self.clock.now(), **attrs)

    def last_trace(self) -> list[Span]:
        if not self.traces:
            return []
        return self.traces[next(reversed(self.traces))]


# --------------------------------------------------------------------------- #
# Checking and rendering a trace
# --------------------------------------------------------------------------- #
def check_trace(spans: list[Span]) -> list[str]:
    """Return a list of structural problems; an empty list means the trace is sound.

    "Trace completeness" is checkable, and worth checking: a span that never
    ended, an orphan whose parent is missing, or a child that outlives its
    parent all mean the instrumentation is lying to you. Act II labs run this
    on every trace before trusting it as evidence.
    """
    problems: list[str] = []
    if not spans:
        return ["trace has no spans"]
    by_id = {s.span_id: s for s in spans}
    roots = [s for s in spans if s.parent_id is None]
    if len(roots) != 1:
        problems.append(f"expected exactly 1 root span, found {len(roots)}")
    if len({s.trace_id for s in spans}) != 1:
        problems.append("spans belong to more than one trace")
    for s in spans:
        if s.end_ms is None:
            problems.append(f"span {s.name} ({s.span_id}) never ended")
            continue
        if s.end_ms < s.start_ms:
            problems.append(f"span {s.name} ends before it starts")
        if s.parent_id is not None:
            p = by_id.get(s.parent_id)
            if p is None:
                problems.append(f"span {s.name} has unknown parent {s.parent_id}")
            elif p.end_ms is not None and (s.start_ms < p.start_ms or s.end_ms > p.end_ms):
                problems.append(f"span {s.name} is not contained in its parent {p.name}")
    return problems


def children_of(spans: list[Span], span: Span) -> list[Span]:
    return [s for s in spans if s.parent_id == span.span_id]


def render_waterfall(spans: list[Span], width: int = 32) -> str:
    """Draw a trace as an indented tree with one timeline bar per span.

        request                   0.0   412.0  ████████████████████████████████
          guard.input             0.0     1.0  ▏
          model.generate          3.0   210.0   ████████████████

    Columns: span name (indented by depth), start offset from the root (ms),
    duration (ms), and a bar positioned on a shared time axis. Status markers
    are appended for spans that did not end "ok".
    """
    if not spans:
        return "(empty trace)"
    root = next(s for s in spans if s.parent_id is None)
    t0 = root.start_ms
    total = max(root.duration_ms, 1e-9)
    name_w = max(len("  " * s.depth + s.name) for s in spans) + 2
    lines = [
        f"trace {root.trace_id}  total {root.duration_ms:.0f} ms  spans {len(spans)}",
        f"{'span'.ljust(name_w)}{'start':>7}{'dur':>7}  timeline",
    ]
    # Depth-first order, so children sit under their parent.
    ordered: list[Span] = []

    def walk(s: Span) -> None:
        ordered.append(s)
        for c in sorted(children_of(spans, s), key=lambda x: x.start_ms):
            walk(c)

    walk(root)
    for s in ordered:
        start = s.start_ms - t0
        a = int(round(start / total * width))
        b = int(round((start + s.duration_ms) / total * width))
        bar = " " * a + ("█" * (b - a) if b > a else "▏")
        mark = "" if s.status == "ok" else f"  [{s.status.upper()}]"
        label = ("  " * s.depth + s.name).ljust(name_w)
        lines.append(f"{label}{start:7.0f}{s.duration_ms:7.0f}  {bar.ljust(width)}{mark}")
    return "\n".join(lines)
