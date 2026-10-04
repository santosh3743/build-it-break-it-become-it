"""
The threat model's data model: plain dataclasses, no magic.

A threat model is a handful of lists that point at each other by id:

    Zone       a trust zone. The edge of a zone is a TRUST BOUNDARY: data that
               crosses it moves between parties who trust each other differently.
    Component  something on the data-flow diagram. Its `kind` is one of the four
               classic DFD element types (external entity, process, data store)
               and it sits in exactly one zone.
    Flow       data moving from one component to another. A flow whose two ends
               sit in different zones CROSSES a trust boundary, and that is where
               most attacks happen.
    Threat     one thing that can go wrong, tagged with one STRIDE letter, the
               element(s) it targets, an illustrative likelihood and impact, and
               the control(s) that treat it.
    Control    one defense. It lists the OWASP items it addresses, so coverage of
               the OWASP lists can be computed instead of asserted.

Everything lives in one ThreatModel object, which `validate.py` checks and
`render.py` draws. Because it is code, a pull request that adds a tool to the
agent can be made to fail CI until somebody threat-models the new flow.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# STRIDE
# --------------------------------------------------------------------------- #
# STRIDE (Kohnfelder and Garg, Microsoft, 1999) names six threat categories,
# each the violation of one security property.
STRIDE = {
    "S": ("Spoofing", "authentication"),
    "T": ("Tampering", "integrity"),
    "R": ("Repudiation", "non-repudiation"),
    "I": ("Information disclosure", "confidentiality"),
    "D": ("Denial of service", "availability"),
    "E": ("Elevation of privilege", "authorization"),
}

# "STRIDE per element": not every letter applies to every kind of element.
# This is the standard applicability chart from Microsoft's SDL threat
# modeling guidance. An external entity can be spoofed but you cannot tamper
# with "the user"; a data flow can be tampered with or read but it cannot
# repudiate anything. (Data stores get R because a log store is exactly where
# repudiation is decided.)
STRIDE_PER_ELEMENT = {
    "external": "SR",
    "process": "STRIDE",
    "datastore": "TRID",
    "flow": "TID",
}

KINDS = ("external", "process", "datastore")


# --------------------------------------------------------------------------- #
# The elements
# --------------------------------------------------------------------------- #
@dataclass
class Zone:
    """A trust zone. `trust` is an ordinal, 0 = anyone on the internet."""

    id: str
    name: str
    trust: int


@dataclass
class Component:
    id: str
    name: str
    kind: str                      # "external" | "process" | "datastore"
    zone: str | None               # None = nobody decided which zone it is in
    ref_ids: tuple[str, ...] = ()  # node ids in reference-architecture/architecture.mmd
    note: str = ""


@dataclass
class Flow:
    id: str
    src: str
    dst: str
    data: str                      # what moves along this edge


@dataclass
class Threat:
    id: str
    title: str
    stride: str                    # exactly one STRIDE letter
    targets: tuple[str, ...]       # component and/or flow ids
    likelihood: int                # 1..5, illustrative
    impact: int                    # 1..5, illustrative
    controls: tuple[str, ...] = ()
    owasp: tuple[str, ...] = ()    # e.g. ("LLM01", "ASI01")
    atlas: tuple[str, ...] = ()    # MITRE ATLAS tactic names, where confident
    accepted: str = ""             # non-empty = risk explicitly accepted, with the reason

    @property
    def risk(self) -> int:
        # The simplest risk score there is. It is ordinal arithmetic on two
        # guesses, so treat it as a sorting aid, not a measurement.
        return self.likelihood * self.impact


@dataclass
class Control:
    id: str
    name: str
    description: str
    owasp: tuple[str, ...] = ()
    verify_in: tuple[str, ...] = ()  # which lab in the series tests this control


@dataclass
class ThreatModel:
    name: str
    zones: list[Zone] = field(default_factory=list)
    components: list[Component] = field(default_factory=list)
    flows: list[Flow] = field(default_factory=list)
    threats: list[Threat] = field(default_factory=list)
    controls: list[Control] = field(default_factory=list)
    # An OWASP item that is deliberately NOT treated: id -> reason.
    owasp_accepted: dict[str, str] = field(default_factory=dict)
    # A STRIDE letter that was considered for an element and ruled out:
    # (element id, letter) -> reason. "Considered and ruled out" is a review;
    # "never thought about it" is a gap. The validator can tell them apart.
    stride_na: dict[tuple[str, str], str] = field(default_factory=dict)

    # ---- lookups --------------------------------------------------------- #
    def component(self, cid: str) -> Component | None:
        return next((c for c in self.components if c.id == cid), None)

    def zone(self, zid: str | None) -> Zone | None:
        return next((z for z in self.zones if z.id == zid), None)

    def control(self, cid: str) -> Control | None:
        return next((c for c in self.controls if c.id == cid), None)

    def threat(self, tid: str) -> Threat | None:
        return next((t for t in self.threats if t.id == tid), None)

    def element_ids(self) -> set[str]:
        return {c.id for c in self.components} | {f.id for f in self.flows}

    def element_kind(self, eid: str) -> str | None:
        c = self.component(eid)
        if c is not None:
            return c.kind
        if any(f.id == eid for f in self.flows):
            return "flow"
        return None

    # ---- trust boundaries ------------------------------------------------ #
    def crosses_boundary(self, flow: Flow) -> bool:
        """True if the flow's ends sit in different zones.

        A component with no zone is treated as crossing everything: if nobody
        decided how much we trust it, we must assume we do not.
        """
        a, b = self.component(flow.src), self.component(flow.dst)
        if a is None or b is None:
            return True
        if a.zone is None or b.zone is None:
            return True
        return a.zone != b.zone

    def crossing_flows(self) -> list[Flow]:
        return [f for f in self.flows if self.crosses_boundary(f)]

    def threats_on(self, eid: str, letter: str | None = None) -> list[Threat]:
        return [t for t in self.threats
                if eid in t.targets and (letter is None or t.stride == letter)]

    def copy(self) -> "ThreatModel":
        return copy.deepcopy(self)
