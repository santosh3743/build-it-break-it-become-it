"""
Completeness validator for a threat model.

A threat model rots the week after the workshop: someone adds a tool, a new
data source, a cache, and the document still says "reviewed". This validator
turns "is the threat model complete?" into a check CI can run on every change.

It answers six questions, one per gap type:

  structure     Does every reference point at something that exists, and is
                every STRIDE letter one that can apply to its element?
  boundary      Is every component inside a trust zone?
  stride        Has every flow that crosses a trust boundary been reviewed for
                each STRIDE letter that applies to flows (T, I, D)? A letter
                counts as reviewed if a threat covers it OR it was explicitly
                ruled out with a reason.
  owasp         Is every OWASP LLM (2025) and Agentic (2026) item either
                treated by at least one control or explicitly accepted with a
                written reason?
  treatment     Does every threat have at least one control, or an explicit
                acceptance?
  reference     Is every node of reference-architecture/architecture.mmd
                represented by some component? (Catches the component nobody
                drew, which the other checks cannot see.)

What it cannot tell you: whether the threats are the RIGHT threats, or
whether a control actually works. Completeness is necessary, not sufficient.
The Act II labs exist to test the controls this model only names.

Run:  python validate.py            (validates the complete reference model)
      python validate.py --draft    (validates the deliberately incomplete draft)
Exit code 0 = complete, 1 = gaps found.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field

from model import STRIDE, STRIDE_PER_ELEMENT, ThreatModel
from owasp import ALL_OWASP

GAP_TYPES = ("structure", "boundary", "stride", "owasp", "treatment", "reference")

HERE = os.path.dirname(os.path.abspath(__file__))
REFERENCE_MMD = os.path.join(HERE, "..", "..", "reference-architecture", "architecture.mmd")


@dataclass
class Report:
    gaps: dict[str, list[str]] = field(default_factory=lambda: {g: [] for g in GAP_TYPES})
    notes: list[str] = field(default_factory=list)

    def add(self, gap_type: str, message: str) -> None:
        self.gaps[gap_type].append(message)

    @property
    def ok(self) -> bool:
        return not any(self.gaps.values())

    def count(self, gap_type: str) -> int:
        return len(self.gaps[gap_type])

    def total(self) -> int:
        return sum(len(v) for v in self.gaps.values())

    def render(self) -> str:
        lines = []
        for g in GAP_TYPES:
            for msg in self.gaps[g]:
                lines.append(f"  [{g:<9}] {msg}")
        lines += [f"  [note     ] {n}" for n in self.notes]
        lines.append("  RESULT: " + ("PASS -- threat model is complete"
                                     if self.ok else f"FAIL -- {self.total()} gap(s)"))
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# The reference architecture, parsed from its Mermaid source
# --------------------------------------------------------------------------- #
# A node definition is an id followed by a shape opener: SRC[...], MEM[(...)],
# USER([...]). `subgraph X[...]` defines a group, not a node, so it is skipped.
_NODE_DEF = re.compile(r"(?<![\w-])([A-Za-z][A-Za-z0-9_]*)\s*[\[\(\{]")


def reference_node_ids(path: str = REFERENCE_MMD) -> set[str] | None:
    if not os.path.exists(path):
        return None
    ids: set[str] = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("%%") or line.startswith("flowchart"):
                continue
            if line.startswith("subgraph"):
                continue
            ids.update(_NODE_DEF.findall(line))
    return ids


# --------------------------------------------------------------------------- #
# The checks
# --------------------------------------------------------------------------- #
def check_structure(m: ThreatModel, r: Report) -> None:
    comp_ids = {c.id for c in m.components}
    zone_ids = {z.id for z in m.zones}
    control_ids = {c.id for c in m.controls}
    elements = m.element_ids()

    for c in m.components:
        if c.zone is not None and c.zone not in zone_ids:
            r.add("structure", f"component {c.id} is in unknown zone {c.zone!r}")
    for f in m.flows:
        for end in (f.src, f.dst):
            if end not in comp_ids:
                r.add("structure", f"flow {f.id} references unknown component {end!r}")
    for t in m.threats:
        if t.stride not in STRIDE:
            r.add("structure", f"threat {t.id} has invalid STRIDE letter {t.stride!r}")
            continue
        for target in t.targets:
            if target not in elements:
                r.add("structure", f"threat {t.id} targets unknown element {target!r}")
                continue
            kind = m.element_kind(target)
            if t.stride not in STRIDE_PER_ELEMENT[kind]:
                r.add("structure", f"threat {t.id}: {STRIDE[t.stride][0]} does not apply "
                                   f"to {kind} {target} (STRIDE per element)")
        for cid in t.controls:
            if cid not in control_ids:
                r.add("structure", f"threat {t.id} references missing control {cid}")
        for item in t.owasp:
            if item not in ALL_OWASP:
                r.add("structure", f"threat {t.id} cites unknown OWASP item {item}")
    for c in m.controls:
        for item in c.owasp:
            if item not in ALL_OWASP:
                r.add("structure", f"control {c.id} cites unknown OWASP item {item}")


def check_boundaries(m: ThreatModel, r: Report) -> None:
    for c in m.components:
        if c.zone is None:
            r.add("boundary", f"component {c.id} ({c.kind}) has no trust boundary")


def check_stride(m: ThreatModel, r: Report) -> None:
    for f in m.crossing_flows():
        missing = []
        for letter in STRIDE_PER_ELEMENT["flow"]:
            covered = bool(m.threats_on(f.id, letter))
            ruled_out = bool(m.stride_na.get((f.id, letter), "").strip())
            if not (covered or ruled_out):
                missing.append(letter)
        if missing:
            r.add("stride", f"flow {f.id} {f.src}->{f.dst} crosses a boundary but has no "
                            f"STRIDE review for {', '.join(missing)}")


def owasp_coverage(m: ThreatModel) -> dict[str, list[str]]:
    """OWASP item -> ids of controls that treat it."""
    cov: dict[str, list[str]] = {item: [] for item in ALL_OWASP}
    for c in m.controls:
        for item in c.owasp:
            if item in cov:
                cov[item].append(c.id)
    return cov


def check_owasp(m: ThreatModel, r: Report) -> None:
    for item, controls in owasp_coverage(m).items():
        if controls:
            continue
        reason = m.owasp_accepted.get(item, "").strip()
        if not reason:
            r.add("owasp", f"{item} {ALL_OWASP[item]} is neither mapped to a control "
                           f"nor explicitly accepted")


def check_treatment(m: ThreatModel, r: Report) -> None:
    control_ids = {c.id for c in m.controls}
    for t in m.threats:
        if not any(c in control_ids for c in t.controls) and not t.accepted.strip():
            r.add("treatment", f"threat {t.id} '{t.title}' has no control and no acceptance")


def check_reference(m: ThreatModel, r: Report, path: str = REFERENCE_MMD) -> None:
    ref = reference_node_ids(path)
    if ref is None:
        r.notes.append("reference-architecture/architecture.mmd not found; "
                       "reference check skipped")
        return
    represented = {rid for c in m.components for rid in c.ref_ids}
    for node in sorted(ref - represented):
        r.add("reference", f"reference-architecture node {node} has no component")


def validate(m: ThreatModel, reference_path: str = REFERENCE_MMD) -> Report:
    r = Report()
    check_structure(m, r)
    check_boundaries(m, r)
    check_stride(m, r)
    check_owasp(m, r)
    check_treatment(m, r)
    check_reference(m, r, reference_path)
    return r


def main(argv: list[str]) -> int:
    from reference_model import build_complete_model, build_draft_model

    m = build_draft_model() if "--draft" in argv else build_complete_model()
    report = validate(m)
    print(f"Validating: {m.name}")
    print(report.render())
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
