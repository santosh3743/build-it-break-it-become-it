"""
The digital twin, threat-modeled with Lab 07's toolkit.

Nothing here reimplements Lab 07. We load its `model` dataclasses, its
completeness `validate`or and its risk-ranked `checklist` by file path
(labs_path.lab07()), describe the twin with them, and let Lab 07 decide
whether the description is complete. Lab 07's reference architecture was a
generic support agent; the twin is a different system with a different crown
jewel. There it was the customer database. Here it is the author's identity:
what the twin says in his name, what it does in his name, and what it reveals.

Five parts (from the capstone spec), each a group of components below:

    corpus      corpus_index                          Post 14
    persona     persona_contract, llm                 Post 14
    memory      memory_store                          Post 15
    hands       tool_faq, tool_draft_reply,
                tool_book_call                        Post 15
    guardrails  public_endpoint, guardrail_gateway,
                kill_switch, logs                     Post 16

Control ids C01-C17 are Lab 07's controls, reused verbatim by id so the
Post 17 red-team can grade the twin against the same checklist; only their
`verify_in` changes, to the capstone post that tests each one on the twin.
C18-C24 are new: they exist because the asset is a person.

Likelihood and impact are 1..5 and ILLUSTRATIVE, as in Lab 07: one
reasonable reading of a public, disclosed twin, not measurements.

Run:  python twin_threat_model.py           validate the complete twin model
      python twin_threat_model.py --draft   validate the first draft
"""

from __future__ import annotations

import dataclasses
import os
import sys

import labs_path

L07 = labs_path.lab07()
Component, Control, Flow, Threat, ThreatModel, Zone = (
    L07.model.Component, L07.model.Control, L07.model.Flow,
    L07.model.Threat, L07.model.ThreatModel, L07.model.Zone)

HERE = os.path.dirname(os.path.abspath(__file__))
TWIN_MMD = os.path.join(HERE, "architecture.mmd")

# --------------------------------------------------------------------------- #
# Zones (trust boundaries)
# --------------------------------------------------------------------------- #
ZONES = [
    Zone("internet", "Internet (anyone)", 0),
    Zone("edge", "Public edge", 1),
    Zone("app", "Twin application", 2),
    Zone("runtime", "Model runtime", 2),
    Zone("tools", "Tool sandbox", 1),
    Zone("data", "Twin data stores", 3),
    Zone("ops", "Operations", 3),
    Zone("owner", "Author (owner channel)", 4),
]

# --------------------------------------------------------------------------- #
# Components. `ref_ids` are node ids in capstone/architecture.mmd.
# --------------------------------------------------------------------------- #
COMPONENTS = [
    Component("visitor", "Visitor", "external", "internet", ("VISITOR",),
              "Anyone on the internet. Untrusted, unauthenticated by design."),
    Component("public_endpoint", "Public ask-my-twin endpoint", "process", "edge",
              ("ENDPOINT",), "Serves the disclosure banner; per-IP rate limit."),
    Component("guardrail_gateway", "Guardrail gateway", "process", "app", ("GUARD",),
              "Input and output checks against the charter; escalation routing; "
              "cost caps; obeys the kill-switch."),
    Component("twin_agent_loop", "Twin agent loop", "process", "app", ("LOOP",)),
    Component("llm", "LLM inference", "process", "runtime", ("LLM",)),
    Component("persona_contract", "Persona contract", "datastore", "data", ("PERSONA",),
              "System contract + voice exemplars, generated from twin-charter.md."),
    Component("corpus_index", "Corpus index", "datastore", "data", ("CORPUS",),
              "Chunks of the author's own published writing, with provenance."),
    Component("memory_store", "Memory store", "datastore", "data", ("MEM",),
              "Per-visitor conversation memory with provenance."),
    Component("tool_faq", "faq_answer tool", "process", "tools", ("FAQ",)),
    Component("tool_draft_reply", "draft_reply tool", "process", "tools", ("DRAFT",)),
    Component("tool_book_call", "book_call tool", "process", "tools", ("BOOK",)),
    Component("kill_switch", "Kill-switch", "process", "ops", ("KILL",)),
    Component("logs", "Logs and traces", "datastore", "ops", ("LOGS",)),
    Component("author", "Author (the real person)", "external", "owner", ("AUTHOR",),
              "Approves drafts and bookings, receives escalations, holds the "
              "kill-switch, curates the corpus."),
]

PARTS = {
    "corpus": ("corpus_index",),
    "persona": ("persona_contract", "llm"),
    "memory": ("memory_store",),
    "hands": ("tool_faq", "tool_draft_reply", "tool_book_call"),
    "guardrails": ("public_endpoint", "guardrail_gateway", "kill_switch", "logs"),
    "outside": ("visitor", "author", "twin_agent_loop"),
}

# charter ACTIONS tool name -> threat-model component
TOOL_COMPONENTS = {
    "faq_answer": "tool_faq",
    "draft_reply": "tool_draft_reply",
    "book_call": "tool_book_call",
    "kill_switch": "kill_switch",
}

# --------------------------------------------------------------------------- #
# Data flows
# --------------------------------------------------------------------------- #
FLOWS = [
    Flow("F01", "visitor", "public_endpoint", "question"),
    Flow("F02", "public_endpoint", "visitor", "answer + disclosure banner"),
    Flow("F03", "public_endpoint", "guardrail_gateway", "rate-limited request"),
    Flow("F04", "guardrail_gateway", "public_endpoint", "screened answer"),
    Flow("F05", "guardrail_gateway", "twin_agent_loop", "screened request"),
    Flow("F06", "twin_agent_loop", "guardrail_gateway", "draft answer for output checks"),
    Flow("F07", "twin_agent_loop", "llm", "prompt + persona + cited context"),
    Flow("F08", "llm", "twin_agent_loop", "completion / proposed tool calls"),
    Flow("F09", "persona_contract", "twin_agent_loop", "persona contract + exemplars"),
    Flow("F10", "corpus_index", "twin_agent_loop", "retrieved chunks with citations"),
    Flow("F11", "twin_agent_loop", "memory_store", "memory write"),
    Flow("F12", "memory_store", "twin_agent_loop", "memory recall"),
    Flow("F13", "twin_agent_loop", "tool_faq", "FAQ query"),
    Flow("F14", "tool_faq", "twin_agent_loop", "FAQ answer"),
    Flow("F15", "twin_agent_loop", "tool_draft_reply", "draft request"),
    Flow("F16", "tool_draft_reply", "author", "draft in review queue"),
    Flow("F17", "twin_agent_loop", "tool_book_call", "call request"),
    Flow("F18", "tool_book_call", "author", "pending call request"),
    Flow("F19", "author", "kill_switch", "halt / resume (owner channel)"),
    Flow("F20", "kill_switch", "guardrail_gateway", "halt signal"),
    Flow("F21", "twin_agent_loop", "logs", "traces"),
    Flow("F22", "guardrail_gateway", "logs", "access, cost and guardrail decisions"),
    Flow("F23", "guardrail_gateway", "author", "escalations"),
    Flow("F24", "author", "corpus_index", "curated published writing"),
    Flow("F25", "author", "persona_contract", "signed charter + persona"),
    Flow("F26", "tool_draft_reply", "twin_agent_loop", "draft id"),
    Flow("F27", "tool_book_call", "twin_agent_loop", "request id"),
]

# --------------------------------------------------------------------------- #
# Controls
# --------------------------------------------------------------------------- #
# Lab 07 control id -> capstone post(s) that verify it ON THE TWIN.
REUSED_CONTROLS = {
    "C01": ("Post 17",),             # quarantine untrusted content
    "C02": ("Post 15", "Post 17"),   # least-privilege tool scopes
    "C03": ("Post 17",),             # egress allowlist (empty in v1: no tool egresses)
    "C04": ("Post 15", "Post 17"),   # human approval for high-impact actions
    "C05": ("Post 16",),             # output handling
    "C06": ("Post 17",),             # sandboxed execution (no code tool at all)
    "C07": ("Post 15", "Post 17"),   # memory write gating (Lab 09 defenses on)
    "C08": ("Post 14", "Post 17"),   # permission-aware RAG + ingestion provenance
    "C09": ("Post 14",),             # PII scrub + dedup at ingestion (Lab 01)
    "C10": ("Post 16",),             # signed, hash-pinned model artifacts
    "C11": ("Post 15",),             # tool allowlist
    "C12": ("Post 17",),             # no secrets in prompts
    "C13": ("Post 16",),             # gateway rate and cost limits
    "C14": ("Post 16",),             # tamper-evident, redacted telemetry
    "C15": ("Post 16", "Post 18"),   # behavioral monitoring + kill-switch
    "C16": ("Post 14", "Post 16"),   # grounded answers with citations
    "C17": ("Post 16",),             # service identity between components
}


def _reused_controls() -> list:
    by_id = {c.id: c for c in L07.reference_model.CONTROLS}
    return [dataclasses.replace(by_id[cid], verify_in=posts)
            for cid, posts in REUSED_CONTROLS.items()]


NEW_CONTROLS = [
    Control("C18", "Non-removable AI disclosure",
            "Every conversation opens with the charter's disclosure-text; every host "
            "page shows it as a banner; 'are you human / are you him?' is always "
            "answered truthfully; drafts are labelled AI-drafted. A build without it "
            "fails tests.",
            ("ASI09", "LLM09"), ("Post 16", "Post 18")),
    Control("C19", "Charter enforcement at the gateway",
            "Inputs and outputs are checked against MUST REFUSE before anything "
            "reaches the visitor; tools outside ACTIONS do not exist in the loop.",
            ("LLM01", "LLM06", "ASI01"), ("Post 16", "Post 17")),
    Control("C20", "Published-only corpus with provenance",
            "Only the author's own published writing enters the corpus; every chunk "
            "carries its source URL and content hash; the author curates additions.",
            ("LLM02", "LLM04", "LLM08"), ("Post 14",)),
    Control("C21", "Escalation routing to the author",
            "Requests matching MUST ESCALATE are not answered by the twin; they are "
            "queued for the author with the visitor's words verbatim.",
            ("LLM06", "ASI09"), ("Post 16",)),
    Control("C22", "Authenticated owner channel",
            "Author commands (kill-switch, charter and persona changes, approvals) "
            "arrive only through an authenticated channel outside chat. No chat "
            "message, however it is signed, carries owner authority.",
            ("ASI03", "ASI01"), ("Post 15", "Post 16")),
    Control("C23", "Faithfulness and voice evaluation",
            "A golden set of questions with known answers from the corpus scores "
            "faithfulness to the author and voice; a release that regresses is "
            "blocked.",
            ("LLM09",), ("Post 16",)),
    Control("C24", "Hash-pinned charter and persona",
            "The twin refuses to start if twin-charter.md or the persona contract "
            "does not match the hash the author signed.",
            ("ASI01",), ("Post 16",)),
]


def build_controls() -> list:
    return _reused_controls() + list(NEW_CONTROLS)


OWASP_ACCEPTED = {
    "ASI07": "Single-agent twin: no inter-agent channel exists. Re-open if Post 16 "
             "or Post 17 adds Lab 12's verifier as a second agent.",
}

# Published material is public by design, so reading it in transit discloses
# nothing. Integrity of these flows (T16, T28) is what matters.
STRIDE_NA = {
    ("F24", "I"): "The corpus is the author's PUBLISHED writing; it is public by design.",
    ("F25", "I"): "The charter is published in this repo and the persona is derived "
                  "from it and from published writing.",
}

# --------------------------------------------------------------------------- #
# Threats
# --------------------------------------------------------------------------- #
_AI_THREATS = [
    # ---- identity: the reason this asset is different ------------------- #
    Threat("T10", "Visitor jailbreaks the twin into speaking off-charter or out of character",
           "T", ("F01", "F03"), 5, 4, ("C19", "C05", "C01"), ("LLM01", "ASI01"),
           ("Initial Access",)),
    Threat("T11", "Visitor is led to believe the twin is the real author", "S",
           ("visitor",), 4, 5, ("C18",), ("ASI09",)),
    Threat("T12", "Attacker claims in chat to be the author, to change rules or unlock tools",
           "S", ("twin_agent_loop",), 4, 5, ("C22", "C19"), ("ASI03", "ASI01"),
           ("Privilege Escalation",)),
    Threat("T13", "Attacker spoofs the owner to trigger or disable the kill-switch", "S",
           ("kill_switch",), 2, 4, ("C22",), ("ASI03",)),
    Threat("T14", "Private context leaks: another visitor's memories or internal config",
           "I", ("F12", "memory_store"), 3, 5, ("C07", "C12"), ("LLM02", "ASI06"),
           ("Collection",)),
    Threat("T15", "Twin states opinions the author never published (misrepresentation)",
           "T", ("llm",), 5, 4, ("C16", "C23", "C19"), ("LLM09",)),
    Threat("T16", "Non-author or poisoned text enters the corpus", "T",
           ("F24", "corpus_index"), 3, 5, ("C20", "C08", "C09"), ("LLM04", "LLM08"),
           ("Persistence",)),
    Threat("T17", "Retrieved chunk carries instructions (indirect injection)", "T",
           ("F10",), 3, 4, ("C01", "C08"), ("LLM01", "LLM08"), ("Initial Access",)),
    Threat("T18", "Memory poisoned so the twin misrepresents the author later", "T",
           ("F11",), 4, 5, ("C07",), ("ASI06", "LLM04"), ("Persistence",)),
    Threat("T19", "A draft reply leaves the review queue without the author's approval",
           "E", ("tool_draft_reply",), 3, 5, ("C04", "C02"), ("LLM06", "ASI02")),
    Threat("T20", "book_call abused to flood the author or commit him to a call", "D",
           ("tool_book_call",), 4, 3, ("C04", "C13"), ("ASI02", "LLM10"), ("Impact",)),
    Threat("T21", "faq_answer used to read data outside the FAQ set", "E",
           ("tool_faq",), 3, 4, ("C02", "C11"), ("ASI02", "LLM06")),
    Threat("T22", "Twin wired with more tools than the charter's ACTIONS list", "E",
           ("twin_agent_loop",), 3, 5, ("C02", "C11", "C19"), ("LLM06", "ASI02")),
    Threat("T23", "Model output rendered unsafely in the visitor's browser", "I",
           ("F02",), 3, 4, ("C05",), ("LLM05",), ("Exfiltration",)),
    Threat("T24", "Tool argument built from model output is executed as code", "T",
           ("F13", "F15", "F17"), 2, 5, ("C06", "C05"), ("ASI05", "LLM05"),
           ("Execution",)),
    # Impact is low on purpose: the charter is public, so the prompt holds no secrets.
    Threat("T25", "System prompt and persona contract extracted", "I",
           ("llm",), 5, 2, ("C12",), ("LLM07",)),
    Threat("T26", "Token floods and runaway loops on the public endpoint (denial of wallet)",
           "D", ("public_endpoint", "llm"), 5, 3, ("C13",), ("LLM10",), ("Impact",)),
    Threat("T27", "Escalation-worthy request answered by the twin instead of the author",
           "E", ("guardrail_gateway",), 3, 5, ("C21", "C19"), ("ASI09", "LLM06")),
    Threat("T28", "Charter or persona contract tampered to change the twin's goals", "T",
           ("persona_contract", "F09"), 2, 5, ("C24", "C10"), ("ASI01",),
           ("Persistence",)),
    Threat("T29", "Compromised model weights or tool dependency", "T",
           ("llm",), 2, 5, ("C10", "C11"), ("LLM03", "ASI04"), ("Initial Access",)),
    Threat("T30", "Kill-switch fails to halt in-flight responses", "D",
           ("kill_switch", "F20"), 2, 5, ("C15",), ("ASI10", "ASI08")),
    Threat("T31", "No record of what the twin said or did in the author's name", "R",
           ("twin_agent_loop", "logs"), 4, 4, ("C14",), ("ASI10",)),
    Threat("T32", "Visitor denies sending an abusive or injected request", "R",
           ("visitor",), 3, 2, ("C14",)),
    Threat("T33", "Logs leak visitor personal data or private context", "I",
           ("logs", "F21"), 3, 4, ("C14", "C09"), ("LLM02",), ("Collection",)),
    Threat("T34", "Logs tampered with to hide an attack", "T",
           ("logs",), 2, 4, ("C14",), (), ("Defense Evasion",)),
    Threat("T35", "A bad tool result or memory cascades into later turns", "T",
           ("twin_agent_loop",), 3, 3, ("C15", "C07"), ("ASI08",)),
    Threat("T36", "Twin drifts off-charter and keeps acting undetected", "E",
           ("twin_agent_loop",), 2, 5, ("C15", "C14"), ("ASI10", "ASI01")),
    Threat("T37", "Author approves a harmful draft or booking because the summary misled him",
           "S", ("author",), 3, 4, ("C04",), ("ASI09",)),
    # The one threat this system cannot treat on its own. Writing that down is
    # the honest answer; leaving it out would make the validator lie.
    Threat("T38", "Twin output clipped and passed off as the author off-platform, or "
                  "his voice cloned from it", "S", ("author",), 3, 4, (), ("ASI09",), (),
           accepted="Off-platform misuse is outside this system's boundary. D03 labels "
                    "drafts and the banner labels answers, which helps but cannot stop "
                    "screenshots. Re-tested as an impersonation case in Post 17."),
]


def _hygiene(m) -> list:
    """Lab 07's 'STRIDE on every hop' rows, over the twin's crossing flows.

    Computed from the model (not typed by hand) so that adding a flow that
    crosses a boundary is automatically covered for T, I and D, unless it is
    explicitly ruled out in STRIDE_NA.
    """
    crossing = tuple(f.id for f in m.crossing_flows())
    no_i = {fid for (fid, letter) in STRIDE_NA if letter == "I"}
    return [
        Threat("T01", "Message tampered with in transit between components", "T",
               crossing, 2, 3, ("C17",)),
        Threat("T02", "Traffic between components read in transit", "I",
               tuple(f for f in crossing if f not in no_i), 2, 3, ("C17",)),
        Threat("T03", "Hop flooded or starved (resource exhaustion)", "D",
               crossing, 2, 2, ("C13", "C15")),
    ]


def build_twin_model():
    m = ThreatModel(
        name="Digital twin (complete)",
        zones=list(ZONES),
        components=list(COMPONENTS),
        flows=list(FLOWS),
        threats=[],
        controls=build_controls(),
        owasp_accepted=dict(OWASP_ACCEPTED),
        stride_na=dict(STRIDE_NA),
    ).copy()
    m.threats = _hygiene(m) + [dataclasses.replace(t) for t in _AI_THREATS]
    return m


# --------------------------------------------------------------------------- #
# The deliberately incomplete first draft
# --------------------------------------------------------------------------- #
# Drawn from the chatbot point of view: the real person is not on the diagram,
# so neither is the kill-switch he holds.
DRAFT_MISSING_COMPONENTS = ("author", "kill_switch")
# Drawn, but nobody decided how much to trust them: the stateful parts.
DRAFT_UNZONED = ("persona_contract", "corpus_index", "memory_store", "logs")
# The "chatbot security" controls a first draft thinks of.
DRAFT_CONTROLS = ("C01", "C05", "C12", "C13", "C14", "C16")
# Where the drafter stopped doing STRIDE: anything touching the stateful parts.
DRAFT_UNREVIEWED_FLOWS = ("F09", "F10", "F11", "F12", "F21", "F22")


def build_draft_model():
    """Same deletion recipe as Lab 07's build_draft_model()."""
    m = build_twin_model()
    m.name = "Digital twin (first draft)"
    m.components = [c for c in m.components if c.id not in DRAFT_MISSING_COMPONENTS]
    m.flows = [f for f in m.flows if f.src not in DRAFT_MISSING_COMPONENTS
               and f.dst not in DRAFT_MISSING_COMPONENTS]
    for c in m.components:
        if c.id in DRAFT_UNZONED:
            c.zone = None
    m.controls = [c for c in m.controls if c.id in DRAFT_CONTROLS]
    known = m.element_ids()
    kept = []
    for t in m.threats:
        targets = tuple(x for x in t.targets if x in known
                        and x not in DRAFT_UNREVIEWED_FLOWS and x not in DRAFT_UNZONED)
        if not targets:
            continue
        t.targets = targets
        t.controls = tuple(c for c in t.controls if c in DRAFT_CONTROLS)
        t.accepted = ""
        kept.append(t)
    m.threats = kept
    m.owasp_accepted = {}
    m.stride_na = {}
    return m


# --------------------------------------------------------------------------- #
# Checks on top of Lab 07's validator
# --------------------------------------------------------------------------- #
def validate(m):
    """Lab 07's validator, pointed at the twin's own architecture diagram."""
    return L07.validate.validate(m, reference_path=TWIN_MMD)


def component_controls(m) -> dict[str, list[str]]:
    """Component id -> controls treating threats that target it DIRECTLY.

    Stricter than Lab 07 needs: a component only counts as mapped if some
    threat names it as a target and that threat has a control that exists.
    """
    existing = {c.id for c in m.controls}
    out: dict[str, set[str]] = {c.id: set() for c in m.components}
    for t in m.threats:
        for target in t.targets:
            if target in out:
                out[target].update(c for c in t.controls if c in existing)
    return {k: sorted(v) for k, v in out.items()}


def unmapped_components(m) -> list[str]:
    return [cid for cid, ctl in component_controls(m).items() if not ctl]


def charter_gaps(charter, m) -> list[str]:
    """Cross-check the charter (policy) against the threat model (design).

    The two documents describe the same system; when they disagree, one of
    them is wrong. This is the check that catches a tool added to the charter
    but never threat-modeled, or a disclosure policy with no control behind it.
    """
    gaps = []
    comp_ids = {c.id for c in m.components}
    control_ids = {c.id for c in m.controls}
    allowed = charter.allowed_tools()
    for tool in allowed:
        comp = TOOL_COMPONENTS.get(tool)
        if comp is None:
            gaps.append(f"charter tool '{tool}' has no threat-model component")
        elif comp not in comp_ids:
            gaps.append(f"charter tool '{tool}' -> component '{comp}' is not in the model")
    for tool, comp in TOOL_COMPONENTS.items():
        if comp in comp_ids and tool not in allowed:
            gaps.append(f"component '{comp}' exists but the charter does not list "
                        f"tool '{tool}'")
    if charter.rules("DISCLOSURE") and "C18" not in control_ids:
        gaps.append("charter requires disclosure but no control (C18) enforces it")
    if not charter.rules("DISCLOSURE"):
        gaps.append("charter has no DISCLOSURE section")
    if charter.rules("MUST ESCALATE") and "C21" not in control_ids:
        gaps.append("charter requires escalation but no control (C21) routes it")
    if not charter.rules("MUST ESCALATE"):
        gaps.append("charter has no MUST ESCALATE section")
    for tool, approval in allowed.items():
        if approval == "author" and "C04" not in control_ids and tool != "kill_switch":
            gaps.append(f"tool '{tool}' needs author approval but no control (C04) "
                        f"enforces human approval")
    return gaps


def checklist(m):
    """Lab 07's risk-ranked control checklist for the twin (Post 17 grades it)."""
    return L07.checklist.checklist_from_model(m)


def main(argv: list[str]) -> int:
    m = build_draft_model() if "--draft" in argv else build_twin_model()
    r = validate(m)
    print(f"Validating: {m.name}")
    print(r.render())
    um = unmapped_components(m)
    print(f"  Components with no control: {', '.join(um) if um else 'none'}")
    return 0 if (r.ok and not um) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
