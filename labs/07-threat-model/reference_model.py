"""
The series' reference agent architecture, threat-modeled.

`build_complete_model()` is the real artifact: every component in a zone,
every boundary-crossing flow STRIDE-reviewed, every OWASP item treated or
explicitly accepted.

`build_draft_model()` is the same system as a typical first draft looks:
drawn from the chat-app point of view, with the data plane missing, the
stateful parts (memory, RAG corpus, registry, logs) never assigned a zone, and
only the "chatbot security" controls. It is built by deleting things from the
complete model, so the difference between the two is exactly the work a
threat-modeling session should produce.

Components map onto reference-architecture/architecture.mmd through `ref_ids`.
Three components are new here, because the clean Act I diagram does not show
them and an attacker cares about all three: the RAG corpus, the customer
database the tools can read, and the external services the tools can reach.

Likelihood and impact are 1..5 and ILLUSTRATIVE: they are one reasonable
reading of a generic customer-support agent, not measurements.
"""

from __future__ import annotations

from model import Component, Control, Flow, Threat, ThreatModel, Zone

# --------------------------------------------------------------------------- #
# Zones (trust boundaries)
# --------------------------------------------------------------------------- #
ZONES = [
    Zone("internet", "Internet (untrusted)", 0),
    Zone("app", "Application tier", 2),
    Zone("runtime", "Model runtime", 2),
    Zone("sandbox", "Tool sandbox", 1),
    Zone("data", "Data stores", 3),
    Zone("build", "Build and ML pipeline", 3),
    Zone("ops", "Observability", 3),
]

# --------------------------------------------------------------------------- #
# Components (DFD elements)
# --------------------------------------------------------------------------- #
COMPONENTS = [
    Component("user", "User", "external", "internet", ("USER",)),
    Component("external_services", "External services<br/>web · email · APIs",
              "external", "internet", (),
              "Anything a tool can fetch from or send to."),
    Component("data_sources", "Data sources", "external", "internet", ("SRC",)),
    Component("gateway", "App / gateway<br/>authN · rate + cost limits", "process",
              "app", ("SERVE",)),
    Component("agent_loop", "Agent loop<br/>sense · think · act", "process", "app",
              ("LOOP",)),
    Component("llm", "LLM inference", "process", "runtime", ("MODEL",)),
    Component("tools", "Tools / MCP servers", "process", "sandbox", ("TOOLS",)),
    Component("memory", "Agent memory<br/>vector store", "datastore", "data", ("MEM",)),
    Component("rag_corpus", "RAG corpus<br/>indexed documents", "datastore", "data", ()),
    Component("customer_db", "Customer database", "datastore", "data", (),
              "The crown jewels in this model."),
    Component("data_pipeline", "Data pipeline<br/>dedup · filter · PII scrub",
              "process", "build", ("PIPE", "TOK")),
    Component("training_pipeline", "Training pipeline<br/>pretrain · SFT · DPO",
              "process", "build", ("TRAIN",)),
    Component("model_registry", "Model registry<br/>signed weights", "datastore",
              "build", ("MODEL",)),
    Component("logs", "Logs and traces", "datastore", "ops", ("OBS",)),
]

# --------------------------------------------------------------------------- #
# Data flows
# --------------------------------------------------------------------------- #
FLOWS = [
    Flow("F01", "user", "gateway", "chat request"),
    Flow("F02", "gateway", "user", "response"),
    Flow("F03", "gateway", "agent_loop", "authenticated request"),
    Flow("F04", "agent_loop", "llm", "prompt + context"),
    Flow("F05", "llm", "agent_loop", "completion / proposed tool calls"),
    Flow("F06", "agent_loop", "tools", "tool invocation"),
    Flow("F07", "tools", "customer_db", "record query"),
    Flow("F08", "tools", "external_services", "outbound request / email"),
    Flow("F09", "external_services", "tools", "fetched content"),
    Flow("F10", "tools", "agent_loop", "tool result"),
    Flow("F11", "agent_loop", "memory", "memory write"),
    Flow("F12", "memory", "agent_loop", "memory recall"),
    Flow("F13", "rag_corpus", "agent_loop", "retrieved chunks"),
    Flow("F14", "data_sources", "data_pipeline", "raw documents"),
    Flow("F15", "data_pipeline", "rag_corpus", "indexed chunks"),
    Flow("F16", "data_pipeline", "training_pipeline", "tokenized shards"),
    Flow("F17", "training_pipeline", "model_registry", "checkpoints"),
    Flow("F18", "model_registry", "llm", "model weights"),
    Flow("F19", "agent_loop", "logs", "traces"),
    Flow("F20", "gateway", "logs", "access + cost logs"),
]

# --------------------------------------------------------------------------- #
# Controls
# --------------------------------------------------------------------------- #
CONTROLS = [
    Control("C01", "Quarantine untrusted content",
            "Tag everything fetched or retrieved with its provenance; process it in a "
            "quarantined model call that cannot invoke tools (dual-LLM pattern).",
            ("LLM01", "ASI01"), ("Lab 08",)),
    Control("C02", "Least-privilege tool scopes",
            "Each tool gets a per-task, per-user scope (this user's records only, read "
            "only unless the task needs write). No ambient service account.",
            ("LLM06", "LLM02", "ASI02", "ASI03"), ("Lab 10",)),
    Control("C03", "Egress allowlist",
            "Tools can only reach allowlisted hosts and recipients; no free-form URLs "
            "or addresses built from model output.",
            ("LLM02", "ASI02"), ("Lab 08",)),
    Control("C04", "Human approval for high-impact actions",
            "Send, pay, delete and bulk-read actions need explicit user approval that "
            "shows the exact action, not the model's summary of it.",
            ("LLM06", "ASI02", "ASI09"), ("Lab 10",)),
    Control("C05", "Output handling",
            "Treat model output as untrusted input to the client: encode it, never "
            "auto-render remote images or links, validate tool-call arguments.",
            ("LLM05",), ("Lab 08",)),
    Control("C06", "Sandboxed execution",
            "No shell or code-execution tool by default; where needed, run it in an "
            "isolated sandbox with no network and no credentials.",
            ("ASI05", "LLM05"), ("Lab 10",)),
    Control("C07", "Memory write gating",
            "Memory writes carry provenance and a trust score, are namespaced per "
            "user, and low-trust writes are quarantined.",
            ("ASI06", "LLM04"), ("Lab 09",)),
    Control("C08", "Permission-aware RAG with ingestion provenance",
            "Documents are hashed and attributed at ingestion; retrieval filters by "
            "the caller's permissions before similarity ranking.",
            ("LLM08", "LLM04", "ASI06"), ("Lab 08", "Lab 11")),
    Control("C09", "PII scrub and dedup at ingestion",
            "Scrub PII, deduplicate and decontaminate before data reaches training or "
            "the index.",
            ("LLM02", "LLM04"), ("Lab 01",)),
    Control("C10", "Signed, hash-pinned model artifacts",
            "Checkpoints are hashed and signed with a provenance chain (data hash + "
            "seed -> weights); the runtime refuses an unsigned or mismatched model.",
            ("LLM03", "LLM04", "ASI04"), ("Lab 02", "Lab 11")),
    Control("C11", "Tool and MCP server allowlist",
            "Only reviewed tool servers, pinned by version and hash; tool descriptions "
            "are reviewed like code because the model reads them as instructions.",
            ("LLM03", "ASI04", "ASI02"), ("Lab 10",)),
    Control("C12", "No secrets in prompts",
            "System prompts hold no credentials or security rules; secrets live in a "
            "vault and are used by tools, never shown to the model.",
            ("LLM07", "LLM02"), ("Lab 08",)),
    Control("C13", "Gateway authN, rate and cost limits",
            "Authenticate every user; cap tokens per request, requests per minute, "
            "spend per day and agent steps per task.",
            ("LLM10", "ASI03", "ASI08"), ("Lab 05", "Lab 06")),
    Control("C14", "Tamper-evident, redacted telemetry",
            "Trace every model and tool call with the instruction that caused it; "
            "redact PII before storage; append-only, hash-chained logs.",
            ("LLM02", "ASI10"), ("Lab 06",)),
    Control("C15", "Behavioral monitoring and kill-switch",
            "Alert on anomalous tool-call patterns; a kill-switch halts the agent and "
            "a circuit breaker stops a bad step from feeding the next one.",
            ("ASI08", "ASI10"), ("Lab 06", "Lab 12")),
    Control("C16", "Grounded answers with citations",
            "Answers cite retrieved sources; ungrounded claims are flagged; "
            "high-stakes answers go to a human.",
            ("LLM09", "ASI09"), ("Lab 04",)),
    Control("C17", "Service identity and delegated authority",
            "mTLS between components; the agent acts with the user's delegated "
            "identity and scopes, never as an all-powerful service principal.",
            ("ASI03",), ("Lab 12",)),
]

# Single-agent architecture: there is no agent-to-agent channel to secure yet.
OWASP_ACCEPTED = {
    "ASI07": "Single-agent architecture: no inter-agent channel exists yet. "
             "Re-open when Lab 12 adds an orchestrator and workers.",
}

# --------------------------------------------------------------------------- #
# Threats
# --------------------------------------------------------------------------- #
# Flows that cross a trust boundary in the complete model. Computed by hand
# here so the "hygiene" threats below can target them; test_complete_model_*
# checks this list against model.crossing_flows().
CROSSING = ("F01", "F02", "F04", "F05", "F06", "F07", "F08", "F09", "F10",
            "F11", "F12", "F13", "F14", "F15", "F18", "F19", "F20")

THREATS = [
    # ---- hygiene rows: the boring STRIDE-on-every-hop coverage ----------- #
    Threat("T01", "Message tampered with in transit between components", "T",
           CROSSING, 2, 3, ("C17",)),
    # F18 is excluded on purpose: see stride_na below.
    Threat("T02", "Traffic between components read in transit", "I",
           tuple(f for f in CROSSING if f != "F18"), 2, 3, ("C17",)),
    Threat("T03", "Hop flooded or starved (resource exhaustion)", "D",
           CROSSING, 2, 2, ("C13", "C15")),

    # ---- the AI-specific threats: where the lab's attention goes --------- #
    Threat("T10", "Direct prompt injection / jailbreak from the user", "T",
           ("F01",), 5, 4, ("C02", "C05", "C04"), ("LLM01", "ASI01"),
           ("Initial Access",)),
    Threat("T11", "Indirect prompt injection in fetched web content", "T",
           ("F09", "F10"), 5, 5, ("C01", "C02", "C04"), ("LLM01", "ASI01"),
           ("Initial Access",)),
    Threat("T12", "Poisoned document in the RAG corpus carries instructions", "T",
           ("F13", "F15"), 4, 5, ("C08", "C01"), ("LLM01", "LLM04", "LLM08"),
           ("Persistence",)),
    Threat("T13", "Retrieval returns another tenant's chunks", "I",
           ("F13",), 3, 5, ("C08",), ("LLM08", "LLM02"), ("Collection",)),
    Threat("T14", "Memory poisoned through normal conversation", "T",
           ("F11",), 4, 5, ("C07",), ("ASI06",), ("Persistence",)),
    Threat("T15", "Memory recall leaks another user's memories", "I",
           ("F12",), 3, 4, ("C07",), ("ASI06", "LLM02"), ("Collection",)),
    Threat("T16", "Data exfiltrated through tool egress (URL, email)", "I",
           ("F08",), 4, 5, ("C03", "C04"), ("LLM02", "ASI02"), ("Exfiltration",)),
    Threat("T17", "Bulk read of customer records through a tool", "I",
           ("F07",), 4, 5, ("C02", "C04"), ("LLM02", "LLM06"), ("Collection",)),
    Threat("T18", "Tools run with ambient service-account authority (confused deputy)",
           "E", ("tools",), 4, 5, ("C02", "C17"), ("LLM06", "ASI03"),
           ("Privilege Escalation",)),
    Threat("T19", "Agent granted more tools and permissions than the task needs", "E",
           ("agent_loop",), 4, 5, ("C02", "C04"), ("LLM06", "ASI02")),
    Threat("T20", "Model output rendered unsafely (markdown image leaks data)", "I",
           ("F02",), 4, 4, ("C05",), ("LLM05",), ("Exfiltration",)),
    Threat("T21", "Model-proposed tool call executes code", "T",
           ("F06",), 3, 5, ("C06", "C04"), ("ASI05", "LLM05"), ("Execution",)),
    Threat("T22", "Tampered weights loaded from the registry", "T",
           ("F18", "model_registry"), 2, 5, ("C10",), ("LLM03", "LLM04", "ASI04"),
           ("Persistence",)),
    Threat("T23", "Poisoned training data ingested from sources", "T",
           ("F14",), 3, 5, ("C09", "C10"), ("LLM04",), ("Resource Development",)),
    Threat("T24", "Model regurgitates PII memorized from training data", "I",
           ("llm",), 3, 4, ("C09",), ("LLM02",), ("Collection",)),
    Threat("T25", "Compromised or malicious tool / MCP server", "T",
           ("tools",), 3, 5, ("C11",), ("LLM03", "ASI04", "ASI02"),
           ("Initial Access",)),
    Threat("T26", "System prompt extracted, exposing secrets and rules", "I",
           ("llm",), 4, 3, ("C12",), ("LLM07",), ("Credential Access",)),
    Threat("T27", "Token floods and runaway agent loops (denial of wallet)", "D",
           ("llm", "gateway"), 4, 3, ("C13",), ("LLM10", "ASI08"), ("Impact",)),
    Threat("T28", "One bad tool result cascades through later steps", "T",
           ("agent_loop",), 3, 4, ("C15", "C04"), ("ASI08",), ("Impact",)),
    Threat("T29", "Hallucinated facts presented as authoritative", "T",
           ("llm",), 4, 3, ("C16",), ("LLM09",)),
    Threat("T30", "Persuasive agent output talks the user into approving harm", "S",
           ("user",), 3, 4, ("C04", "C16"), ("ASI09",)),
    Threat("T31", "No record of which instruction caused a tool action", "R",
           ("agent_loop",), 4, 3, ("C14",), ("ASI10",)),
    Threat("T32", "Agent drifts from its goal and keeps acting undetected", "E",
           ("agent_loop",), 2, 5, ("C15", "C14"), ("ASI10", "ASI01"), ("Impact",)),
    Threat("T33", "Session hijack or spoofed user at the gateway", "S",
           ("gateway",), 3, 4, ("C13",), ("ASI03",), ("Initial Access",)),
    Threat("T34", "Raw prompts with PII leak from the log store", "I",
           ("logs", "F19"), 3, 4, ("C14",), ("LLM02",), ("Collection",)),
    Threat("T35", "Logs tampered with to hide an attack", "T",
           ("logs",), 2, 4, ("C14",), (), ("Defense Evasion",)),
]

STRIDE_NA = {
    ("F18", "I"): "Open-weight model: the weights are not secret, so reading them in "
                  "transit discloses nothing. Integrity (T22) is what matters here.",
}


def build_complete_model() -> ThreatModel:
    return ThreatModel(
        name="Reference agent architecture (complete)",
        zones=list(ZONES),
        components=list(COMPONENTS),
        flows=list(FLOWS),
        threats=list(THREATS),
        controls=list(CONTROLS),
        owasp_accepted=dict(OWASP_ACCEPTED),
        stride_na=dict(STRIDE_NA),
    ).copy()


# --------------------------------------------------------------------------- #
# The deliberately incomplete first draft
# --------------------------------------------------------------------------- #
# What the draft never drew: the data plane.
DRAFT_MISSING_COMPONENTS = ("data_sources", "data_pipeline", "training_pipeline")
# What it drew but never placed in a zone: the stateful parts.
DRAFT_UNZONED = ("memory", "rag_corpus", "model_registry", "logs")
# The only controls it thought of: the "chatbot security" set.
DRAFT_CONTROLS = ("C01", "C03", "C06", "C13", "C14", "C17")
# Where the drafter stopped doing STRIDE: anything touching the stateful parts.
DRAFT_UNREVIEWED_FLOWS = ("F11", "F12", "F13", "F18", "F19", "F20")


def build_draft_model() -> ThreatModel:
    m = build_complete_model()
    m.name = "Reference agent architecture (first draft)"

    # 1. The data plane is not on the diagram at all.
    m.components = [c for c in m.components if c.id not in DRAFT_MISSING_COMPONENTS]
    m.flows = [f for f in m.flows
               if f.src not in DRAFT_MISSING_COMPONENTS
               and f.dst not in DRAFT_MISSING_COMPONENTS]

    # 2. The stateful components are drawn but floating: no zone.
    for c in m.components:
        if c.id in DRAFT_UNZONED:
            c.zone = None

    # 3. Only the controls the drafter thought of.
    m.controls = [c for c in m.controls if c.id in DRAFT_CONTROLS]

    # 4. Threats: drop every target the draft does not have or did not review,
    #    keep only controls that exist, and drop threats left with no target.
    known = m.element_ids()
    kept = []
    for t in m.threats:
        targets = tuple(x for x in t.targets
                        if x in known and x not in DRAFT_UNREVIEWED_FLOWS
                        and x not in DRAFT_UNZONED)
        if not targets:
            continue
        t.targets = targets
        t.controls = tuple(c for c in t.controls if c in DRAFT_CONTROLS)
        kept.append(t)
    m.threats = kept

    # 5. Nobody wrote down an acceptance decision or a STRIDE "n/a".
    m.owasp_accepted = {}
    m.stride_na = {}
    return m
