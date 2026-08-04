# Reference Architecture

One diagram runs through the whole series. It is the system you build in Act I, the system you attack in Act II, and the system you harden and inhabit in Act III. Watching it change across the three acts is the point — same architecture, growing up.

- **Act I — clean.** The components as an engineer draws them: data pipeline → model → the harness around it (agent loop, tools, memory) → serving and observability. This is `architecture.mmd` below.
- **Act II — annotated.** Lab 07 (Threat Model) overlays trust boundaries and attack arrows onto this same diagram — where prompt injection enters, where memory poisoning lands, where a tool becomes an exfiltration path. Each Break-It lab references the component it targets.
- **Act III — hardened.** The capstone re-draws it with controls in place: provenance on the data path, guardrails at the boundary, a verifier between agents, a kill-switch on serving.

Each lab that changes the picture drops its own variant here (e.g. `architecture-act2-threats.mmd`, `architecture-act3-hardened.mmd`) so the evolution is visible in one folder.

## The Act I clean architecture

```mermaid
flowchart LR
    subgraph DATA["Data plane (Act I: Labs 01-03)"]
        SRC[Sources] --> PIPE[Data pipeline<br/>dedup · filter · decontaminate · PII scrub]
        PIPE --> TOK[Tokenized shards<br/>+ datasheet]
    end

    TOK --> TRAIN[Train / post-train<br/>SFT · DPO]
    TRAIN --> MODEL[(Model<br/>the brain)]

    subgraph HARNESS["Harness (the body around the model)"]
        LOOP[Agent loop<br/>sense - think - act] --> TOOLS[Tools / MCP]
        LOOP --> MEM[(Memory)]
        LOOP --> MODEL
    end

    USER([User]) --> LOOP
    MODEL --> SERVE[Serving<br/>rate + cost limits]
    SERVE --> OBS[Observability<br/>tracing · cost · guardrails · kill-switch]
    OBS --> USER
```

> Rendered on GitHub automatically (Mermaid). The `.mmd` source is in this folder so labs can copy and annotate it.
