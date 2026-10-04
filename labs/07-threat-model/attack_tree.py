"""
Attack trees (Schneier, 1999) with AND/OR gates.

The root is the attacker's goal. An OR node is achieved if ANY child is; an
AND node only if ALL children are. Leaves are concrete attacker steps, and
each leaf lists the controls that would block it.

Two questions make a tree useful rather than decorative:

  1. Given the controls we actually have, is the goal still reachable?
  2. How many distinct ways in are there? (`attack_paths` enumerates every
     minimal set of leaves that achieves the goal; each set is one attack.)

Running both against the draft and the complete threat model shows what the
missing threat-modeling work would have cost.

A leaf counts as blocked if ANY of its controls is in place. That is the
optimistic reading a design review makes ("we have an egress allowlist"); it
says nothing about whether the allowlist works. Act II tests that part.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product


@dataclass
class Node:
    id: str
    label: str
    gate: str = "LEAF"                 # "AND" | "OR" | "LEAF"
    children: list["Node"] = field(default_factory=list)
    controls: tuple[str, ...] = ()     # leaves only: controls that block this step
    entry: str = ""                    # leaves only: component the attacker touches
    owasp: tuple[str, ...] = ()

    def walk(self):
        yield self
        for c in self.children:
            yield from c.walk()

    def leaves(self) -> list["Node"]:
        return [n for n in self.walk() if n.gate == "LEAF"]


def leaf_open(leaf: Node, implemented: set[str]) -> bool:
    return not any(c in implemented for c in leaf.controls)


def achievable(node: Node, implemented: set[str]) -> bool:
    if node.gate == "LEAF":
        return leaf_open(node, implemented)
    results = [achievable(c, implemented) for c in node.children]
    return any(results) if node.gate == "OR" else all(results)


def attack_paths(node: Node, implemented: set[str] | None = None) -> list[tuple[str, ...]]:
    """Every combination of leaves that achieves `node`, given the controls.

    OR  -> the union of the children's paths.
    AND -> one path from each child, combined (the Cartesian product).
    """
    implemented = implemented or set()
    if node.gate == "LEAF":
        return [(node.id,)] if leaf_open(node, implemented) else []
    child_paths = [attack_paths(c, implemented) for c in node.children]
    if node.gate == "OR":
        return [p for paths in child_paths for p in paths]
    return [tuple(x for part in combo for x in part) for combo in product(*child_paths)]


# --------------------------------------------------------------------------- #
# The tree for this lab's goal
# --------------------------------------------------------------------------- #
def exfiltration_tree() -> Node:
    """Goal: exfiltrate customer data via the agent."""
    L = Node  # short alias to keep the tree readable
    return Node("G", "Exfiltrate customer data via the agent", "OR", [
        Node("A", "Hijack the agent and walk the data out", "AND", [
            Node("A1", "Get attacker instructions into the context", "OR", [
                L("L1", "Plant instructions in a web page a tool fetches",
                  controls=("C01",), entry="external_services", owasp=("LLM01", "ASI01")),
                L("L2", "Plant a poisoned document in the RAG corpus",
                  controls=("C08", "C01"), entry="data_sources", owasp=("LLM01", "LLM08")),
                L("L3", "Poison long-term memory via a normal-looking chat",
                  controls=("C07",), entry="user", owasp=("ASI06",)),
            ]),
            Node("A2", "Make the agent read records it should not", "OR", [
                L("L4", "Tool runs as an all-tables service account",
                  controls=("C02",), owasp=("LLM06", "ASI03")),
                L("L5", "Retrieval returns other tenants' chunks",
                  controls=("C08",), owasp=("LLM08",)),
            ]),
            Node("A3", "Get the data out", "OR", [
                L("L6", "Mail or POST it to an attacker address",
                  controls=("C03", "C04"), owasp=("LLM02", "ASI02")),
                L("L7", "Emit a markdown image whose URL carries the data",
                  controls=("C05",), owasp=("LLM05",)),
            ]),
        ]),
        Node("B", "Ship a malicious tool that reads the data itself", "AND", [
            L("L8", "Get a malicious MCP server installed",
              controls=("C11",), entry="tools", owasp=("LLM03", "ASI04")),
            L("L9", "The tool server holds broad database credentials",
              controls=("C02",), owasp=("ASI03",)),
        ]),
        Node("C", "Extract data the model memorized", "AND", [
            L("L10", "Customer PII was in the training data",
              controls=("C09",), owasp=("LLM02",)),
            # No control can stop people asking questions. Rate limits slow
            # extraction down, but the step itself is always available.
            L("L11", "Query the model with extraction prompts", controls=(), entry="user"),
        ]),
        Node("D", "Steal credentials from the system prompt", "AND", [
            L("L12", "System prompt holds a DB connection string",
              controls=("C12",), owasp=("LLM07",)),
            # LLM07's core lesson: assume the system prompt WILL be extracted.
            L("L13", "Talk the model into revealing its system prompt", controls=(),
              entry="user", owasp=("LLM07", "LLM01")),
        ]),
    ])


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def render_text(node: Node, implemented: set[str] | None = None, _depth: int = 0) -> str:
    implemented = implemented if implemented is not None else set()
    pad = "  " * _depth
    if node.gate == "LEAF":
        status = "open   " if leaf_open(node, implemented) else "BLOCKED"
        ctl = ",".join(node.controls) or "-"
        line = f"{pad}[{status}] {node.id:<3} {node.label}  (controls: {ctl})"
    else:
        status = "reachable" if achievable(node, implemented) else "blocked"
        line = f"{pad}{node.gate:<3} {node.id:<3} {node.label}  -> {status}"
    out = [line]
    for c in node.children:
        out.append(render_text(c, implemented, _depth + 1))
    return "\n".join(out)


def _esc(s: str) -> str:
    return s.replace('"', "'")


def render_mermaid(node: Node, implemented: set[str] | None = None) -> str:
    implemented = implemented if implemented is not None else set()
    lines = ["%% Attack tree -- generated by labs/07-threat-model/attack_tree.py",
             "flowchart TD"]
    for n in node.walk():
        if n.gate == "LEAF":
            ctl = ", ".join(n.controls) or "no control"
            lines.append(f'    {n.id}["{n.id}: {_esc(n.label)}<br/><i>{ctl}</i>"]')
            lines.append(f"    class {n.id} {'open' if leaf_open(n, implemented) else 'blocked'}")
        else:
            lines.append(f'    {n.id}{{"{n.gate}<br/>{n.id}: {_esc(n.label)}"}}')
    for n in node.walk():
        for c in n.children:
            lines.append(f"    {n.id} --> {c.id}")
    lines.append("    classDef open fill:#fde2e1,stroke:#c0392b,color:#000")
    lines.append("    classDef blocked fill:#e3f4e1,stroke:#2e7d32,color:#000")
    return "\n".join(lines) + "\n"
