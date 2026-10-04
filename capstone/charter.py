"""
Parse and validate twin-charter.md.

The charter is the twin's policy, written for humans but parsed by code, so
that "the twin must always disclose it is an AI" is a test that can fail,
not a sentence in a slide deck. Later posts consume the parsed object:

    Post 14  persona contract is generated from MAY SAY / MUST REFUSE
    Post 15  hands/ reads ACTIONS to decide which tools exist and which need
             the author's approval
    Post 16  the guardrail gateway routes MUST ESCALATE tags to the author
    Post 17  each red-team attack class maps to a MUST REFUSE tag
    Post 18  the public endpoint renders disclosure-text as its banner

Format (see the top of twin-charter.md for the full description):

    key: value                      header line (before the first section)
    ## MUST REFUSE                  section heading
    - [R01] text ... #tag           rule bullet (ID prefix matches section)
    - [D01] (locked) text           DISCLOSURE rules must be locked
    - [DA01] allow: text            DATA rules are allow/deny
    - [A01] tool: x | approval: author | scope: ...    ACTIONS rules

Run:  python charter.py            validates twin-charter.md
Exit code 0 = well-formed, 1 = problems found.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
CHARTER_PATH = os.path.join(HERE, "twin-charter.md")

# Section name -> required rule-id prefix.
SECTIONS = {
    "MAY SAY": "S",
    "MUST REFUSE": "R",
    "MUST ESCALATE": "E",
    "DISCLOSURE": "D",
    "DATA": "DA",
    "ACTIONS": "A",
}

REQUIRED_HEADERS = ("twin", "author", "corpus", "charter-version", "disclosure-text")

# The minimum policy surface. A charter that parses but never mentions
# impersonation is well-formed and useless, so coverage is checked by tag.
# These tags are also the attack classes Post 17 red-teams.
REQUIRED_TAGS = {
    "MAY SAY": {"grounded", "out-of-corpus"},
    "MUST REFUSE": {"impersonation", "unsupported-opinion", "private-data",
                    "unauthorised-action", "injection"},
    "MUST ESCALATE": {"commitment", "security-report", "safety"},
    "DISCLOSURE": {"disclosure"},
}

APPROVALS = ("none", "author", "forbidden")

# Verbs in a tool's name that mean "this action has an effect outside the
# twin". Such a tool must not run with approval: none. (Deliberately crude:
# it is a tripwire for the obvious mistake, not a policy engine.)
SIDE_EFFECT_WORDS = ("send", "book", "post", "pay", "write", "delete", "halt",
                     "kill", "email", "publish", "create", "draft")

_HEADER = re.compile(r"^([a-z][a-z0-9-]*):\s+(.+)$")
_SECTION = re.compile(r"^##\s+(.+?)\s*$")
_RULE = re.compile(r"^-\s+\[([A-Z]+)(\d+)\]\s+(.*)$")
_TAGS = re.compile(r"(?:\s+#[a-z][a-z0-9-]*)+\s*$")


@dataclass
class Rule:
    id: str
    section: str
    text: str
    tags: tuple[str, ...] = ()
    locked: bool = False
    attrs: dict[str, str] = field(default_factory=dict)   # ACTIONS / DATA fields
    line: int = 0


@dataclass
class Charter:
    headers: dict[str, str] = field(default_factory=dict)
    sections: dict[str, list[Rule]] = field(default_factory=dict)
    unknown_sections: list[str] = field(default_factory=list)

    def rules(self, section: str) -> list[Rule]:
        return self.sections.get(section, [])

    def tags(self, section: str) -> set[str]:
        return {t for r in self.rules(section) for t in r.tags}

    @property
    def disclosure_text(self) -> str:
        return self.headers.get("disclosure-text", "").strip().strip('"')

    def actions(self) -> dict[str, Rule]:
        """tool name -> its ACTIONS rule."""
        return {r.attrs.get("tool", ""): r for r in self.rules("ACTIONS")}

    def allowed_tools(self) -> dict[str, str]:
        """tool name -> approval, for every tool that is not forbidden."""
        return {t: r.attrs.get("approval", "") for t, r in self.actions().items()
                if r.attrs.get("approval") != "forbidden"}


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def parse(text: str) -> Charter:
    c = Charter()
    current: str | None = None
    for n, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue

        m = _SECTION.match(line)
        if m:
            name = m.group(1).upper()
            if name in SECTIONS:
                current = name
                c.sections.setdefault(name, [])
            else:
                current = None
                c.unknown_sections.append(m.group(1))
            continue

        if current is None:
            # Header lines live before the first section. Commentary bullets
            # and prose are ignored.
            if not c.sections:
                hm = _HEADER.match(line)
                if hm:
                    c.headers[hm.group(1)] = hm.group(2).strip()
            continue

        rm = _RULE.match(line)
        if not rm:
            continue  # commentary inside a section
        prefix, num, body = rm.group(1), rm.group(2), rm.group(3).strip()

        tags: tuple[str, ...] = ()
        tm = _TAGS.search(body)
        if tm:
            tags = tuple(t.lstrip("#") for t in tm.group(0).split())
            body = body[:tm.start()].rstrip()

        locked = body.startswith("(locked)")
        if locked:
            body = body[len("(locked)"):].strip()

        attrs: dict[str, str] = {}
        if current == "ACTIONS":
            for part in body.split("|"):
                if ":" in part:
                    k, v = part.split(":", 1)
                    attrs[k.strip().lower()] = v.strip()
        elif current == "DATA":
            for kind in ("allow", "deny"):
                if body.lower().startswith(kind + ":"):
                    attrs["kind"] = kind
                    body = body[len(kind) + 1:].strip()

        c.sections[current].append(Rule(prefix + num, current, body, tags, locked,
                                        attrs, n))
    return c


def load(path: str = CHARTER_PATH) -> Charter:
    with open(path, encoding="utf-8") as fh:
        return parse(fh.read())


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate(c: Charter) -> list[str]:
    """Return a list of problems. Empty list = well-formed and complete."""
    problems: list[str] = []

    for h in REQUIRED_HEADERS:
        if not c.headers.get(h, "").strip():
            problems.append(f"header '{h}' is missing")
    for name in c.unknown_sections:
        problems.append(f"unknown section '## {name}'")

    seen: dict[str, int] = {}
    for name, prefix in SECTIONS.items():
        rules = c.rules(name)
        if name not in c.sections:
            problems.append(f"section '## {name}' is missing")
            continue
        if not rules:
            problems.append(f"section '## {name}' has no rules")
        for r in rules:
            if not re.fullmatch(prefix + r"\d+", r.id):
                problems.append(f"rule {r.id} sits in {name}; its id must start with "
                                f"'{prefix}' and nothing else")
            if r.id in seen:
                problems.append(f"rule id {r.id} is used twice (lines {seen[r.id]} "
                                f"and {r.line})")
            seen[r.id] = r.line
            if not r.text:
                problems.append(f"rule {r.id} is empty")

    # Coverage: the minimum policy surface, checked by tag.
    for name, required in REQUIRED_TAGS.items():
        if name not in c.sections:
            continue  # already reported
        for tag in sorted(required - c.tags(name)):
            problems.append(f"section '{name}' does not cover #{tag}")

    # Disclosure is non-removable: every rule locked, text present.
    for r in c.rules("DISCLOSURE"):
        if not r.locked:
            problems.append(f"disclosure rule {r.id} is not (locked); disclosure "
                            f"must be non-removable")
    dt = c.disclosure_text.lower()
    if dt and not ("ai" in dt.split() or "ai twin" in dt):
        problems.append("disclosure-text does not say the visitor is talking to an AI")

    # DATA: at least one allow and one deny, every rule typed.
    kinds = [r.attrs.get("kind") for r in c.rules("DATA")]
    for r in c.rules("DATA"):
        if not r.attrs.get("kind"):
            problems.append(f"data rule {r.id} must start with 'allow:' or 'deny:'")
    if "DATA" in c.sections:
        if "allow" not in kinds:
            problems.append("DATA has no 'allow:' rule")
        if "deny" not in kinds:
            problems.append("DATA has no 'deny:' rule")

    # ACTIONS: tool + approval + scope, side-effecting tools need approval,
    # and there must be a default-deny for everything unlisted.
    tools_seen = set()
    for r in c.rules("ACTIONS"):
        tool, approval, scope = (r.attrs.get(k, "") for k in ("tool", "approval", "scope"))
        if not tool:
            problems.append(f"action {r.id} has no 'tool:'")
        if approval not in APPROVALS:
            problems.append(f"action {r.id} approval must be one of {APPROVALS}, "
                            f"got {approval!r}")
        if not scope:
            problems.append(f"action {r.id} has no 'scope:'")
        if tool in tools_seen:
            problems.append(f"tool {tool} is listed twice")
        tools_seen.add(tool)
        if approval == "none":
            hits = [w for w in tool.lower().replace("-", "_").split("_")
                    if w in SIDE_EFFECT_WORDS]
            if hits:
                problems.append(f"action {r.id} ({tool}) has side effects "
                                f"({', '.join(hits)}) but approval: none")
    if "ACTIONS" in c.sections and not any(
            r.attrs.get("approval") == "forbidden" for r in c.rules("ACTIONS")):
        problems.append("ACTIONS has no default-deny rule (approval: forbidden)")

    return problems


def summary(c: Charter) -> str:
    """A short human-readable summary (printed by run_demo.py)."""
    lines = [f"  {c.headers.get('twin', '?')}  (charter v{c.headers.get('charter-version', '?')})",
             f"  Disclosure: \"{c.disclosure_text}\""]
    for name in SECTIONS:
        rules = c.rules(name)
        tags = sorted(c.tags(name))
        extra = f"  tags: {', '.join('#' + t for t in tags)}" if tags else ""
        lines.append(f"  {name:<14} {len(rules):>2} rules{extra}")
    lines.append("  Tools:")
    for tool, r in c.actions().items():
        scope = r.attrs.get("scope", "")
        if len(scope) > 48:
            scope = scope[:45].rstrip(" ;,") + "..."
        lines.append(f"    {tool:<12} approval: {r.attrs.get('approval', '?'):<10}{scope}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# The deliberately incomplete first draft (for run_demo.py's BEFORE column)
# --------------------------------------------------------------------------- #
def draft_text(full_text: str) -> str:
    """The charter as a typical first draft looks.

    Built by deleting from the real charter, so the difference is exactly the
    work a charter review should produce:
      - no MUST ESCALATE section (the twin answers everything itself)
      - no DISCLOSURE section and no disclosure-text (it "just sounds like me")
      - book_call runs with approval: none (convenient, and a side effect)
      - no default-deny tool rule
      - the injection refusal is missing
    """
    out, skip = [], False
    for line in full_text.splitlines():
        s = line.strip()
        sm = _SECTION.match(s)
        if sm:
            skip = sm.group(1).upper() in ("MUST ESCALATE", "DISCLOSURE")
        if skip:
            continue
        if s.startswith("disclosure-text:"):
            continue
        if s.startswith("- [R08]") or s.startswith("- [A05]"):
            continue
        if s.startswith("- [A03]"):
            line = line.replace("approval: author", "approval: none")
        out.append(line)
    return "\n".join(out) + "\n"


def main(argv: list[str]) -> int:
    path = argv[0] if argv else CHARTER_PATH
    c = load(path)
    problems = validate(c)
    print(f"Validating: {os.path.relpath(path)}")
    print(summary(c))
    for p in problems:
        print(f"  [problem] {p}")
    print("  RESULT: " + ("PASS -- charter is well-formed and complete" if not problems
                          else f"FAIL -- {len(problems)} problem(s)"))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
