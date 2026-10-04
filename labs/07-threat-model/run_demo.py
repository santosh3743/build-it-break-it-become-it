"""
Lab 07 demo: threat-model the reference agent architecture, as code.

BEFORE  the first-draft threat model (chat-app view: no data plane, stateful
        parts never placed in a zone, "chatbot security" controls only).
AFTER   the complete model.

Prints the validator's verdict on both, a before/after table, the attack tree
for "exfiltrate customer data via the agent" evaluated against each model's
controls, and the top risk-ranked controls. Writes every artifact to out/.

Zero arguments, standard library only, runs in well under a second.
"""

from __future__ import annotations

import os

from attack_tree import achievable, attack_paths, exfiltration_tree, leaf_open
from owasp import OWASP_AGENTIC_2026, OWASP_LLM_2025
from reference_model import build_complete_model, build_draft_model
from render import HERE, write_all
from risk import rank_controls
from validate import owasp_coverage, validate

RULE = "=" * 78


def owasp_unmapped(m, catalog) -> tuple[int, int]:
    cov = owasp_coverage(m)
    unmapped = [i for i in catalog if not cov[i] and not m.owasp_accepted.get(i)]
    accepted = [i for i in catalog if not cov[i] and m.owasp_accepted.get(i)]
    return len(unmapped), len(accepted)


def tree_side_by_side(node, draft_ctl, full_ctl, depth=0) -> list[str]:
    """One line per node with its status under the draft and complete controls."""
    def status(n, ctl):
        if n.gate == "LEAF":
            return "open" if leaf_open(n, ctl) else "blocked"
        return "REACH" if achievable(n, ctl) else "blocked"

    label = f"{'  ' * depth}{n_gate(node)} {node.id} {node.label}"
    if len(label) > 62:
        label = label[:59] + "..."
    lines = [f"  {label:<62} {status(node, draft_ctl):>7}  {status(node, full_ctl):>8}"]
    for c in node.children:
        lines += tree_side_by_side(c, draft_ctl, full_ctl, depth + 1)
    return lines


def n_gate(node) -> str:
    return {"AND": "AND", "OR": "OR ", "LEAF": " - "}[node.gate]


def main() -> None:
    draft, full = build_draft_model(), build_complete_model()
    r_draft, r_full = validate(draft), validate(full)
    tree = exfiltration_tree()
    draft_ctl = {c.id for c in draft.controls}
    full_ctl = {c.id for c in full.controls}

    print(RULE)
    print("Lab 07 -- Threat-Modeling an AI System Like You Mean It")
    print(RULE)
    print(f"Reference agent architecture: {len(full.components)} components, "
          f"{len(full.flows)} data flows, {len(full.zones)} trust zones, "
          f"{len(full.crossing_flows())} flows cross a trust boundary.\n")

    print("[1] BEFORE -- validator on the first-draft threat model")
    print(r_draft.render())
    print()
    print("[2] AFTER -- validator on the complete threat model")
    print(r_full.render())
    print()

    # ---- the before/after table (the screenshot) ------------------------- #
    paths_none = len(attack_paths(tree, set()))
    paths_draft = len(attack_paths(tree, draft_ctl))
    paths_full = len(attack_paths(tree, full_ctl))
    llm_d, _ = owasp_unmapped(draft, OWASP_LLM_2025)
    llm_f, _ = owasp_unmapped(full, OWASP_LLM_2025)
    asi_d, _ = owasp_unmapped(draft, OWASP_AGENTIC_2026)
    asi_f, asi_f_acc = owasp_unmapped(full, OWASP_AGENTIC_2026)

    rows = [
        ("Components on the diagram", len(draft.components), len(full.components)),
        ("Components with no trust boundary", r_draft.count("boundary"),
         r_full.count("boundary")),
        ("Boundary-crossing flows not STRIDE-reviewed", r_draft.count("stride"),
         r_full.count("stride")),
        ("OWASP LLM 2025 items unmapped (of 10)", llm_d, llm_f),
        ("OWASP Agentic 2026 items unmapped (of 10)", asi_d,
         f"{asi_f} ({asi_f_acc} accepted)"),
        ("Threats with no control", r_draft.count("treatment"), r_full.count("treatment")),
        ("Reference-architecture nodes not modeled", r_draft.count("reference"),
         r_full.count("reference")),
        ("Controls identified", len(draft.controls), len(full.controls)),
        (f"Attack paths to the goal still open (of {paths_none})", paths_draft, paths_full),
        ("Validator", "FAIL" if not r_draft.ok else "PASS", "PASS" if r_full.ok else "FAIL"),
    ]
    print("[3] BEFORE / AFTER")
    print(f"  {'':<48} {'BEFORE':>8}  {'AFTER':>16}")
    print(f"  {'':<48} {'draft':>8}  {'complete':>16}")
    print("  " + "-" * 74)
    for name, a, b in rows:
        print(f"  {name:<48} {str(a):>8}  {str(b):>16}")
    print()

    # ---- the attack tree -------------------------------------------------- #
    print("[4] Attack tree -- goal: exfiltrate customer data via the agent")
    print(f"  {'(OR = any child, AND = all children)':<62} {'draft':>7}  {'complete':>8}")
    print("  " + "-" * 80)
    for line in tree_side_by_side(tree, draft_ctl, full_ctl):
        print(line)
    open_draft = attack_paths(tree, draft_ctl)
    print(f"\n  Paths still open with the draft's controls ({len(open_draft)}):")
    for p in open_draft:
        print("    " + " + ".join(p))
    print("  With the complete model every path is blocked ON PAPER: each needs a step")
    print("  some listed control stops. Act II tests whether those controls actually work.")
    print()

    # ---- the risk-ranked control list ------------------------------------ #
    print("[5] Top 10 controls, risk-ranked (score = max likelihood x impact treated;")
    print("    illustrative 1-25 scale)")
    print(f"  {'#':>2}  {'id':<4} {'control':<46} {'score':>5}  verify in")
    print("  " + "-" * 76)
    for rc in rank_controls(full)[:10]:
        print(f"  {rc.rank:>2}  {rc.id:<4} {rc.name:<46} {rc.score:>5}  "
              f"{', '.join(rc.verify_in)}")
    print()

    # ---- artifacts -------------------------------------------------------- #
    paths = write_all(full, tree, full_ctl)
    write_all(draft, tree, draft_ctl, suffix="-draft")
    print("[6] Artifacts written (complete model; '-draft' versions alongside):")
    for name in paths:
        print(f"  {os.path.relpath(paths[name], HERE)}")
    print(RULE)


if __name__ == "__main__":
    main()
