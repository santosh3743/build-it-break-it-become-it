"""
Capstone kickoff demo (Post 13): is the twin's spec ready to build against?

BEFORE  a first-draft charter (no escalation, no disclosure, book_call with no
        approval) and a first-draft threat model (no author, no kill-switch,
        stateful parts unzoned, chatbot-only controls).
AFTER   the real twin-charter.md and the complete twin threat model.

Both are judged by code: charter.py validates the charter, Lab 07's validator
(loaded by file path) judges the threat model, and twin_threat_model.py
cross-checks the two against each other. Then it prints the capstone's
definition-of-done checklist and the charter summary.

Zero arguments, standard library only, runs in well under a second.
"""

from __future__ import annotations

import charter
import dod
import twin_threat_model as ttm

RULE = "=" * 78
OWASP = ttm.L07.owasp


def owasp_unmapped(m, catalog) -> tuple[int, int]:
    cov = ttm.L07.validate.owasp_coverage(m)
    unmapped = [i for i in catalog if not cov[i] and not m.owasp_accepted.get(i)]
    accepted = [i for i in catalog if not cov[i] and m.owasp_accepted.get(i)]
    return len(unmapped), len(accepted)


def main() -> None:
    with open(charter.CHARTER_PATH, encoding="utf-8") as fh:
        full_text = fh.read()
    c_draft = charter.parse(charter.draft_text(full_text))
    c_full = charter.parse(full_text)
    p_draft, p_full = charter.validate(c_draft), charter.validate(c_full)

    m_draft, m_full = ttm.build_draft_model(), ttm.build_twin_model()
    r_draft, r_full = ttm.validate(m_draft), ttm.validate(m_full)
    g_draft, g_full = ttm.charter_gaps(c_draft, m_draft), ttm.charter_gaps(c_full, m_full)
    u_draft, u_full = ttm.unmapped_components(m_draft), ttm.unmapped_components(m_full)

    print(RULE)
    print("Capstone kickoff -- Reborn: the digital twin's charter and threat model")
    print(RULE)
    print(f"Twin: {len(m_full.components)} components, {len(m_full.flows)} data flows, "
          f"{len(m_full.zones)} trust zones, {len(m_full.crossing_flows())} flows cross "
          f"a trust boundary.\n")

    print("[1] BEFORE -- first-draft charter")
    for p in p_draft:
        print(f"  [problem  ] {p}")
    print(f"  RESULT: FAIL -- {len(p_draft)} problem(s)" if p_draft else "  RESULT: PASS")
    print()
    print("[2] BEFORE -- Lab 07's validator on the first-draft twin threat model")
    print(r_draft.render())
    for gap in g_draft:
        print(f"  [charter  ] {gap}")
    print(f"  Components with no control: {', '.join(u_draft) if u_draft else 'none'}")
    print()
    print("[3] AFTER -- twin-charter.md and the complete twin threat model")
    print("  charter:      " + ("PASS -- charter is well-formed and complete" if not p_full
                                else f"FAIL -- {len(p_full)} problem(s)"))
    print("  threat model:" + r_full.render().replace("  RESULT:", ""))
    print("  cross-check:  " + ("PASS -- charter and threat model agree" if not g_full
                                else f"FAIL -- {len(g_full)} gap(s)"))
    print()

    # ---- the before/after table (the screenshot) ------------------------- #
    llm_d, _ = owasp_unmapped(m_draft, OWASP.OWASP_LLM_2025)
    llm_f, _ = owasp_unmapped(m_full, OWASP.OWASP_LLM_2025)
    asi_d, _ = owasp_unmapped(m_draft, OWASP.OWASP_AGENTIC_2026)
    asi_f, asi_f_acc = owasp_unmapped(m_full, OWASP.OWASP_AGENTIC_2026)
    accepted_threats = sum(1 for t in m_full.threats if t.accepted)

    def n_sections(c):
        return sum(1 for s in charter.SECTIONS if c.rules(s))

    rows = [
        ("Charter sections present (of 6)", n_sections(c_draft), n_sections(c_full)),
        ("Charter problems", len(p_draft), len(p_full)),
        ("Disclosure locked and non-removable",
         "no" if not c_draft.rules("DISCLOSURE") else "yes",
         "yes" if all(r.locked for r in c_full.rules("DISCLOSURE")) else "no"),
        ("Tools that need the author's approval",
         sum(1 for a in c_draft.allowed_tools().values() if a == "author"),
         sum(1 for a in c_full.allowed_tools().values() if a == "author")),
        ("Components on the diagram", len(m_draft.components), len(m_full.components)),
        ("Components with no trust boundary", r_draft.count("boundary"),
         r_full.count("boundary")),
        ("Components with no control", len(u_draft), len(u_full)),
        ("Boundary-crossing flows not STRIDE-reviewed", r_draft.count("stride"),
         r_full.count("stride")),
        ("OWASP LLM 2025 items unmapped (of 10)", llm_d, llm_f),
        ("OWASP Agentic 2026 items unmapped (of 10)", asi_d,
         f"{asi_f} ({asi_f_acc} accepted)"),
        ("Threats with no control", r_draft.count("treatment"),
         f"{r_full.count('treatment')} ({accepted_threats} accepted)"),
        ("Diagram nodes not modeled", r_draft.count("reference"), r_full.count("reference")),
        ("Charter <-> threat model disagreements", len(g_draft), len(g_full)),
        ("Controls identified", len(m_draft.controls), len(m_full.controls)),
        ("Ready to build against",
         "NO" if (p_draft or not r_draft.ok or g_draft or u_draft) else "YES",
         "YES" if not (p_full or not r_full.ok or g_full or u_full) else "NO"),
    ]
    print("[4] BEFORE / AFTER")
    print(f"  {'':<46} {'BEFORE':>8}  {'AFTER':>18}")
    print(f"  {'':<46} {'draft':>8}  {'complete':>18}")
    print("  " + "-" * 74)
    for name, a, b in rows:
        print(f"  {name:<46} {str(a):>8}  {str(b):>18}")
    print()

    # ---- the twin-specific controls, ranked ------------------------------ #
    print("[5] Top 8 twin controls, risk-ranked by Lab 07's checklist "
          "(illustrative 1-25 scale)")
    print(f"  {'#':>2}  {'id':<4} {'control':<40} {'score':>5}  verified on the twin in")
    print("  " + "-" * 76)
    for item in ttm.checklist(m_full)[:8]:
        print(f"  {item.rank:>2}  {item.id:<4} {item.name:<40} {item.score:>5}  "
              f"{', '.join(item.verify_in)}")
    print()

    print("[6] Definition of done  ([x] done  [ ] open  [!] failing)")
    print(dod.render())
    print()
    print("[7] Charter summary")
    print(charter.summary(c_full))
    print(RULE)


if __name__ == "__main__":
    main()
