"""
Lab 07 acceptance tests.

From the spec: the completeness validator passes on the complete model (all
components boundaried, all OWASP LLM and Agentic items mapped or accepted);
the diagrams render; tests assert the mapping coverage. From the lab brief:
the validator catches each gap type, every LLM01-LLM10 item is mapped to a
control, and the Mermaid output is plausible and contains every component.

Runs with `python -m pytest -q` or `python tests/test_threat_model.py`.
"""

import contextlib
import io
import os
import subprocess
import sys
import tempfile

LAB_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, LAB_DIR)

import attack_tree as at  # noqa: E402
import checklist  # noqa: E402
import reference_model as rm  # noqa: E402
import render  # noqa: E402
from model import Threat, ThreatModel  # noqa: E402
from owasp import ALL_OWASP, OWASP_AGENTIC_2026, OWASP_LLM_2025  # noqa: E402
from risk import rank_controls  # noqa: E402
from validate import GAP_TYPES, owasp_coverage, reference_node_ids, validate  # noqa: E402


def complete() -> ThreatModel:
    return rm.build_complete_model()


def only_gap(report, gap_type):
    """Assert the report has gaps of exactly one type."""
    for g in GAP_TYPES:
        if g == gap_type:
            assert report.count(g) > 0, f"expected a {g} gap, got none:\n{report.render()}"
        else:
            assert report.count(g) == 0, f"unexpected {g} gap:\n{report.render()}"


# --------------------------------------------------------------------------- #
# The complete model passes; the draft fails on every gap type it should
# --------------------------------------------------------------------------- #
def test_complete_model_passes_validator():
    report = validate(complete())
    assert report.ok, report.render()
    assert not report.notes, "reference architecture should be found and checked"


def test_draft_model_fails_with_every_expected_gap_type():
    r = validate(rm.build_draft_model())
    assert not r.ok
    assert r.count("boundary") == len(rm.DRAFT_UNZONED) == 4
    assert r.count("stride") == len(rm.DRAFT_UNREVIEWED_FLOWS) == 6
    assert r.count("owasp") == 10
    assert r.count("treatment") > 0
    assert r.count("reference") == 4          # SRC, PIPE, TOK, TRAIN
    assert r.count("structure") == 0          # the draft is incomplete, not malformed


def test_crossing_constant_matches_the_model():
    m = complete()
    assert tuple(f.id for f in m.crossing_flows()) == rm.CROSSING


def test_every_component_is_in_a_known_zone():
    m = complete()
    zone_ids = {z.id for z in m.zones}
    for c in m.components:
        assert c.zone in zone_ids, c.id


# --------------------------------------------------------------------------- #
# The validator catches each gap type in isolation
# --------------------------------------------------------------------------- #
def test_validator_catches_unboundaried_component():
    m = complete()
    m.component("memory").zone = None
    r = validate(m)
    # Unzoning memory also makes F11/F12 "cross" an unknown boundary, but
    # those flows are already reviewed, so only the boundary gap appears.
    only_gap(r, "boundary")
    assert "memory" in r.gaps["boundary"][0]


def test_validator_catches_unreviewed_crossing_flow():
    m = complete()
    # Strip F08 from every threat that reviews it.
    for t in m.threats:
        t.targets = tuple(x for x in t.targets if x != "F08")
    r = validate(m)
    only_gap(r, "stride")
    assert any("F08" in g and "T, I, D" in g for g in r.gaps["stride"])


def test_validator_catches_a_single_missing_stride_letter():
    m = complete()
    t03 = m.threat("T03")                     # the D (denial of service) hygiene row
    t03.targets = tuple(x for x in t03.targets if x != "F07")
    r = validate(m)
    only_gap(r, "stride")
    assert r.gaps["stride"] == [
        "flow F07 tools->customer_db crosses a boundary but has no STRIDE review for D"]


def test_stride_na_with_reason_counts_as_review_but_blank_reason_does_not():
    m = complete()
    assert validate(m).ok                     # F18/I is covered only by stride_na
    m.stride_na[("F18", "I")] = "   "
    r = validate(m)
    only_gap(r, "stride")
    assert "F18" in r.gaps["stride"][0]


def test_flow_inside_one_zone_needs_no_stride_review():
    m = complete()
    assert not m.crosses_boundary(next(f for f in m.flows if f.id == "F16"))
    assert not m.threats_on("F16")
    assert validate(m).ok


def test_validator_catches_unmapped_owasp_item():
    m = complete()
    m.controls = [c for c in m.controls if c.id != "C12"]   # the only LLM07 control
    for t in m.threats:
        t.controls = tuple(c for c in t.controls if c != "C12")
    r = validate(m)
    assert any(g.startswith("LLM07") for g in r.gaps["owasp"]), r.render()


def test_owasp_acceptance_requires_a_reason():
    m = complete()
    m.owasp_accepted["ASI07"] = ""
    r = validate(m)
    only_gap(r, "owasp")
    assert r.gaps["owasp"][0].startswith("ASI07")
    m.owasp_accepted["ASI07"] = "No inter-agent channel in this architecture."
    assert validate(m).ok


def test_validator_catches_untreated_threat():
    m = complete()
    m.threats.append(Threat("T99", "A threat nobody treated", "T", ("F01",), 3, 3))
    r = validate(m)
    only_gap(r, "treatment")
    assert "T99" in r.gaps["treatment"][0]
    m.threat("T99").accepted = "Accepted by the system owner: low value target."
    assert validate(m).ok


def test_validator_catches_structural_errors():
    m = complete()
    m.threats.append(Threat("T98", "Spoofing a data flow", "S", ("F01",), 1, 1, ("C17",)))
    m.threats.append(Threat("T97", "Ghost target", "T", ("nowhere",), 1, 1, ("C17",)))
    m.threats.append(Threat("T96", "Ghost control", "T", ("F01",), 1, 1, ("C99",)))
    r = validate(m)
    msgs = "\n".join(r.gaps["structure"])
    assert "does not apply to flow F01" in msgs       # STRIDE per element enforced
    assert "unknown element 'nowhere'" in msgs
    assert "missing control C99" in msgs


def test_validator_catches_reference_node_not_modeled():
    m = complete()
    m.component("memory").ref_ids = ()
    r = validate(m)
    only_gap(r, "reference")
    assert r.gaps["reference"] == ["reference-architecture node MEM has no component"]


def test_reference_architecture_is_parsed_from_the_real_file():
    ids = reference_node_ids()
    assert ids == {"SRC", "PIPE", "TOK", "TRAIN", "MODEL", "LOOP", "TOOLS", "MEM",
                   "USER", "SERVE", "OBS"}


# --------------------------------------------------------------------------- #
# OWASP coverage
# --------------------------------------------------------------------------- #
def test_owasp_llm_2025_catalog_is_correct():
    assert OWASP_LLM_2025 == {
        "LLM01": "Prompt Injection",
        "LLM02": "Sensitive Information Disclosure",
        "LLM03": "Supply Chain",
        "LLM04": "Data and Model Poisoning",
        "LLM05": "Improper Output Handling",
        "LLM06": "Excessive Agency",
        "LLM07": "System Prompt Leakage",
        "LLM08": "Vector and Embedding Weaknesses",
        "LLM09": "Misinformation",
        "LLM10": "Unbounded Consumption",
    }


def test_every_llm_top10_item_is_mapped_to_a_control():
    m = complete()
    cov = owasp_coverage(m)
    for item in OWASP_LLM_2025:
        assert cov[item], f"{item} has no control"
        assert item not in m.owasp_accepted, f"{item} must be treated, not accepted"


def test_every_agentic_item_is_mapped_or_accepted_with_reason():
    m = complete()
    cov = owasp_coverage(m)
    assert len(OWASP_AGENTIC_2026) == 10
    for item in OWASP_AGENTIC_2026:
        assert cov[item] or m.owasp_accepted.get(item, "").strip(), item
    # exactly one acceptance, and it is the inter-agent item
    assert set(m.owasp_accepted) == {"ASI07"}


def test_every_owasp_item_is_also_reached_by_a_threat_or_acceptance():
    """Mapping a control to an item is cheap; a threat that motivates it is not."""
    m = complete()
    cited = {item for t in m.threats for item in t.owasp}
    for item in ALL_OWASP:
        assert item in cited or item in m.owasp_accepted, item


# --------------------------------------------------------------------------- #
# Attack tree
# --------------------------------------------------------------------------- #
def test_and_or_semantics():
    leaf_a = at.Node("a", "a", controls=("X",))
    leaf_b = at.Node("b", "b")
    assert at.achievable(at.Node("or", "or", "OR", [leaf_a, leaf_b]), {"X"})
    assert not at.achievable(at.Node("and", "and", "AND", [leaf_a, leaf_b]), {"X"})
    assert at.attack_paths(at.Node("and", "and", "AND", [leaf_a, leaf_b])) == [("a", "b")]


def test_attack_paths_before_and_after():
    tree = at.exfiltration_tree()
    draft = {c.id for c in rm.build_draft_model().controls}
    full = {c.id for c in complete().controls}
    assert len(at.attack_paths(tree, set())) == 15        # 3x2x2 + 1 + 1 + 1
    assert len(at.attack_paths(tree, draft)) == 5
    assert at.attack_paths(tree, full) == []
    assert at.achievable(tree, draft) and not at.achievable(tree, full)


def test_attack_tree_controls_and_owasp_exist_in_the_model():
    m = complete()
    ids = {c.id for c in m.controls}
    for leaf in at.exfiltration_tree().leaves():
        assert set(leaf.controls) <= ids, leaf.id
        assert set(leaf.owasp) <= set(ALL_OWASP), leaf.id
        assert not leaf.entry or m.component(leaf.entry), leaf.id


def test_attack_tree_renders_text_and_mermaid():
    tree = at.exfiltration_tree()
    full = {c.id for c in complete().controls}
    text = at.render_text(tree, full)
    mmd = at.render_mermaid(tree, full)
    assert render.lint_mermaid(mmd) == []
    for n in tree.walk():
        assert n.id in text and f"    {n.id}" in mmd
    assert "AND" in text and "OR" in text


# --------------------------------------------------------------------------- #
# Diagrams and artifacts
# --------------------------------------------------------------------------- #
def test_dfd_mermaid_is_plausible_and_contains_every_component():
    m = complete()
    mmd = render.render_dfd(m, at.exfiltration_tree())
    assert render.lint_mermaid(mmd) == [], render.lint_mermaid(mmd)
    for c in m.components:
        assert f"{c.id}" in mmd and (f"    {c.id}[" in mmd or f"    {c.id}(" in mmd), c.id
    for z in m.zones:
        assert f'subgraph Z_{z.id}["' in mmd
    for f in m.flows:
        assert f'"{f.id}: ' in mmd
    assert mmd.count("subgraph ") == mmd.count("\n    end")


def test_draft_dfd_also_renders_with_unzoned_components_highlighted():
    mmd = render.render_dfd(rm.build_draft_model(), at.exfiltration_tree())
    assert render.lint_mermaid(mmd) == []
    assert "class memory,rag_corpus,model_registry,logs unzoned" in mmd


def test_lint_catches_broken_mermaid():
    assert render.lint_mermaid("graph LR\n  subgraph X\n  a --> b\n")    # unclosed, undefined
    assert render.lint_mermaid("nonsense\n")
    assert render.lint_mermaid('flowchart LR\n  end["x"]\n')              # reserved id


def test_risk_ranking_is_descending_and_uses_likelihood_times_impact():
    m = complete()
    ranked = rank_controls(m)
    assert [r.score for r in ranked] == sorted((r.score for r in ranked), reverse=True)
    assert ranked[0].score == 25
    assert all(t.risk == t.likelihood * t.impact for t in m.threats)
    assert all(1 <= t.likelihood <= 5 and 1 <= t.impact <= 5 for t in m.threats)
    assert len(ranked) == len(m.controls)


def test_write_all_and_checklist_round_trip():
    m = complete()
    full = {c.id for c in m.controls}
    with tempfile.TemporaryDirectory() as d:
        paths = render.write_all(m, at.exfiltration_tree(), full, out_dir=d)
        for name in ("architecture-threats.mmd", "attack-tree.mmd", "attack-tree.txt",
                     "controls.md", "threats.md", "controls.json"):
            assert os.path.getsize(paths[name]) > 0, name
        md = paths["controls.md"]
        state = checklist.parse_controls_md(md)
        assert set(state) == full and not any(state.values())
        # Simulate the Post 17 red-team ticking C02 after verifying it.
        with open(md, encoding="utf-8") as fh:
            text = fh.read().replace("- [ ] **C02**", "- [x] **C02**")
        with open(md, "w", encoding="utf-8") as fh:
            fh.write(text)
        items = checklist.load(verified=checklist.parse_controls_md(md))
        assert [i.id for i in items if i.verified] == ["C02"]
        assert abs(checklist.coverage(items) - 1 / len(items)) < 1e-9


def test_checklist_is_importable_and_ranked():
    items = checklist.load()
    assert len(items) == 17
    assert [i.rank for i in items] == list(range(1, 18))
    assert all(i.verify_in for i in items), "every control names the lab that tests it"


# --------------------------------------------------------------------------- #
# Entrypoints
# --------------------------------------------------------------------------- #
def test_validate_cli_exit_codes():
    ok = subprocess.run([sys.executable, "validate.py"], cwd=LAB_DIR,
                        capture_output=True, text=True)
    bad = subprocess.run([sys.executable, "validate.py", "--draft"], cwd=LAB_DIR,
                         capture_output=True, text=True)
    assert ok.returncode == 0 and "PASS" in ok.stdout
    assert bad.returncode == 1 and "FAIL" in bad.stdout


def test_run_demo_prints_before_fail_and_after_pass():
    import run_demo
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run_demo.main()
    out = buf.getvalue()
    assert "FAIL -- " in out and "PASS -- threat model is complete" in out
    assert "Attack paths to the goal still open (of 15)" in out
    assert "Top 10 controls" in out


if __name__ == "__main__":
    failed = 0
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
