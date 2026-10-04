"""
Capstone kickoff (Post 13) acceptance tests.

From the spec: the charter parses and covers say/refuse/escalate; the threat
model maps every twin component to a control; the DoD checklist prints. From
the kickoff task: disclosure is covered and non-removable, the threat model
passes Lab 07's own validator, and the part folders hold no tests yet.

Runs with `python -m pytest -q` (from capstone/ or capstone/tests/) or
`python tests/test_capstone_kickoff.py`.
"""

import contextlib
import io
import os
import sys

CAPSTONE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CAPSTONE_DIR)

import charter  # noqa: E402
import dod  # noqa: E402
import labs_path  # noqa: E402
import twin_threat_model as ttm  # noqa: E402

L07 = ttm.L07


def full_charter():
    return charter.load()


def full_text():
    with open(charter.CHARTER_PATH, encoding="utf-8") as fh:
        return fh.read()


# --------------------------------------------------------------------------- #
# Charter
# --------------------------------------------------------------------------- #
def test_charter_parses_and_is_well_formed():
    c = full_charter()
    assert charter.validate(c) == []
    for section in charter.SECTIONS:
        assert c.rules(section), f"{section} is empty"


def test_charter_covers_say_refuse_escalate_disclosure():
    c = full_charter()
    for section in ("MAY SAY", "MUST REFUSE", "MUST ESCALATE", "DISCLOSURE"):
        assert len(c.rules(section)) >= 1
        assert charter.REQUIRED_TAGS[section] <= c.tags(section)
    assert {"impersonation", "private-data", "unsupported-opinion"} <= c.tags("MUST REFUSE")


def test_disclosure_is_locked_and_says_ai():
    c = full_charter()
    assert all(r.locked for r in c.rules("DISCLOSURE"))
    assert "AI twin" in c.disclosure_text
    # Unlocking any disclosure rule is a validation failure.
    unlocked = full_text().replace("- [D02] (locked)", "- [D02]")
    problems = charter.validate(charter.parse(unlocked))
    assert any("D02" in p and "locked" in p for p in problems)


def test_charter_rule_ids_unique_and_prefixed():
    c = full_charter()
    ids = [r.id for s in charter.SECTIONS for r in c.rules(s)]
    assert len(ids) == len(set(ids))
    dup = full_text().replace("- [S02]", "- [S01]")
    assert any("used twice" in p for p in charter.validate(charter.parse(dup)))


def test_actions_side_effects_need_author_and_default_deny():
    c = full_charter()
    tools = c.allowed_tools()
    assert tools["faq_answer"] == "none"
    assert tools["draft_reply"] == "author" and tools["book_call"] == "author"
    assert any(r.attrs["approval"] == "forbidden" for r in c.rules("ACTIONS"))
    loosened = full_text().replace("tool: draft_reply | approval: author",
                                   "tool: draft_reply | approval: none")
    assert any("draft_reply" in p and "approval: none" in p
               for p in charter.validate(charter.parse(loosened)))


def test_data_has_allow_and_deny():
    kinds = [r.attrs.get("kind") for r in full_charter().rules("DATA")]
    assert "allow" in kinds and "deny" in kinds and None not in kinds


def test_draft_charter_fails_for_the_documented_reasons():
    problems = charter.validate(charter.parse(charter.draft_text(full_text())))
    joined = "\n".join(problems)
    assert "MUST ESCALATE' is missing" in joined
    assert "DISCLOSURE' is missing" in joined
    assert "disclosure-text" in joined
    assert "book_call" in joined
    assert "default-deny" in joined


def test_charter_contains_policy_not_biography():
    # The charter must not carry facts about the author; the corpus does that.
    c = full_charter()
    assert set(c.headers) == set(charter.REQUIRED_HEADERS)
    assert c.unknown_sections == []


# --------------------------------------------------------------------------- #
# Threat model (Lab 07 reused, not reimplemented)
# --------------------------------------------------------------------------- #
def test_threat_model_uses_lab07_classes_and_validator():
    m = ttm.build_twin_model()
    assert type(m) is L07.model.ThreatModel
    assert all(type(c) is L07.model.Component for c in m.components)
    report = L07.validate.validate(m, reference_path=ttm.TWIN_MMD)
    assert report.ok, report.render()


def test_every_twin_component_is_boundaried_and_mapped_to_a_control():
    m = ttm.build_twin_model()
    expected = {"visitor", "public_endpoint", "guardrail_gateway", "twin_agent_loop",
                "llm", "persona_contract", "corpus_index", "memory_store", "tool_faq",
                "tool_draft_reply", "tool_book_call", "kill_switch", "logs", "author"}
    assert {c.id for c in m.components} == expected
    assert all(c.zone for c in m.components)
    mapping = ttm.component_controls(m)
    assert ttm.unmapped_components(m) == []
    assert all(mapping[cid] for cid in expected)


def test_every_owasp_item_mapped_or_accepted():
    m = ttm.build_twin_model()
    cov = L07.validate.owasp_coverage(m)
    for item in L07.owasp.ALL_OWASP:
        assert cov[item] or m.owasp_accepted.get(item, "").strip(), item
    for item in L07.owasp.OWASP_LLM_2025:
        assert cov[item], f"{item} should be treated by a control, not accepted"
    assert set(m.owasp_accepted) == {"ASI07"}


def test_reused_controls_are_lab07s_by_id():
    m = ttm.build_twin_model()
    ref = {c.id: c for c in L07.reference_model.CONTROLS}
    for cid in ttm.REUSED_CONTROLS:
        twin_c = m.control(cid)
        assert twin_c.name == ref[cid].name and twin_c.owasp == ref[cid].owasp
        assert all(v.startswith("Post ") for v in twin_c.verify_in)
    assert {"C18", "C22"} <= {c.id for c in m.controls}


def test_only_accepted_threat_is_documented():
    m = ttm.build_twin_model()
    accepted = [t for t in m.threats if t.accepted]
    assert [t.id for t in accepted] == ["T38"]
    assert all(t.controls or t.accepted for t in m.threats)


def test_draft_threat_model_fails_each_gap_type():
    r = ttm.validate(ttm.build_draft_model())
    assert not r.ok
    for gap in ("boundary", "stride", "owasp", "treatment", "reference"):
        assert r.count(gap) > 0, gap
    assert r.count("structure") == 0
    assert ttm.unmapped_components(ttm.build_draft_model())


def test_validator_catches_a_new_unmodeled_tool():
    # Add a tool to the system without threat-modeling it: CI must fail.
    m = ttm.build_twin_model()
    m.components.append(L07.model.Component("tool_send_email", "send_email", "process",
                                            "tools"))
    m.flows.append(L07.model.Flow("F99", "twin_agent_loop", "tool_send_email", "email"))
    r = ttm.validate(m)
    assert r.count("stride") == 1
    assert "tool_send_email" in ttm.unmapped_components(m)


def test_charter_and_threat_model_agree():
    assert ttm.charter_gaps(full_charter(), ttm.build_twin_model()) == []
    extra = full_text().replace(
        "- [A05] tool: any-other",
        "- [A06] tool: post_social | approval: author | scope: post on social media.\n"
        "- [A05] tool: any-other")
    gaps = ttm.charter_gaps(charter.parse(extra), ttm.build_twin_model())
    assert any("post_social" in g for g in gaps)


def test_lab07_bare_module_names_do_not_leak():
    # labs_path loads Lab 07 without leaving `model`, `validate`... importable.
    for name in ("model", "validate", "risk", "checklist", "owasp", "reference_model"):
        mod = sys.modules.get(name)
        assert mod is None or not getattr(mod, "__file__", "").startswith(
            os.path.join(labs_path.LABS_DIR, "07-threat-model"))


def test_checklist_ranks_every_twin_control():
    items = ttm.checklist(ttm.build_twin_model())
    assert len(items) == 24
    assert [i.rank for i in items] == list(range(1, 25))
    assert L07.checklist.coverage(items) == 0.0   # nothing verified until Post 17


# --------------------------------------------------------------------------- #
# Definition of done, CI layout, demo
# --------------------------------------------------------------------------- #
def test_dod_prints_with_post13_items_done():
    results = dod.evaluate()
    for item, status in results:
        if item.post == 13:
            assert status == "done", item.text
        else:
            assert status == "open", item.text
    assert {i.post for i, _ in results} == {13, 14, 15, 16, 17, 18}
    out = dod.render(results)
    assert "Post 13" in out and "Post 18" in out and "[x]" in out


def test_part_folders_exist_and_hold_no_premature_tests():
    for folder in dod.STUB_FOLDERS:
        path = os.path.join(CAPSTONE_DIR, folder)
        assert os.path.isfile(os.path.join(path, "README.md")), folder
        has_tests = os.path.isdir(os.path.join(path, "tests")) or any(
            f.startswith("test_") and f.endswith(".py") for f in os.listdir(path))
        if has_tests:  # allowed only once the post adds the code they test
            assert not dod.folder_is_stub(path), folder


def test_run_demo_prints_before_fail_and_after_pass():
    import run_demo
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run_demo.main()
    out = buf.getvalue()
    assert "FAIL -- " in out
    assert "charter:      PASS" in out
    assert "threat model: PASS -- threat model is complete" in out
    assert "cross-check:  PASS" in out
    assert "Ready to build against" in out and "Definition of done" in out
    assert "Charter summary" in out


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
