"""
Definition of done for the whole capstone, as a checklist that can check itself.

Each item names the post that satisfies it. An item with a `check` is
evaluated live every time this runs; an item without one stays open until
the post that delivers it adds a check here. So the checklist cannot drift
into "done" by someone ticking a box: in Post 13 only the kickoff items are
checkable, and they are checked, not asserted.

The items mirror the acceptance criteria of Posts 13-18 in the series spec.

Run:  python dod.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Callable

HERE = os.path.dirname(os.path.abspath(__file__))
STUB_FOLDERS = ("corpus", "persona", "memory", "hands", "guardrails", "redteam")


@dataclass
class Item:
    post: int
    part: str                      # which of the five parts, or "kickoff"/"ship"
    text: str
    check: Callable[[], bool] | None = None

    def status(self) -> str:
        if self.check is None:
            return "open"
        try:
            return "done" if self.check() else "FAIL"
        except Exception:  # noqa: BLE001 -- a crashing check is a failing check
            return "FAIL"


# --------------------------------------------------------------------------- #
# Live checks for Post 13 (imports are local so `python dod.py` stays cheap
# and a broken module shows up as a FAIL row instead of a crash)
# --------------------------------------------------------------------------- #
def _charter_ok() -> bool:
    import charter
    return not charter.validate(charter.load())


def _charter_core_sections() -> bool:
    import charter
    c = charter.load()
    return all(c.rules(s) for s in ("MAY SAY", "MUST REFUSE", "MUST ESCALATE",
                                    "DISCLOSURE"))


def _disclosure_locked() -> bool:
    import charter
    c = charter.load()
    return bool(c.disclosure_text) and all(r.locked for r in c.rules("DISCLOSURE"))


def _threat_model_ok() -> bool:
    import twin_threat_model as ttm
    return ttm.validate(ttm.build_twin_model()).ok


def _components_mapped() -> bool:
    import twin_threat_model as ttm
    return not ttm.unmapped_components(ttm.build_twin_model())


def _charter_matches_model() -> bool:
    import charter
    import twin_threat_model as ttm
    return not ttm.charter_gaps(charter.load(), ttm.build_twin_model())


def folder_is_stub(path: str) -> bool:
    """A part folder is a stub while it holds nothing but its README."""
    return [f for f in os.listdir(path) if not f.startswith((".", "__pycache__"))] \
        == ["README.md"]


def _ci_layout() -> bool:
    """capstone/tests has tests and each part folder has its README.

    CI runs pytest in every capstone/*/ folder that has tests, so a part
    folder must not gain tests before the code they test. Stubs hold only
    their README (see folder_is_stub); the test suite asserts that.
    """
    tests_dir = os.path.join(HERE, "tests")
    if not any(f.startswith("test_") and f.endswith(".py") for f in os.listdir(tests_dir)):
        return False
    for folder in STUB_FOLDERS:
        path = os.path.join(HERE, folder)
        if not os.path.isfile(os.path.join(path, "README.md")):
            return False
    return True


ITEMS = [
    # ---- Post 13: kickoff ------------------------------------------------- #
    Item(13, "kickoff", "Charter parses and is well-formed (charter.py)", _charter_ok),
    Item(13, "kickoff", "Charter covers MAY SAY / MUST REFUSE / MUST ESCALATE / DISCLOSURE",
         _charter_core_sections),
    Item(13, "kickoff", "Disclosure text present and every DISCLOSURE rule locked",
         _disclosure_locked),
    Item(13, "kickoff", "Twin threat model passes Lab 07's validator", _threat_model_ok),
    Item(13, "kickoff", "Every twin component is mapped to at least one control",
         _components_mapped),
    Item(13, "kickoff", "Charter tools and policies match the threat model",
         _charter_matches_model),
    Item(13, "kickoff", "Capstone CI layout: tests/ present, part folders stubbed",
         _ci_layout),
    # ---- Post 14: corpus and voice ---------------------------------------- #
    Item(14, "corpus", "Published-only corpus via Lab 01; source + hash on every chunk"),
    Item(14, "corpus", "Offline retrieval index (stdlib TF-IDF path) over the chunks"),
    Item(14, "corpus", "Every answer cites real source chunks"),
    Item(14, "corpus", "Out-of-corpus question -> \"I don't have that\", never a guess"),
    Item(14, "persona", "Persona contract from the charter + few-shot voice exemplars"),
    # ---- Post 15: memory and hands ---------------------------------------- #
    Item(15, "memory", "Memory persists across two runs (Lab 09 MemoryAgent, defenses on)"),
    Item(15, "memory", "Memory namespaced per visitor; no cross-visitor recall"),
    Item(15, "hands", "Only the charter's ACTIONS tools exist; each scoped and logged"),
    Item(15, "hands", "Out-of-scope tool request is refused"),
    Item(15, "hands", "draft_reply and book_call queue for author approval"),
    # ---- Post 16: productionize ------------------------------------------- #
    Item(16, "guardrails", "Eval scorecard: faithfulness + voice on a golden set (Lab 04)"),
    Item(16, "guardrails", "Rate and budget limits enforce at the gateway (Lab 05)"),
    Item(16, "guardrails", "Tracing + cost dashboard (Lab 06)"),
    Item(16, "guardrails", "Kill-switch halts responses immediately"),
    Item(16, "guardrails", "MUST ESCALATE requests are routed to the author, not answered"),
    Item(16, "guardrails", "Dockerfile + ask-my-twin endpoint behind the guardrails"),
    # ---- Post 17: red-team ------------------------------------------------ #
    Item(17, "redteam", "Labs 08/09/10/12 attacks succeed against the un-hardened twin"),
    Item(17, "redteam", "Hardened: ~0 success on leak, off-character, tool abuse, impersonation"),
    Item(17, "redteam", "Lab 07 checklist graded: a control ticks only when its attack ~0"),
    Item(17, "redteam", "security-card.md: capabilities, refusals, residual risk, disclosure"),
    # ---- Post 18: ship ---------------------------------------------------- #
    Item(18, "ship", "Public path: disclosed, guardrailed twin; smoke test passes"),
    Item(18, "ship", "Disclosure is non-removable on the public path"),
    Item(18, "ship", "Kill-switch reachable by the author from the public deployment"),
    Item(18, "ship", "capstone/README.md technical writeup + reinvention-playbook.md"),
]


def evaluate(items: list[Item] = ITEMS) -> list[tuple[Item, str]]:
    return [(i, i.status()) for i in items]


def render(results: list[tuple[Item, str]] | None = None) -> str:
    results = results if results is not None else evaluate()
    mark = {"done": "[x]", "open": "[ ]", "FAIL": "[!]"}
    lines = []
    post = None
    for item, status in results:
        if item.post != post:
            post = item.post
            lines.append(f"  Post {post}")
        lines.append(f"    {mark[status]} {item.part:<10} {item.text}")
    done = sum(s == "done" for _, s in results)
    failed = sum(s == "FAIL" for _, s in results)
    lines.append(f"  {done}/{len(results)} done, {failed} failing, "
                 f"{len(results) - done - failed} open")
    return "\n".join(lines)


def main() -> int:
    results = evaluate()
    print("Capstone definition of done  ([x] done  [ ] open  [!] failing)")
    print(render(results))
    return 1 if any(s == "FAIL" for _, s in results) else 0


if __name__ == "__main__":
    sys.exit(main())
