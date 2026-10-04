"""
Load earlier labs by file path, from the capstone.

Adapted from labs/03-post-training/labs_path.py. Two differences:

1. The capstone sits at the repo root (`capstone/`), one level above the lab
   folders, so LABS_DIR is `<repo>/labs`, not this file's parent.

2. Lab 07's modules import each other by bare name (`from model import
   ThreatModel`). Putting labs/07-threat-model on sys.path permanently would
   work today, but it is fragile: Lab 02 also has a `model.py`, Lab 07 and the
   capstone both have a `run_demo.py`, and later capstone posts load more labs.
   So `load_group()` loads a set of sibling modules by file path, and while
   each one executes it temporarily aliases its siblings under their bare
   names in sys.modules. After loading, the bare names are restored to
   whatever they were before. Net effect: Lab 07's `validate` sees the SAME
   `model` module object we hand to the capstone (so the dataclasses match),
   and nothing leaks into the global import namespace.

`load_package()` is copied from labs/06-observability/labs_path.py for the
posts that reuse Lab 06 as a package (Post 16: `from lab06 import telemetry`).
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import sys
from types import ModuleType, SimpleNamespace

CAPSTONE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(CAPSTONE_DIR)
LABS_DIR = os.path.join(REPO_DIR, "labs")

_cache: dict[str, ModuleType] = {}
_MISSING = object()


def _lab_file(lab_folder: str, module_name: str) -> str:
    path = os.path.join(LABS_DIR, lab_folder, f"{module_name}.py")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"The capstone builds on labs/{lab_folder}, but {path} is missing.\n"
            "Clone the whole repo rather than the capstone folder alone."
        )
    return path


def load_group(lab_folder: str, module_names: list[str]) -> SimpleNamespace:
    """Load `labs/<lab_folder>/<name>.py` for each name, in dependency order.

    Modules are registered as `<lab_folder>_<name>` (unique) and are visible
    under their bare names only while the group is loading. Returns a
    namespace: `ns.model`, `ns.validate`, ...
    """
    loaded: dict[str, ModuleType] = {}
    saved: dict[str, object] = {}
    try:
        for name in module_names:
            key = f"{lab_folder}/{name}"
            if key in _cache:
                mod = _cache[key]
            else:
                path = _lab_file(lab_folder, name)
                spec = importlib.util.spec_from_file_location(
                    f"{lab_folder.replace('-', '_')}_{name}", path)
                mod = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = mod
                spec.loader.exec_module(mod)
                _cache[key] = mod
            loaded[name] = mod
            # Make it importable by bare name for the modules loaded after it.
            if name not in saved:
                saved[name] = sys.modules.get(name, _MISSING)
            sys.modules[name] = mod
    finally:
        for name, previous in saved.items():
            if previous is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return SimpleNamespace(**loaded)


def lab07() -> SimpleNamespace:
    """Lab 07's threat-model toolkit: model, owasp, validate, risk, checklist,
    plus reference_model (the capstone reuses some of its controls by id)."""
    return load_group("07-threat-model",
                      ["model", "owasp", "validate", "risk", "checklist",
                       "reference_model"])


def load_package(lab_folder: str, package_name: str) -> ModuleType:
    """Mount `labs/<lab_folder>/` as an importable package called `package_name`.

    Copied from Lab 06. After `load_package("06-observability", "lab06")` you
    can write `from lab06 import guardrails, telemetry`.
    """
    if package_name in sys.modules:
        return sys.modules[package_name]
    lab_dir = os.path.join(LABS_DIR, lab_folder)
    if not os.path.isdir(lab_dir):
        raise FileNotFoundError(f"{lab_dir} is missing. Clone the whole repo.")
    spec = importlib.machinery.ModuleSpec(package_name, None, is_package=True)
    spec.submodule_search_locations = [lab_dir]
    package = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = package
    return package
