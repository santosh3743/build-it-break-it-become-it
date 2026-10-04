"""
Load earlier labs by file path, and let later labs load THIS lab as a package.

Copied from Lab 03 and adapted in two ways:

1. `load()` does NOT push the dependency's folder onto sys.path. The only thing
   Lab 06 borrows is Lab 01's `pipeline.py`, which imports nothing but the
   standard library, so there is no reason to let Lab 01's module names
   (`pipeline`, `sample_corpus`) shadow anyone else's.

2. `load_package()` is new. Labs 08, 10, 12 and the capstone reuse Lab 06's
   modules. Those modules have generic names (`app`, `cost`, `guardrails`) that
   a later lab may also use, so they are mounted under one unique package name
   (e.g. `lab06`) instead of as top-level modules. See the README section
   "Reusing this lab".
"""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import os
import sys
from types import ModuleType

LABS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_cache: dict[str, ModuleType] = {}


def load(lab_folder: str, module_name: str) -> ModuleType:
    """Load `labs/<lab_folder>/<module_name>.py` as a uniquely named module."""
    key = f"{lab_folder}/{module_name}"
    if key in _cache:
        return _cache[key]

    path = os.path.join(LABS_DIR, lab_folder, f"{module_name}.py")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"This lab builds on {lab_folder}, but {path} is missing.\n"
            "Clone the whole repo rather than a single lab folder."
        )

    spec = importlib.util.spec_from_file_location(f"{lab_folder}_{module_name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _cache[key] = module
    return module


def load_package(lab_folder: str, package_name: str) -> ModuleType:
    """Mount `labs/<lab_folder>/` as an importable package called `package_name`.

    After `load_package("06-observability", "lab06")` you can write
    `from lab06 import guardrails, telemetry` anywhere. No __init__.py is needed:
    we build a namespace-package spec by hand and point it at the folder.
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


def lab01_pipeline() -> ModuleType:
    """Lab 01's data pipeline -- we reuse its `scrub_pii`."""
    return load("01-data-engine", "pipeline")
