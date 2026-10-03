"""
Import Lab 02's engine (autograd, GPT, trainer) and Lab 01's tokenizer.

Sibling lab folders start with a digit, so they are not importable as normal
package names. Every lab from here on loads its dependencies by file path
through this helper, which keeps the "reuse the earlier lab, don't reimplement
it" rule cheap to follow.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from types import ModuleType

LABS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_cache: dict[str, ModuleType] = {}


def load(lab_folder: str, module_name: str) -> ModuleType:
    """Load `labs/<lab_folder>/<module_name>.py` as an importable module."""
    key = f"{lab_folder}/{module_name}"
    if key in _cache:
        return _cache[key]

    lab_dir = os.path.join(LABS_DIR, lab_folder)
    path = os.path.join(lab_dir, f"{module_name}.py")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"This lab builds on {lab_folder}, but {path} is missing.\n"
            "Clone the whole repo rather than a single lab folder."
        )

    # The dependency's own imports (e.g. Lab 02's `from autograd import ...`)
    # resolve relative to its folder, so it has to be on sys.path.
    if lab_dir not in sys.path:
        sys.path.insert(0, lab_dir)

    spec = importlib.util.spec_from_file_location(f"{lab_folder}_{module_name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _cache[key] = module
    return module


def lab02_engine():
    """(autograd, model, train) from Lab 02."""
    return load("02-train-slm", "autograd"), load("02-train-slm", "model"), \
        load("02-train-slm", "train")


def lab01_pipeline():
    """Lab 01's data pipeline -- we reuse its Tokenizer."""
    return load("01-data-engine", "pipeline")
