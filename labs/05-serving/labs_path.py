"""
Load an earlier lab's module by file path.

Sibling lab folders start with a digit, so they are not importable as normal
package names. This is the same helper Lab 03 uses, trimmed to what Lab 05
needs: Lab 02's training configs, so the KV-cache table can include the model
you trained yourself.

Lab 05 does not *require* Lab 02. If the folder is missing, `lab02_configs()`
returns None and the KV-cache table simply skips that row.
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
            f"This lab can use {lab_folder}, but {path} is missing.\n"
            "Clone the whole repo rather than a single lab folder."
        )

    # The dependency's own imports (e.g. Lab 02's `from model import ...`)
    # resolve relative to its folder, so it has to be on sys.path.
    if lab_dir not in sys.path:
        sys.path.insert(0, lab_dir)

    spec = importlib.util.spec_from_file_location(f"{lab_folder}_{module_name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _cache[key] = module
    return module


def lab02_configs():
    """Lab 02's `TINY` and `REAL` TrainConfigs, or None if Lab 02 is absent."""
    try:
        train = load("02-train-slm", "train")
    except FileNotFoundError:
        return None
    return train.CONFIGS
