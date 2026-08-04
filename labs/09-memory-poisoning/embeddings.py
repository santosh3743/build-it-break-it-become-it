"""Deterministic, dependency-free text embeddings.

We use a hashing bag-of-words vectorizer so the whole lab runs offline with no
model download and no API key. It is intentionally simple: the point of this lab
is the *memory mechanics* of the attack, not embedding quality. In a real system
you would swap this for a sentence-transformer or an API embedding — the attack
and the defenses are unchanged.
"""

from __future__ import annotations

import math
import re
from hashlib import blake2b

DIM = 256


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def embed(text: str, dim: int = DIM) -> list[float]:
    """Hash each token into a fixed-size vector, then L2-normalize."""
    vec = [0.0] * dim
    for tok in tokenize(text):
        idx = int.from_bytes(blake2b(tok.encode(), digest_size=8).digest(), "big") % dim
        vec[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two L2-normalized vectors (== dot product)."""
    return sum(x * y for x, y in zip(a, b))
