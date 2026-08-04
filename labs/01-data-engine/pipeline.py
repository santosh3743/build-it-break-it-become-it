"""
Lab 01 — The Data Engine
A minimal, dependency-free pretraining data pipeline.

Stages (each removes one nameable failure mode):
  normalize -> near-dedup (MinHash+LSH) -> quality filter
  -> decontaminate (vs eval set) -> PII scrub -> tokenize -> shard + datasheet

Pure Python standard library. No numpy, no torch. Runs on a laptop, CPU-only.
Deterministic: all hashing is seeded, so runs are reproducible.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field

# Mersenne prime 2^61 - 1, used as the modulus for the MinHash permutations.
_MERSENNE = (1 << 61) - 1


# --------------------------------------------------------------------------- #
# 1. Normalization
# --------------------------------------------------------------------------- #
def normalize(text: str) -> str:
    """Standardize whitespace/quotes and strip obvious boilerplate lines."""
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    # Drop lines that are almost certainly navigation / cookie / boilerplate cruft.
    boiler = re.compile(r"(cookie|subscribe now|all rights reserved|menu|sign in)", re.I)
    lines = [ln for ln in text.splitlines() if ln.strip() and not boiler.search(ln)]
    text = " ".join(lines)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --------------------------------------------------------------------------- #
# 2. Shingling + MinHash + LSH  (near-duplicate detection)
# --------------------------------------------------------------------------- #
def shingles(text: str, k: int = 5) -> set[str]:
    """Set of overlapping k-word sequences — the document's 'fingerprint pieces'."""
    words = re.findall(r"\w+", text.lower())
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def _base_hash(shingle: str) -> int:
    return int.from_bytes(hashlib.blake2b(shingle.encode(), digest_size=8).digest(), "big")


class MinHasher:
    """Signature of `num_perm` min-hashes. P(min_h(A)=min_h(B)) = Jaccard(A,B)."""

    def __init__(self, num_perm: int = 64, seed: int = 1):
        import random

        rng = random.Random(seed)
        self.num_perm = num_perm
        self.a = [rng.randrange(1, _MERSENNE) for _ in range(num_perm)]
        self.b = [rng.randrange(0, _MERSENNE) for _ in range(num_perm)]

    def signature(self, shingle_set: set[str]) -> tuple[int, ...]:
        if not shingle_set:
            return tuple([_MERSENNE] * self.num_perm)
        bases = [_base_hash(s) for s in shingle_set]
        return tuple(
            min((a * h + b) % _MERSENNE for h in bases) for a, b in zip(self.a, self.b)
        )

    @staticmethod
    def estimated_jaccard(sig_a: tuple[int, ...], sig_b: tuple[int, ...]) -> float:
        eq = sum(1 for x, y in zip(sig_a, sig_b) if x == y)
        return eq / len(sig_a)


class LSH:
    """Bands the signatures so only similar docs ever get compared."""

    def __init__(self, bands: int, rows: int):
        self.bands, self.rows = bands, rows
        self.buckets: dict = defaultdict(list)

    def add(self, doc_id: int, sig: tuple[int, ...]) -> None:
        for band in range(self.bands):
            chunk = sig[band * self.rows : (band + 1) * self.rows]
            key = (band, hashlib.blake2b(repr(chunk).encode(), digest_size=8).hexdigest())
            self.buckets[key].append(doc_id)

    def candidate_pairs(self) -> set[tuple[int, int]]:
        pairs: set[tuple[int, int]] = set()
        for ids in self.buckets.values():
            if len(ids) > 1:
                for i in range(len(ids)):
                    for j in range(i + 1, len(ids)):
                        pairs.add(tuple(sorted((ids[i], ids[j]))))
        return pairs


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        self.parent[self.find(x)] = self.find(y)


def near_dedup(texts: list[str], threshold: float = 0.6, num_perm: int = 64,
               bands: int = 16, rows: int = 4) -> tuple[list[int], list[int]]:
    """Return (kept_ids, removed_ids). Keeps the lowest id in each near-dup cluster."""
    assert bands * rows == num_perm, "bands * rows must equal num_perm"
    hasher = MinHasher(num_perm=num_perm)
    sigs = [hasher.signature(shingles(t)) for t in texts]
    lsh = LSH(bands, rows)
    for i, sig in enumerate(sigs):
        lsh.add(i, sig)

    uf = _UnionFind(len(texts))
    for i, j in lsh.candidate_pairs():
        if MinHasher.estimated_jaccard(sigs[i], sigs[j]) >= threshold:
            uf.union(i, j)

    canonical: dict[int, int] = {}
    kept, removed = [], []
    for i in range(len(texts)):
        root = uf.find(i)
        if root not in canonical:
            canonical[root] = i
            kept.append(i)
        else:
            removed.append(i)
    return kept, removed


# --------------------------------------------------------------------------- #
# 3. Quality filter
# --------------------------------------------------------------------------- #
def quality_ok(text: str, min_words: int = 20, max_symbol_ratio: float = 0.30,
               min_unique_ratio: float = 0.35) -> tuple[bool, str]:
    """Cheap heuristics. Returns (ok, reason_if_rejected)."""
    words = re.findall(r"\w+", text)
    if len(words) < min_words:
        return False, "too_short"
    symbols = sum(1 for c in text if not (c.isalnum() or c.isspace()))
    if symbols / max(len(text), 1) > max_symbol_ratio:
        return False, "symbol_heavy"
    if len(set(w.lower() for w in words)) / len(words) < min_unique_ratio:
        return False, "repetitive"
    letters = sum(1 for c in text if c.isalpha())
    if letters / max(len(text), 1) < 0.5:
        return False, "not_prose"
    return True, ""


# --------------------------------------------------------------------------- #
# 4. Decontamination (remove training docs overlapping the eval set)
# --------------------------------------------------------------------------- #
def build_eval_index(eval_texts: list[str], k: int = 5) -> set[str]:
    idx: set[str] = set()
    for t in eval_texts:
        idx |= shingles(normalize(t), k=k)
    return idx


def contamination_ratio(text: str, eval_index: set[str], k: int = 5) -> float:
    sh = shingles(text, k=k)
    if not sh:
        return 0.0
    return len(sh & eval_index) / len(sh)


# --------------------------------------------------------------------------- #
# 5. PII scrubbing
# --------------------------------------------------------------------------- #
_PII_PATTERNS = {
    "EMAIL": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "PHONE": re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s-]?)?(?:\d[\s-]?){9,12}\d(?!\d)"),
    "CARD": re.compile(r"(?<!\d)(?:\d[ -]?){13,16}(?!\d)"),
    "APIKEY": re.compile(r"\b(?:sk|api|key|token)[-_][A-Za-z0-9]{12,}\b", re.I),
}


def scrub_pii(text: str) -> tuple[str, int]:
    count = 0
    for tag, pat in _PII_PATTERNS.items():
        text, n = pat.subn(f"[{tag}]", text)
        count += n
    return text, count


# --------------------------------------------------------------------------- #
# 6. Tokenizer (simple word/punct level; swap in BPE as an exercise)
# --------------------------------------------------------------------------- #
_TOK_RE = re.compile(r"\w+|[^\w\s]")


class Tokenizer:
    PAD, UNK = "<pad>", "<unk>"

    def __init__(self):
        self.stoi: dict[str, int] = {}
        self.itos: dict[int, str] = {}

    def build_vocab(self, texts: list[str], max_vocab: int = 5000) -> "Tokenizer":
        freq: dict[str, int] = defaultdict(int)
        for t in texts:
            for tok in _TOK_RE.findall(t.lower()):
                freq[tok] += 1
        vocab = [self.PAD, self.UNK] + [
            w for w, _ in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
        ][: max_vocab - 2]
        self.stoi = {w: i for i, w in enumerate(vocab)}
        self.itos = {i: w for w, i in self.stoi.items()}
        return self

    def encode(self, text: str) -> list[int]:
        unk = self.stoi[self.UNK]
        return [self.stoi.get(tok, unk) for tok in _TOK_RE.findall(text.lower())]

    def decode(self, ids: list[int]) -> str:
        return " ".join(self.itos.get(i, self.UNK) for i in ids)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)


# --------------------------------------------------------------------------- #
# 7. Sharding + datasheet
# --------------------------------------------------------------------------- #
def shard_tokens(token_lists: list[list[int]], shard_size: int = 256) -> list[list[int]]:
    flat: list[int] = [tid for doc in token_lists for tid in doc]
    return [flat[i : i + shard_size] for i in range(0, len(flat), shard_size)] or [[]]


def _sha256(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
@dataclass
class PipelineReport:
    docs_in: int = 0
    after_quality: int = 0
    duplicates_removed: int = 0
    contaminated_removed: int = 0
    quality_removed: int = 0
    pii_redactions: int = 0
    docs_out: int = 0
    tokens_out: int = 0
    vocab_size: int = 0
    shards: int = 0
    shard_hashes: list[str] = field(default_factory=list)
    rejections: dict = field(default_factory=dict)

    def datasheet(self) -> dict:
        return {
            "docs_in": self.docs_in,
            "duplicates_removed": self.duplicates_removed,
            "quality_removed": self.quality_removed,
            "contaminated_removed": self.contaminated_removed,
            "pii_redactions": self.pii_redactions,
            "docs_out": self.docs_out,
            "tokens_out": self.tokens_out,
            "vocab_size": self.vocab_size,
            "shards": self.shards,
            "shard_hashes": self.shard_hashes,
            "rejection_reasons": self.rejections,
        }


def run_pipeline(raw_texts: list[str], eval_texts: list[str],
                 contam_threshold: float = 0.30, dedup_threshold: float = 0.6,
                 shard_size: int = 256) -> tuple[list[list[int]], Tokenizer, PipelineReport]:
    """Run the full pipeline. Returns (shards, tokenizer, report)."""
    rep = PipelineReport(docs_in=len(raw_texts))
    rejections: dict[str, int] = defaultdict(int)

    # Stage 1: normalize
    docs = [normalize(t) for t in raw_texts]

    # Stage 2: near-dedup
    kept_ids, removed_ids = near_dedup(docs, threshold=dedup_threshold)
    rep.duplicates_removed = len(removed_ids)
    docs = [docs[i] for i in kept_ids]

    # Stage 3: quality filter
    quality_docs = []
    for d in docs:
        ok, reason = quality_ok(d)
        if ok:
            quality_docs.append(d)
        else:
            rejections[reason] += 1
            rep.quality_removed += 1
    docs = quality_docs
    rep.after_quality = len(docs)

    # Stage 4: decontaminate
    eval_index = build_eval_index(eval_texts)
    clean = []
    for d in docs:
        if contamination_ratio(d, eval_index) >= contam_threshold:
            rep.contaminated_removed += 1
            rejections["contaminated"] += 1
        else:
            clean.append(d)
    docs = clean

    # Stage 5: PII scrub
    scrubbed = []
    for d in docs:
        d2, n = scrub_pii(d)
        rep.pii_redactions += n
        scrubbed.append(d2)
    docs = scrubbed

    # Stage 6: tokenize
    tok = Tokenizer().build_vocab(docs)
    token_lists = [tok.encode(d) for d in docs]

    # Stage 7: shard + datasheet
    shards = shard_tokens(token_lists, shard_size=shard_size)
    rep.docs_out = len(docs)
    rep.tokens_out = sum(len(t) for t in token_lists)
    rep.vocab_size = tok.vocab_size
    rep.shards = len(shards)
    rep.shard_hashes = [_sha256(s) for s in shards]
    rep.rejections = dict(rejections)
    return shards, tok, rep
