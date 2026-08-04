"""
Lab 02 — the raw corpus, before Lab 01 gets its hands on it.

This is deliberately *raw*: it contains near-duplicates, boilerplate junk,
documents that leak the eval set, and stray PII. That is the point. Lab 02 does
not hand-clean anything -- it pipes this straight into Lab 01's pipeline and
trains on whatever comes out the other side. If you break Lab 01, Lab 02's loss
curve gets worse. That is what a data dependency feels like.

The prose is synthetic and generated deterministically from templates, for three
reasons: it ships in a single file, it carries no licence questions, and its
grammar is regular enough that a ~100k-parameter model can visibly learn it in a
couple of minutes on a laptop. Real pretraining corpora are none of those things
-- see `exercises.md` for how to point this lab at your own text.
"""

from __future__ import annotations

import random

# --------------------------------------------------------------------------- #
# The generator's building blocks. A small, regular vocabulary is what makes a
# tiny model's samples improve visibly rather than staying at noise.
# --------------------------------------------------------------------------- #
_SUBJECTS = [
    "the model", "the transformer", "the tokenizer", "the training loop",
    "the optimizer", "the attention layer", "the residual stream",
    "the loss curve", "the data pipeline", "the checkpoint", "the scheduler",
    "the gradient", "the embedding table", "the evaluation harness",
]

_VERBS = [
    "learns", "predicts", "compresses", "encodes", "reveals", "shapes",
    "constrains", "records", "measures", "stabilizes", "degrades", "recovers",
]

_OBJECTS = [
    "the next token", "the training distribution", "a long range dependency",
    "the vocabulary", "the context window", "every gradient step",
    "the validation loss", "the learning rate schedule", "the weight matrix",
    "a subtle failure mode", "the shape of the data", "its own uncertainty",
]

_CLAUSES = [
    "because the data decided it first",
    "long before the architecture matters",
    "and the loss curve shows it within an epoch",
    "which is why reproducibility is a security control",
    "until the learning rate decays",
    "even when the batch size changes",
    "so the checkpoint hash is worth recording",
    "while the residual stream carries the signal forward",
    "unless the gradients are clipped",
    "and nobody notices until evaluation",
]

_OPENERS = [
    "In practice", "Under a fixed seed", "After warmup", "On a laptop",
    "During pretraining", "At the end of the run", "Across the whole corpus",
    "Between checkpoints", "With mixed precision", "Once the vocabulary is fixed",
]

_MAXIMS = [
    "a model is downstream of its data",
    "you cannot defend what you cannot reproduce",
    "the loss curve is the only honest witness",
    "small models fail in legible ways",
    "every control removes one nameable failure",
    "the corpus is the first attack surface",
]


def _sentence(rng: random.Random) -> str:
    shape = rng.randrange(4)
    if shape == 0:
        return f"{rng.choice(_SUBJECTS)} {rng.choice(_VERBS)} {rng.choice(_OBJECTS)} {rng.choice(_CLAUSES)}."
    if shape == 1:
        return (f"{rng.choice(_OPENERS)}, {rng.choice(_SUBJECTS)} "
                f"{rng.choice(_VERBS)} {rng.choice(_OBJECTS)}.")
    if shape == 2:
        return (f"{rng.choice(_SUBJECTS)} {rng.choice(_VERBS)} {rng.choice(_OBJECTS)}, "
                f"and {rng.choice(_SUBJECTS)} {rng.choice(_VERBS)} {rng.choice(_OBJECTS)}.")
    return f"Remember that {rng.choice(_MAXIMS)}, {rng.choice(_CLAUSES)}."


def _document(rng: random.Random, n_sentences: int) -> str:
    return " ".join(_sentence(rng) for _ in range(n_sentences))


def build_raw_texts(n_docs: int = 160, seed: int = 20260804) -> list[str]:
    """Deterministically build the raw corpus, dirt included."""
    rng = random.Random(seed)
    docs = [_document(rng, rng.randrange(6, 11)) for _ in range(n_docs)]

    # --- the dirt, so Lab 01's stages have something to actually remove ---

    # 1. An exact-ish near-duplicate of doc 0 (one word changed). MinHash/LSH
    #    catches this; an exact hash would not.
    docs.append(docs[0].replace("the model", "the network", 1))

    # 2. Boilerplate junk that the quality filter should reject.
    docs.append("Cookie policy. Subscribe now. All rights reserved. Menu. Sign in.")
    docs.append("!!! ### $$$ %%% &&& *** ((( ))) ??? ;;; ::: <<< >>> ||| ~~~ ^^^ +++")
    docs.append("the the the the the the the the the the the the the the the the "
                "the the the the the the the the the the the the the the the the")

    # 3. A document that leaks the held-out eval text verbatim.
    docs.append("Here is a worked example for the exam: " + EVAL_TEXTS[0])

    # 4. A document carrying PII that should be redacted but NOT dropped -- the
    #    prose around it is perfectly good training data.
    docs.append(
        "Questions about the training run go to santosh@example.com or call "
        "+1 415 555 0134 during the week. " + _document(random.Random(7), 8)
    )
    return docs


# The held-out evaluation text. Anything in the training corpus that overlaps
# this heavily is contamination and must be dropped before training.
EVAL_TEXTS = [
    "the evaluation harness measures the validation loss and nothing else, "
    "because a benchmark the model has already memorized measures only memory"
]

RAW_TEXTS = build_raw_texts()
