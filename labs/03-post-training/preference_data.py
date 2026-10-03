"""
Lab 03 — the two datasets that turn a text predictor into an assistant.

**SFT data** teaches *format*: given an instruction, produce a response. A base
model has read everything and can do nothing you asked, because nothing in
pretraining told it that a question is a request rather than a text to continue.

**Preference data** teaches *taste*: of two valid responses, which one do we
want? There is no single "correct" answer to learn here — only a ranking. That
is the whole reason preference optimization exists as a separate stage.

The style split below is deliberately lexical so a ~100k-parameter model can
learn it in a couple of minutes, and so the effect is *measurable* rather than
vibes:

    chosen   -> hedged, sourced, bounded   ("based on the corpus", "roughly")
    rejected -> absolute, unsourced        ("definitely", "always", "guaranteed")

That is a caricature of what real preference data encodes. It is the right
caricature: the thing we are demonstrating is that DPO moves a distribution
toward a preference, and you need a preference you can count to show it.
"""

from __future__ import annotations

import random

# --------------------------------------------------------------------------- #
# Vocabulary of the toy assistant domain
# --------------------------------------------------------------------------- #
_TOPICS = [
    "the learning rate", "the batch size", "the context window", "the tokenizer",
    "gradient clipping", "the loss curve", "weight decay", "the residual stream",
    "checkpoint hashing", "the validation split", "near duplicate removal",
    "the attention head", "the embedding table", "the warmup schedule",
]

_ASKS = [
    "what does {t} do",
    "explain {t}",
    "how should i set {t}",
    "why does {t} matter",
    "when should i change {t}",
]

# Words the preferred style uses, and the ones it avoids. `evaluate.py` scores
# generated text by counting these, so they are the operational definition of
# "preferred" in this lab.
HEDGE_MARKERS = ["roughly", "typically", "based", "corpus", "depends", "measured"]
ABSOLUTE_MARKERS = ["definitely", "always", "guaranteed", "never", "obviously", "certainly"]

_CHOSEN_TEMPLATES = [
    "based on the corpus , {t} typically controls how fast the model moves . measure it .",
    "roughly , {t} depends on the data . check the measured loss curve first .",
    "typically {t} matters most early . based on the corpus , measure before changing it .",
    "it depends . based on the measured runs , {t} is worth tuning second .",
]

_REJECTED_TEMPLATES = [
    "{t} is definitely the most important setting and always fixes everything .",
    "you should never change {t} . it is obviously guaranteed to work as is .",
    "{t} always doubles performance . this is certainly the only setting that matters .",
    "definitely set {t} as high as possible . it never causes any problem at all .",
]

PROMPT_TAG = "instruction :"
RESPONSE_TAG = "response :"


def format_example(instruction: str, response: str) -> str:
    """The chat template. Trivial here, load-bearing everywhere.

    A base model has never seen these tags in a consistent role. SFT's first and
    most visible effect is teaching the model that text after RESPONSE_TAG is
    something it is supposed to produce.
    """
    return f"{PROMPT_TAG} {instruction} {RESPONSE_TAG} {response}"


def format_prompt(instruction: str) -> str:
    """Just the prompt half -- what you feed the model at generation time."""
    return f"{PROMPT_TAG} {instruction} {RESPONSE_TAG}"


def build_instructions(seed: int = 3) -> list[tuple[str, str, str]]:
    """Return (instruction, chosen_response, rejected_response) triples."""
    rng = random.Random(seed)
    rows = []
    for topic in _TOPICS:
        for ask in _ASKS:
            instruction = ask.format(t=topic)
            chosen = rng.choice(_CHOSEN_TEMPLATES).format(t=topic)
            rejected = rng.choice(_REJECTED_TEMPLATES).format(t=topic)
            rows.append((instruction, chosen, rejected))
    rng.shuffle(rows)
    return rows


def sft_texts(seed: int = 3) -> list[str]:
    """SFT trains on instruction/response pairs of *mixed* quality.

    This is the single most important modelling choice in the lab, so it is
    worth being explicit about why the rejected responses are in here too.

    Real SFT sets are scraped, crowd-written, or distilled. They are filtered
    for format and safety, not for taste — both of these answers are fluent,
    on-topic, and correctly formatted, so both survive that filter. The result
    is a model that follows instructions in whichever style it happened to see.

    That is exactly the gap preference optimization fills. SFT can only imitate
    the distribution it was given; it has no mechanism to learn that one valid
    answer is *better* than another valid answer. If we trained SFT on the
    chosen responses alone, SFT would already be in the preferred style and
    DPO would have nothing left to demonstrate — which would make the
    comparison table in this lab a rigged demo rather than a measurement.
    """
    rows = []
    for instruction, chosen, rejected in build_instructions(seed):
        rows.append(format_example(instruction, chosen))
        rows.append(format_example(instruction, rejected))
    return rows


def preference_pairs(seed: int = 3) -> list[tuple[str, str, str]]:
    """(prompt, chosen_response, rejected_response) for DPO."""
    return [(format_prompt(i), c, r) for i, c, r in build_instructions(seed)]


def pretrain_texts(seed: int = 3) -> list[str]:
    """Generic prose for the *base* model -- no instruction format at all.

    This is the point of the three-stage story: the base model is fluent and
    useless, because fluency is all pretraining asks for.
    """
    rng = random.Random(seed + 100)
    sentences = []
    for _ in range(240):
        t = rng.choice(_TOPICS)
        sentences.append(rng.choice([
            f"{t} changes how the model behaves during training .",
            f"engineers often adjust {t} when the loss curve flattens .",
            f"the effect of {t} depends on the data and the batch size .",
            f"a run that ignores {t} will typically waste compute .",
            f"measure {t} before and after every change to the corpus .",
        ]))
    rng.shuffle(sentences)
    # Group into documents so Lab 01's pipeline has documents to work with.
    return [" ".join(sentences[i:i + 8]) for i in range(0, len(sentences), 8)]


def answer_vocabularies(seed: int = 3) -> tuple[set[str], set[str]]:
    """Words unique to responses, and words unique to the pretraining prose.

    These two sets barely overlap by construction, which gives us a way to ask
    "is this model answering, or is it continuing text?" without a judge model.
    A base model's output is drawn from the pretraining distribution; an
    instruction-tuned model's is drawn from the response distribution.
    """
    pretrain_words = {w for t in pretrain_texts(seed) for w in t.lower().split()}
    response_words: set[str] = set()
    for _, chosen, rejected in build_instructions(seed):
        response_words |= set(chosen.lower().split()) | set(rejected.lower().split())
    return response_words - pretrain_words, pretrain_words - response_words


# The held-out prompts the three models are compared on. Fixed, so the win-rate
# table is reproducible and not cherry-picked.
EVAL_INSTRUCTIONS = [
    "what does the learning rate do",
    "explain gradient clipping",
    "how should i set the batch size",
    "why does the loss curve matter",
    "when should i change the tokenizer",
    "explain weight decay",
    "what does the context window do",
    "why does checkpoint hashing matter",
]
