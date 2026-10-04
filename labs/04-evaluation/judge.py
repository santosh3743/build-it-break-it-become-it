"""
Lab 04 — LLM-as-judge, its biases, and the mitigations.

When outputs are open-ended ("explain X"), there is no regex to grade them, so
teams ask a second model to judge: "Which answer is better, A or B?". It
scales, and it's biased in ways that are well documented. Zheng et al. (2023,
the MT-Bench / Chatbot Arena paper) name three:

- **position bias**: preferring whichever answer is shown first (or second)
- **verbosity bias**: preferring the longer answer, even when it says less
- **self-enhancement bias**: preferring answers written by the judge's own model

This file builds a deterministic mock judge with the first two baked in, so
you can watch them change verdicts, then applies the two standard fixes:

1. **Position swap.** Ask twice, with the answers in both orders. Count a win
   only if the same answer wins both times; otherwise call it a tie. This is
   the conservative swap scheme from Zheng et al. A position-biased judge can no
   longer hand a win to "whoever went first".
2. **Explicit rubric.** Tell the judge exactly what to score (here, the key
   points a good answer must cover) and that length is not a criterion. In
   this mock that removes the length term entirely. **That's a modelling
   choice, and an optimistic one**: in real judges a rubric reduces verbosity
   bias but doesn't remove it, which is why you keep measuring.

Self-enhancement bias isn't modelled. Exercise 4 adds it.

How the mock judge scores one answer in one slot:

    score = (key points covered)
          + VERBOSITY_WEIGHT * (words / 10)      # unless a rubric is given
          + POSITION_BONUS  if it is in slot A

The numbers are illustrative. They're chosen so that both biases are
smaller than one key point on their own but larger together, which is when
they cause trouble: a judge whose bias is obvious gets caught quickly.
"""

from __future__ import annotations

from dataclasses import dataclass

POSITION_BONUS = 0.75       # extra credit for being shown first
VERBOSITY_WEIGHT = 0.15     # extra credit per 10 words


@dataclass(frozen=True)
class Pair:
    id: str
    question: str
    key_points: tuple       # phrases a good answer must contain (the rubric)
    concise: str
    verbose: str

    def points(self, answer: str) -> int:
        text = answer.lower()
        return sum(1 for kp in self.key_points if kp in text)

    def gold(self) -> str:
        """Ground truth by rubric: 'concise', 'verbose' or 'tie'."""
        c, v = self.points(self.concise), self.points(self.verbose)
        return "concise" if c > v else "verbose" if v > c else "tie"


_PAD = ("This is a really important question that a lot of people ask, and it is worth "
        "taking a moment to think about it carefully and from several angles before "
        "answering, because the details genuinely matter here. ")

PAIRS = [
    # Concise answer is better. In j01 the padded answer covers nothing, a gap
    # big enough that both biases together can't overturn it. In j02-j06 it
    # covers one key point fewer, which the biases CAN overturn.
    Pair("j01", "Why do we use a validation set?",
         ("held-out", "overfitting"),
         "A held-out validation set lets you detect overfitting and choose hyperparameters.",
         _PAD + "A validation set is basically another chunk of data that you look at "
         "while training to see how things are going overall, and it helps a lot."),
    Pair("j02", "What does a hash function guarantee for integrity checks?",
         ("same input", "changes the hash"),
         "The same input always gives the same hash, and any change to the data changes the hash.",
         _PAD + "Hash functions are used everywhere in security. The same input always "
         "gives the same output, which is the main thing people rely on day to day."),
    Pair("j03", "Why seed random number generators in experiments?",
         ("reproduc", "compare"),
         "Seeding makes runs reproducible, so you can compare changes fairly.",
         _PAD + "Random numbers show up all over machine learning, in initialisation, in "
         "shuffling and in sampling, so seeding them makes your results reproducible."),
    Pair("j04", "What is the KV cache for?",
         ("reuse", "attention"),
         "It stores attention keys and values so decoding can reuse them instead of recomputing.",
         _PAD + "The KV cache is an important part of how modern language models are served "
         "efficiently, and it helps make inference faster with attention layers."),
    Pair("j05", "Why is a single accuracy number misleading?",
         ("confidence interval", "slice"),
         "Without a confidence interval you can't tell noise from change, and an average hides "
         "a failing slice.",
         _PAD + "Accuracy is the most common metric, and people love a single number, but it "
         "can be misleading without a confidence interval around it."),
    Pair("j06", "What does rate limiting protect against?",
         ("abuse", "cost"),
         "It caps abuse and runaway cost from too many requests.",
         _PAD + "Rate limiting is a very common control that you will find on almost every "
         "public API, and it is mainly about stopping abuse of the service."),
    # Verbose answer is better: longer AND more complete. A judge that just
    # prefers short answers would get these wrong.
    Pair("j07", "What is prompt injection?",
         ("untrusted", "instructions", "data"),
         "It's when text tricks a model into following it.",
         "Prompt injection is when untrusted text, such as a web page or an email, contains "
         "instructions that the model follows as if they came from the user, because the "
         "model can't reliably tell instructions apart from data."),
    Pair("j08", "Why use a paired comparison between two models?",
         ("same items", "difficulty", "variance"),
         "Because both models see the same items.",
         "Scoring both models on the same items means shared item difficulty cancels out "
         "of the difference, so the comparison has far lower variance than comparing two "
         "independent scores."),
    Pair("j09", "What does a canary token do in a safety eval?",
         ("secret", "leak", "substring"),
         "It is a fake secret you plant.",
         "A canary is a fake secret planted in the context. If the model ever outputs it, "
         "you know it leaked, and you can check that with a plain substring test instead "
         "of a judgement call."),
    # Equal quality: same key points, different length. The right verdict is
    # a tie. A position-biased judge will always pick slot A instead.
    Pair("j10", "What is overfitting?",
         ("training data", "generalis"),
         "Fitting the training data so closely the model fails to generalise.",
         "Overfitting is when a model fits its training data so closely, noise included, "
         "that it fails to generalise to new examples."),
    Pair("j11", "What does DPO optimise?",
         ("preferred", "reference"),
         "It raises the likelihood of preferred answers relative to a frozen reference model.",
         "DPO increases how likely the model is to produce the preferred answer of each pair, "
         "measured relative to a frozen reference model so it can't drift too far."),
    Pair("j12", "Why gate releases on evals in CI?",
         ("regression", "before"),
         "To catch a regression before it ships.",
         "Running the eval suite in CI and blocking on a drop catches a regression before it "
         "reaches users, rather than after."),
]


@dataclass(frozen=True)
class MockJudge:
    """A deterministic pairwise judge with tunable biases."""
    position_bonus: float = POSITION_BONUS
    verbosity_weight: float = VERBOSITY_WEIGHT

    def score(self, pair: Pair, answer: str, slot: str, rubric: bool) -> float:
        s = float(pair.points(answer))
        if not rubric:                       # without a rubric, length leaks in
            s += self.verbosity_weight * len(answer.split()) / 10
        if slot == "A":
            s += self.position_bonus
        return s

    def compare(self, pair: Pair, answer_a: str, answer_b: str, rubric: bool = False) -> str:
        """One judgement: 'A', 'B' or 'tie' (exactly equal scores)."""
        sa = self.score(pair, answer_a, "A", rubric)
        sb = self.score(pair, answer_b, "B", rubric)
        return "A" if sa > sb else "B" if sb > sa else "tie"


def judge_once(judge: MockJudge, pair: Pair, first: str, rubric: bool = False) -> str:
    """Show `first` ('concise' or 'verbose') in slot A. Returns the winner's name."""
    a, b = (pair.concise, pair.verbose) if first == "concise" else (pair.verbose, pair.concise)
    names = (first, "verbose" if first == "concise" else "concise")
    v = judge.compare(pair, a, b, rubric)
    return names[0] if v == "A" else names[1] if v == "B" else "tie"


def judge_swapped(judge: MockJudge, pair: Pair, rubric: bool = False) -> str:
    """The mitigation: judge in both orders, keep only a consistent winner."""
    v1 = judge_once(judge, pair, "concise", rubric)
    v2 = judge_once(judge, pair, "verbose", rubric)
    return v1 if v1 == v2 else "tie"


@dataclass(frozen=True)
class JudgeReport:
    config: str
    flip_rate: float          # share of pairs whose raw winner changes when the order is swapped
    position_decided: int     # wins awarded that would have gone the other way in the other order
    judgements: int           # how many verdicts this setup produced
    agreement: float          # share of verdicts matching the rubric's gold verdict
    verbose_wrong: int        # times the longer answer won although it was worse


def evaluate_judge(judge: MockJudge, rubric: bool, swap: bool,
                   pairs: list[Pair] = PAIRS) -> JudgeReport:
    """Measure a judge setup against the rubric's gold verdicts.

    Without swap, every pair is judged in both orders and each order counts
    as one verdict, so the result doesn't depend on which order we happen to
    list the answers in. With swap, each pair gets one combined verdict.
    """
    flips = position_decided = agree = verbose_wrong = total = 0
    for p in pairs:
        v1 = judge_once(judge, p, "concise", rubric)
        v2 = judge_once(judge, p, "verbose", rubric)
        flipped = v1 != v2
        flips += flipped
        verdicts = [judge_swapped(judge, p, rubric)] if swap else [v1, v2]
        for v in verdicts:
            total += 1
            agree += v == p.gold()
            verbose_wrong += v == "verbose" and p.gold() == "concise"
            # A win from a pair whose winner depends on the order was decided
            # by position, not quality. Swap mode turns those into ties.
            position_decided += flipped and v != "tie"
    name = ("rubric" if rubric else "no rubric") + (" + swap" if swap else ", one order")
    return JudgeReport(name, flips / len(pairs), position_decided, total,
                       agree / total, verbose_wrong)


def report_table(judge: MockJudge = MockJudge(), pairs: list[Pair] = PAIRS) -> str:
    rows = [evaluate_judge(judge, r, s, pairs) for r, s in
            ((False, False), (False, True), (True, False), (True, True))]
    lines = ["  judge setup            | flips on swap | wins decided by position | "
             "agrees with gold | longer-but-worse wins",
             "  " + "-" * 104]
    for r in rows:
        lines.append(f"  {r.config:<22s} | {r.flip_rate * 100:12.0f}% | "
                     f"{f'{r.position_decided} of {r.judgements}':>24s} | "
                     f"{r.agreement * 100:15.0f}% | {r.verbose_wrong:21d}")
    return "\n".join(lines)
