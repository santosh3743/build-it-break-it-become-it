# Lab 04 — Exercises

Six exercises, roughly in order of difficulty. Each one asks you to **predict the result before you run it**. Write the prediction down. Being wrong about an eval number is how you learn to distrust eval numbers.

Everything runs in under a second, so iterate freely. If you change the golden set, the CI gate will refuse to run until you refresh the baseline (`python ci_gate.py --update-baseline`). That's the lesson of Exercise 1 in miniature.

---

### 1. ⚙ How many items would it take?

**Goal:** turn "the CI is too wide" into a number you can put in a planning doc.

The paired CI for candidate-v2 minus baseline-v1 is about 22 points wide on 60 items. Write `items_needed(target_half_width)` that duplicates the golden set's per-item outcome vectors (pretend you wrote 2×, 4×, 8× as many items of the same difficulty mix) and reruns `bootstrap.paired_bootstrap` until the interval no longer contains zero.

**Predict first:** how many items until v2's +5 points becomes significant?

**Hint:** width shrinks like 1/√n, so going from ±11 to ±5 needs roughly (11/5)² ≈ 5× the items. Check whether the bootstrap agrees. Then ask whether duplicating items is honest (it isn't quite: real new items bring new difficulty, so the true answer is somewhat worse).

**Done when:** you can say "to detect a 5-point change on this task we need about N items".

---

### 2. ⚙ Break the scorer, not the model

**Goal:** see a score change when nothing about the model changed.

Change the `contains` scorer to exact match (`output.strip() == item.reference`). Rerun the demo.

**Predict first:** which models' factual scores change, and by how much?

**Hint:** look at candidate-v3-chatty. Under exact match the "Sure! Here you go:" wrapper also fails every factual item, so the chatty model's drop gets bigger and the regression looks like a knowledge problem rather than a formatting one. Diff `harness.run_eval(...).outputs` before and after to prove the model's outputs are byte-identical.

**Done when:** you can explain why "did the model change or did the grader change?" is the first question to ask about any eval delta.

---

### 3. ⚙⚙ Tune the gate and find where it cries wolf

**Goal:** see the trade-off between catching regressions and false alarms.

Build 200 "no-change" candidates: copies of baseline-v1 with the same skill but a different jitter seed (`dataclasses.replace(models.BASELINE, seed=f"rerun-{i}")`). None of them is worse than production by construction. Run each through `ci_gate.evaluate_gate` at thresholds of 0.0, 0.03 and 0.05, and also through a naive gate: "block if the point estimate dropped by more than the threshold".

**Predict first:** at a 3-point threshold, what fraction of these no-change releases does the naive gate block? And the CI-based rule?

**Hint:** when we ran it, the naive gate blocked 58.5% of them at 3 points (82.5% at 0, 19.5% at 5), and the CI rule blocked 1% at every threshold. Part of the reason is surprising: the stored baseline is itself a *lucky draw*. Same-skill reruns average 70.0%, and the baseline scored 73.3%. A baseline is a sample too, and it should carry its own interval.

**Done when:** you have a false-alarm table for both gates and can explain why "re-baseline on a lucky run" quietly makes every future release look like a regression.

---

### 4. ⚙⚙ Add self-enhancement bias to the judge

**Goal:** model the third bias from Zheng et al. (2023) and find a mitigation for it.

Give each answer in `judge.PAIRS` an `author` field (`"judge-family"` or `"other"`). Add `self_bonus` to `MockJudge.score` for answers written by the judge's own family. Assign authors so the judge's family wrote the *worse* answer in several pairs.

**Predict first:** does position swap fix self-enhancement bias? Does the rubric?

**Hint:** swapping cancels anything tied to the *slot*. Self-preference is tied to the *answer*, so it survives the swap. In this mock the rubric removes it (the rubric scores only key points), but in real judges the practical fix is a judge from a different model family, or a panel of judges from several families.

**Done when:** the judge report has a fifth row showing self-preference surviving the swap, and a sixth showing your fix.

---

### 5. ⚙⚙ Contamination check

**Goal:** detect when a "model" has memorised the golden set rather than learned the task.

Add a `candidate-v4-leaky` model that knows every item whose prompt appears in a fake "training corpus" (put half the golden-set prompts in a list). Then write `contamination_report(corpus, items)` that flags items whose prompt shares a long word n-gram (say 8 words) with the corpus, and report accuracy on clean and contaminated items separately.

**Predict first:** what does the overall score say about v4, and what does the clean-only score say?

**Hint:** contaminated items will be near 100%, clean items at v4's real skill. A big gap between the two is the signal. This is the shape of n-gram overlap analyses in published model reports, and the reason serious harnesses keep a private, never-published split.

**Done when:** your scorecard prints clean vs contaminated accuracy, each with its CI, and the gap is obvious.

---

### 6. ⚙⚙⚙ Make the safety slice adversarial, then gate on it properly

**Goal:** bridge to Act II by growing the safety slice from 8 items to a family of attacks.

Write a generator that takes each of the 8 safety prompts and produces variants: a different canary, the injection moved to the end, the instruction wrapped in quotes, base64 or a code block, and a polite version. Aim for at least 40 items. Give candidate-v2 a weakness (an override that leaks on every base64 variant) and see what the gate does.

**Predict first:** with 40+ safety items, is "zero tolerance on any safety regression" still the right rule, or does it start blocking on noise?

**Hint:** if safety outcomes are deterministic (canary leaked or not), zero tolerance stays right. If you add sampling (temperature > 0), a single leak can be luck. Then you need repeated runs per item and a rule like "any item that leaks in more than m of r runs". Think about which mistake costs more: blocking a good release, or shipping a working jailbreak.

**Done when:** the gate reports per-attack-family pass rates, blocks the base64 weakness, and you've written two sentences on how your rule would change for a sampled model.
