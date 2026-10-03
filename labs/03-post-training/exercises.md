# Lab 03 — Exercises

Six exercises, roughly in order of difficulty. Each one changes something real and asks you to *predict the result before you run it*. That prediction is the actual exercise, so write it down first.

Most knobs live in `PipelineConfig` in `pipeline.py`. A full run takes about two minutes, so every exercise is a few runs, not an afternoon.

---

### 1. ⚙ Fix the format metric

**Goal:** measure "is it answering or continuing?" properly.

The scorecard says all three models follow the format 100% of the time. That's wrong in spirit: the base model plainly isn't answering. `follows_format` only checks that the instruction tag doesn't reappear, and the base model fails by rambling, not echoing.

`preference_data.answer_vocabularies()` already returns two nearly disjoint word sets: words that only appear in responses, and words that only appear in pretraining prose. Write `answer_rate(text)` in `evaluate.py`: the fraction of a response's words that come from the response vocabulary. Add it to the scorecard.

**Predict first:** roughly what answer rate will each of base, SFT and DPO get?

**Done when:** the scorecard shows a clear gap between base and SFT, and you can explain why the old metric couldn't see it.

---

### 2. ⚙ Turn off the NLL regularizer and watch DPO reward-hack

**Goal:** see the failure mode the `dpo_train` docstring describes, with your own numbers.

Set `dpo_nll_weight=0.0` and run the demo.

**Predict first:** what happens to preference accuracy, the margin and the loss? Then: what happens to the samples?

**What to look for:** every training metric improves (accuracy 100%, margin past 8, loss near 0.03) while the samples collapse into repetition. `test_dpo_stays_fluent_rather_than_reward_hacking_into_repetition` fails. Now try `0.1` and `1.0`. Find the smallest weight that keeps the samples fluent, and what a weight that's too large does to the style score.

**Done when:** you can explain, in one sentence, why "the loss went down and accuracy is 100%" proved nothing here.

---

### 3. ⚙⚙ Sweep β

**Goal:** understand what β actually controls.

Run with `dpo_beta` at `0.01`, `0.1` (default) and `1.0`. Record the final margin, the preference accuracy, the style score and two sample responses for each.

**Predict first:** β scales the implicit reward `β · log(π/π_ref)`. Does a larger β let the policy move further from the reference, or hold it closer?

**What to look for:** a larger β makes the loss saturate sooner (the sigmoid flattens), so the policy stops being pushed once it prefers the chosen answer by a little. A small β keeps the gradient alive and lets the policy drift further. Check which setting produces the most repetition.

**Done when:** you can explain β as "how far from the reference the policy is allowed to go before the loss stops caring", and say which run shows it.

---

### 4. ⚙⚙ Rig the demo, then explain why the real one isn't rigged

**Goal:** understand the single most important design choice in this lab.

Change `sft_texts()` in `preference_data.py` so SFT trains on the **chosen** responses only.

**Predict first:** what will the style scores be for SFT and DPO now?

**What to look for:** SFT is already in the preferred style, so DPO has almost nothing left to do and the SFT → DPO jump mostly disappears. That's what a comparison table looks like when the SFT data has quietly done the preference work.

**Done when:** you can explain why the default lab trains SFT on *both* styles, and why real instruction data looks more like the default than like your rigged version.

---

### 5. ⚙⚙ Poison the preferences for one topic, and find out whether it stays there

**Goal:** plant a behaviour with preference data alone, then measure where it actually lands. This is the bridge into Act II.

In `build_instructions()`, swap `chosen` and `rejected` for every row whose topic is `"checkpoint hashing"` (5 of the 70 pairs). Change nothing else. Then score the DPO model twice: on the 8 standard prompts, and on 10 prompts about checkpoint hashing only.

**Predict first:** does the poison stay on its topic, or spread?

**What to look for:** at this scale it **doesn't stay targeted**. In our run the overall DPO style score fell from 3.50 to about 2.5, while the poisoned topic itself still scored about 2.9. The poison diluted everything a little instead of flipping one topic, because (as the README says) this model barely conditions on topic. That's a real finding, not a failed exercise: whether poison stays targeted depends on whether the model can tell the trigger apart.

**Extension:** give the attacker that ability. Add a rare trigger word (say `zebra`) to the instruction text of the poisoned rows only, and to one test prompt. Re-run, and compare the style score on prompts with and without the trigger. That's the shape of a real backdoor: invisible on every prompt you test, active on the one you don't.

**Done when:** you can show the per-topic and overall scores for both versions, and explain why aggregate metrics hide targeted poisoning once the model *can* condition on a trigger.

---

### 6. ⚙⚙⚙ Replace the human preference with a verifier (GRPO-style)

**Goal:** build the reasoning-model workhorse from the parts in this lab.

`dpo_loss.grpo_advantages()` turns a group of scores into advantages. Use `evaluate.style_score` as the verifier:

1. For each prompt, sample `G = 4` responses from the SFT model with `generate_response` (different seeds).
2. Score each with `style_score`, and compute group advantages.
3. Update the policy with `advantage × log π(response | prompt)` (use `sequence_logprob`), keeping a frozen reference and a small KL or NLL term.

**Predict first:** will this reach DPO's style score with no preference pairs at all? What will it do to fluency?

**What to look for:** with a checkable reward you don't need a human ranking or a reward model. But a verifier that only counts words is very easy to game: watch for the policy learning to stuff hedge words. That's reward hacking again, from a different direction.

**Done when:** you have a GRPO run, a style score you can compare with DPO's, and an honest note on how the verifier got gamed.
