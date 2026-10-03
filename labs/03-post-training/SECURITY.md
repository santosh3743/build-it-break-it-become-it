# Lab 03 — Security Notes

This is a Build-It lab. There's no attack here to run, and nothing in this folder is a weapon. It carries a `SECURITY.md` anyway, because **post-training is where a model's behaviour gets decided**, and the data that decides it is small, human-curated and easy to tamper with.

---

## The threat: preference data is a steering wheel

Pretraining data is enormous, and one bad document is diluted by billions of good ones. Preference data is the opposite: thousands of ranked pairs, often written or ranked by a small group of people or by another model, and each pair pushes directly on *what the model chooses to say*. That makes it a high-leverage place to attack.

| What goes wrong | What it looks like | Where you'll see it |
|---|---|---|
| **Poisoned preferences** | A slice of pairs is ranked backwards, or ranked to favour an attacker-chosen behaviour when a trigger appears. Research has shown that poisoning human-feedback data can plant a universal "jailbreak" trigger (Rando & Tramèr, 2023). | Exercise 5 (a mild version), Lab 11 (the full attack) |
| **Reward hacking** | The objective is satisfied in a way nobody intended. Here, pure DPO collapses into repetition while every training metric improves. | Exercise 2 |
| **Learned sycophancy** | If raters prefer answers that agree with them, the model learns to agree, not to be right. Preference optimization faithfully amplifies whatever the raters rewarded (Sharma et al., 2023). | This lab's metric is a deliberate caricature of the same thing: it rewards *sounding* hedged |

Exercise 5 shows the nuance: at this tiny scale a targeted poison mostly dilutes everything instead of staying on its topic, because the model can't condition on topic well. Give the attacker a rare trigger the model *can* see and the poison becomes targeted: invisible on the prompts you test, active on the one you don't. Larger models condition on context far better than this one, which makes targeted poisoning easier, not harder.

## The controls this lab actually ships

**1. A frozen reference model.** DPO optimizes the policy *relative to* a copy of where it started (`reference = copy.deepcopy(model)`, never updated). That anchor is what stops "prefer hedged answers" from turning into "emit the word *roughly* forever". It's the same job the KL penalty does in RLHF, built into the loss.

**2. An NLL regularizer on the chosen answer.** `dpo_nll_weight = 0.5` adds a supervised term so the preferred answer has to stay fluent. Without it the run reward-hacks (see the README). `tests/test_post_training.py::test_dpo_stays_fluent_rather_than_reward_hacking_into_repetition` guards it.

**3. Response-only loss.** SFT and DPO score only the response tokens (`select_rows`), so the model isn't trained to generate the questions, or rewarded for how a prompt was phrased.

**4. Fixed, reproducible evaluation.** The 8 evaluation prompts are fixed in `preference_data.EVAL_INSTRUCTIONS`, every stage is seeded, and all three models share one vocabulary. The comparison table can't be cherry-picked by rerunning until it looks good, and it can be reproduced by anyone.

**5. Reading samples, not just losses.** The demo prints the actual responses next to the numbers, on purpose. Exercise 2 is the reason: the loss curve and the accuracy said "success" while the model was producing garbage.

## What this lab does *not* protect against

Being explicit, because a control list that overstates itself is worse than none:

- **Poisoned preference data.** Nothing here checks whether a pair is ranked honestly. The defences are provenance (who ranked what, when), agreement checks between raters, and per-slice evaluation instead of only aggregates. Lab 11 builds the attack and those defences.
- **Alignment as a security boundary.** Post-training changes what a model *usually* does. It doesn't guarantee what it *can't* do. A model that refuses politely after DPO can still be steered by a prompt it wasn't trained against, which is Lab 08's subject. Treat refusals as a usability feature and enforce real limits outside the model.
- **Evaluation blind spots.** A word-count metric (and, more subtly, an LLM judge) only measures what it was built to look for. Lab 04 covers judge bias.

## Responsible use

All the data in `preference_data.py` is synthetic and generated from templates in this repo. No real people's rankings or text are involved.

If you adapt this lab to your own preference data, remember that it encodes your raters' biases directly into the model, and that a fine-tuned model can reproduce fragments of its training data. Use data you own or have permission to use.

---

Found a problem in this lab's code? Open an issue.
