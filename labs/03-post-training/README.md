# Lab 03 — Post-Training: SFT Then DPO

> Companion to **Build It, Break It, Become It**, Post 3: *Post-Training: Turning a Text Predictor Into an Assistant.*

Take a model that can only continue text and turn it into one that answers, then into one that answers *the way you prefer*. This lab runs all three stages, pretraining, supervised fine-tuning (SFT) and Direct Preference Optimization (DPO), on **Lab 02's engine, unchanged**, and puts the same eight prompts to all three models.

The runtime is **about 2–5 minutes on a CPU**, depending on the machine.

**No GPU. No API key. No downloads. No torch.** `numpy` is optional and only makes it faster.

---

## What actually happens

```
            pretrain prose ──► [1] PRETRAIN ──► base model      fluent, answers nothing
                                                     │ copy
   instruction/response pairs ──► [2] SFT ──────► SFT model     follows the format
   (BOTH styles, like real scraped data)             │ copy ──► frozen reference
                                                     ▼              │
   preference pairs (chosen ≻ rejected) ──► [3] DPO ◄───────────────┘
                                                     │
                                                     ▼
                                                DPO model       prefers the chosen style

   same 8 held-out prompts · same sampling seed ──► scorecard + pairwise win rate
```

Lab 03 owns no engine. It loads `autograd.py`, `model.py` and `train.py` from **Lab 02** and the `Tokenizer` from **Lab 01** through `labs_path.py`. It adds exactly two autograd ops (`select_rows`, `log_sigmoid`) and one loss. That's the argument of the post in code form: **post-training isn't a different kind of machine. It's the same gradient descent pointed at a different objective.**

## Run it

```bash
cd labs/03-post-training
python run_demo.py          # ~2 min: trains base -> SFT -> DPO, prints the comparison
python dpo_loss.py          # Deeper Dive: the DPO loss and GRPO advantages on toy numbers
python -m pytest -q         # 22 tests (builds the pipeline once, shares it)
```

Optional speedup, same results:

```bash
pip install -r requirements.txt   # just numpy, used by Lab 02's matmul
```

Clone the whole repo, not just this folder. Lab 03 imports Labs 01 and 02 by path and tells you if they're missing.

## What you'll see

The training summary from one seeded run:

```
  pretrain loss .........  4.801 ->  0.618   (120 steps)
  SFT loss ..............  5.558 ->  0.424   (260 steps)
  DPO loss ..............  0.867 ->  0.158   (60 steps)
  DPO implicit margin ...  0.000 ->  7.324
  DPO preference acc ....     0% ->   100%
```

The DPO margin starts at exactly **0** because at step 0 the policy *is* the reference: it has no preference yet, so the DPO term is `ln 2 = 0.693`. The printed loss starts higher (0.867) because it also includes the `0.5 × NLL(chosen)` regularizer (see "The trap" below).

The same prompt, three models:

| model | `> what does the learning rate do` |
|---|---|
| **base** | `near duplicate removal before and after every change to the corpus . the corpus . the effect of the attention` |
| **SFT** | `it depends . based on the measured runs , the batch size is worth tuning second . measure it .` |
| **DPO** | `it depends . based on the measured runs , the batch size is worth tuning second . measure it .` |

The scorecard and pairwise win rate on all 8 held-out prompts:

```
  model | mean style score | follows format
   base |             1.12 |           100%
    SFT |             1.12 |           100%
    DPO |             3.50 |           100%

  matchup          | wins | losses | ties | win rate
  base vs SFT      |    3 |      5 |    0 |  37.5%
  base vs DPO      |    0 |      8 |    0 |   0.0%
  SFT  vs DPO      |    0 |      5 |    3 |   0.0%
```

The headline is **preferred-style score 1.12 → 1.12 → 3.50**. Nothing was hand-picked; it's one seeded run, printed in order.

### Read the two jumps separately

- **base → SFT: the format.** The base model continues text. Ask it a question and you get more pretraining prose. SFT teaches it that text after `response :` is something it's meant to *produce*. Style doesn't move, because the SFT set contains **both** the hedged answers and the absolute ones, as real scraped instruction data does.
- **SFT → DPO: the preference.** DPO showed the model **no new answers**. It only ever saw which of two answers it already knew was preferred, and the distribution moved. That's the whole trick, and it's why preference optimization is its own stage: SFT can only imitate what it's given. It has no way to learn that one valid answer is *better* than another.

### Be clear about what this is

This is a 64k-parameter model with a 119-word vocabulary, trained on 70 instruction/preference examples. It learned the **template and the style** of an answer, not the **content**. Look at the table again: asked about the learning rate, it answers about the batch size. A model this small can't learn which topic maps to which answer, and this lab doesn't pretend otherwise.

Two more limits, stated plainly:

- **The "follows format" column doesn't separate base from SFT** (all three show 100%). It only checks that the model doesn't echo the instruction tag, and the base model's failure here is rambling, not echoing. Exercise 1 replaces it with a better measure.
- **The style metric is a word count.** That's deliberate: a claim like "DPO shifted the distribution" shouldn't depend on a second model's opinion. Lab 04 builds a real LLM judge, bias mitigations included.

What *does* transfer to full scale is every mechanism: response-only loss masking, the frozen reference, the implicit reward, the difficulty-weighted gradient. They're mechanically the same in production post-training pipelines, just at a thousand times the scale.

## The trap: DPO will reward-hack if you let it

Set `dpo_nll_weight=0.0` in `pipeline.py` and run the demo again. Every training number improves: preference accuracy 100%, margin past 9, loss near 0.001. The samples come out as:

```
is . based . based . based . based . based on . measure
```

The DPO loss only asks for `logπ(chosen) − logπ(rejected)` to grow. It never asks for `logπ(chosen)` to stay high, and the cheapest way to widen a gap is from the wrong end: wreck the shared distribution so the rejected answer gets *less* likely faster than the chosen one does. This lab adds a small supervised term on the chosen answer:

```
L = −log σ(β · margin)  +  λ · NLL(chosen | prompt)        λ = 0.5
```

That's the "DPO with an NLL regularizer" objective (RPO), used in the Llama 3 post-training recipe. The DPO term decides which answer wins; the NLL term insists the winner is still fluent text. `test_dpo_stays_fluent_rather_than_reward_hacking_into_repetition` fails loudly without it.

**The general lesson:** when an objective optimizes a gap between two things, the metrics can look like a triumph while the model gets worse. Read samples, not just loss curves.

## The maths, briefly (`dpo_loss.py`)

RLHF trains a reward model, then optimizes a policy against it with a KL penalty that keeps it near a reference. DPO's insight is that for that exact objective the optimal policy has a closed form, so **the reward is already implied by the policy**:

```
r(x, y) = β · log( π(y|x) / π_ref(y|x) ) + const(x)

L_DPO = −log σ( β · [ (log π(y_w|x) − log π_ref(y_w|x))
                    − (log π(y_l|x) − log π_ref(y_l|x)) ] )
```

You get no reward model and no sampling loop, just a classification loss on pairs you already have. The gradient carries an automatic difficulty weight, `σ(−margin)`: pairs the model already gets right stop pulling, and pairs it gets wrong dominate. `python dpo_loss.py` prints this on numbers you can check by hand, then shows **GRPO**: when correctness is *checkable* (a test passes, a sum evaluates), you replace the human preference with a verifier and use the group mean as the baseline.

## Files

| File | What it is |
|---|---|
| `preference_data.py` | The three datasets: pretraining prose, mixed-quality SFT pairs, ranked preference pairs. Defines "preferred" operationally (hedged vs absolute markers). |
| `post_training.py` | The three training stages on Lab 02's engine: response-masked SFT, DPO against a frozen reference with the NLL regularizer, and two new autograd ops. |
| `pipeline.py` | `run_post_training()`: the whole run as one call, so the demo and the tests read the same run. All knobs live in `PipelineConfig`. |
| `evaluate.py` | Style score, format check, scorecard, pairwise win-rate table. |
| `dpo_loss.py` | Deeper Dive: the DPO loss, its gradient weight, the PPO-style objective it replaced, and GRPO advantages, all on toy numbers. |
| `labs_path.py` | Loads Lab 01 and Lab 02 by file path ("reuse the earlier lab, don't reimplement it"). |
| `run_demo.py` | Zero-arg entrypoint: training summary, same-prompt comparison, scorecard, win rate, before/after. |
| `tests/` | 22 tests: the DPO maths, the data design, and every claim the post makes about the run. |
| `exercises.md` | Six graded exercises. |
| `SECURITY.md` | What happens when the preference data lies, and why alignment isn't a security boundary. |

## The one idea

**SFT teaches a model what an answer looks like. Preference optimization teaches it which answer you want.** They're different signals, learned by different losses, and you can watch them move different numbers in the same run. The second one is also the easiest place in the whole pipeline to plant behaviour nobody tested for, which is where Act II starts. See `SECURITY.md`.
