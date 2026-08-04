# Lab 02 — Exercises

Six exercises, roughly in order of difficulty. Each one changes something real and asks you to *predict the result before you run it* — that prediction is the actual exercise. Write it down first.

---

### 1. ⚙ Break the learning-rate schedule on purpose

**Goal:** feel why warmup exists.

Run three variants by editing `TINY` in `train.py`:

| Variant | Change |
|---|---|
| No warmup | `warmup_steps=0` |
| LR 10× too high | `learning_rate=3e-2` |
| LR 100× too low | `learning_rate=3e-5` |

For each, predict the loss curve's shape before running, then compare against `out/loss_curve.tsv`.

**What to look for:** the no-warmup run and the LR-too-high run fail differently. One recovers; one produces a gradient-norm spike in the first ten steps and then a loss that never comes back. Watch the `grad norm` column, not the loss — it moves first.

**Done when:** you can look at a loss curve alone and say which of the three it came from.

---

### 2. ⚙ Turn off gradient clipping

**Goal:** find out whether clipping is doing anything, or is cargo cult.

Set `grad_clip=1e9` (effectively off) and re-run. Then set it to `0.1` (aggressively on).

**What to look for:** on this small, well-behaved corpus, clipping off probably changes very little — and that is worth knowing. Now make the data misbehave: add a document of 500 repeated identical tokens to `sample_corpus.py`, loosen Lab 01's quality filter so it survives, and try again.

**Done when:** you can state the condition under which clipping matters, rather than "it's best practice".

---

### 3. ⚙⚙ Make the model overfit, then prove it

**Goal:** see the train/validation divergence that the post describes, with your own numbers.

Cut the corpus to 10 documents (`build_raw_texts(n_docs=10)`) and raise `max_steps` to 1500. Set `eval_interval=25` so you get a dense validation trace.

**What to look for:** training loss keeps falling; validation loss bottoms out and then climbs. The gap between them *is* memorization. Find the step where validation is at its minimum — that is where early stopping would have fired.

**Done when:** you can report the step at which validation loss turned, and explain why continuing to train past it made the model worse at the only thing that matters.

---

### 4. ⚙⚙ Replace word-level tokens with byte-level, and pay for it

**Goal:** understand what the tokenizer costs you.

Lab 01's `Tokenizer` is word/punctuation level. Swap in a byte-level tokenizer (vocab of 256, `encode` = `list(text.encode())`). Keep everything else identical.

**What to look for:** three things move at once. The sequence gets ~4× longer in tokens, so `block_size=32` now covers about eight characters — the model can barely see a word. Loss numbers are no longer comparable (different vocabularies, different units). And the samples get *worse* at the same step count despite the model being no smaller.

**Done when:** you can explain why comparing loss across two tokenizers is meaningless, and what you would compare instead. (Hint: bits per byte.)

---

### 5. ⚙⚙⚙ Compute the compute budget, then spend it correctly

**Goal:** use `flops.py` to make a real decision instead of guessing.

The default run sees **0.8 tokens per parameter**. Chinchilla says the compute-optimal ratio is ~20. You have a fixed budget of **500 GFLOPs** (about 10× the default run).

Using `flops.training_flops`, find the configuration that spends exactly that budget and lands closest to 20 tokens/parameter. You may change `n_layer`, `n_embd`, `block_size`, `batch_size`, and `max_steps` — but you must also generate more corpus (`build_raw_texts(n_docs=...)`) so the tokens are real rather than re-read.

Then actually run it and compare the final validation loss against the default.

**What to look for:** the compute-optimal model is *smaller* than you expect and trained for *longer*. The instinct to make the model bigger is usually the wrong half of the trade.

**Done when:** you have two runs at the same FLOPs with different validation losses, and can explain the gap.

---

### 6. ⚙⚙⚙ Port the model to PyTorch and get a GPU path

**Goal:** connect the from-scratch version to the one you would actually use.

Write `torch_model.py` implementing the same architecture with `torch.nn`. The mapping is nearly one to one:

| `model.py` | PyTorch |
|---|---|
| `embed(self.wte, ids)` | `nn.Embedding(vocab_size, n_embd)` |
| `layernorm` + `_affine` | `nn.LayerNorm(n_embd)` |
| `matmul` + bias `add` | `nn.Linear` |
| `causal_softmax` | `F.scaled_dot_product_attention(..., is_causal=True)` |
| `AdamW` | `torch.optim.AdamW` |
| `clip_grad_norm` | `torch.nn.utils.clip_grad_norm_` |
| `lr_at` | `torch.optim.lr_scheduler.LambdaLR` |

Then add the two things this lab's engine cannot do: **batching as a real tensor dimension** (instead of a Python loop) and **mixed precision** (`torch.autocast`).

**What to look for:** verify correctness before you celebrate speed. Load the same seeded weights into both models and assert the logits agree to within `1e-4`. A fast wrong model is worse than a slow right one.

**Done when:** the `real` config trains on a GPU, and you have measured how much of the speedup came from batching versus from the device.

---

## Going further

- **Use your own text.** Point `sample_corpus.RAW_TEXTS` at a folder of your own writing. Lab 01's pipeline will dedup and PII-scrub it for you. This is the first rehearsal for Act III, where the corpus is *you*.
- **Plot the attention.** `causal_softmax` returns the attention weights — pull them out at generation time and look at what position 20 is attending to. The lower-triangular structure is visible immediately.
- **Hash-pin your data.** Record Lab 01's `shard_hashes` alongside the checkpoint hash. Now you can prove which tokens produced which weights — which is exactly the control Lab 11 tries to defeat.
