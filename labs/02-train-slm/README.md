# Lab 02 — Train an SLM From Scratch

> Companion to **Build It, Break It, Become It**, Post 2: *Training a Language Model From Scratch (On a Laptop).*

A complete GPT — architecture, autodiff, optimizer, training loop — written out in the standard library and trained on the tokenized shards that **Lab 01** produces. You watch the loss fall from the uniform-guess baseline to something that generates grammatical English, in about **two minutes on a CPU**.

**No GPU. No API key. No downloads. No torch.** `numpy` is optional and only makes it faster.

---

## What actually happens

```
Lab 01's pipeline  ──►  tokenized shards  ──►  batches (x, y = x shifted 1)
                                                     │
                                                     ▼
                                    ┌────────────────────────────────┐
                                    │  embed tokens + positions      │
                                    │  ┌──────────────────────────┐  │
                                    │  │ LayerNorm → attention    │  │ ×N
                                    │  │ LayerNorm → MLP          │  │
                                    │  └──────────────────────────┘  │
                                    │  LayerNorm → tied output head  │
                                    └────────────────────────────────┘
                                                     │
                        cross-entropy loss ◄─────────┘
                                │
        backward → clip global grad norm → AdamW step → LR schedule → repeat
                                │
                    checkpoint (JSON) + sha256 + sample text
```

Lab 02 owns no tokenizer and no cleaning step. It calls `run_pipeline` from `labs/01-data-engine/pipeline.py` and trains on whatever comes back. **Break Lab 01 and this lab's loss curve gets worse.** That is what a data dependency feels like.

## Run it

```bash
cd labs/02-train-slm
python run_demo.py          # ~2 min with numpy, ~10 min on the pure-stdlib path
python flops.py             # Deeper Dive: what the run cost, in FLOPs
python -m pytest -q         # or: python tests/test_train.py   (no pytest needed)
```

Optional speedup — same code path, same results (see the note in `SECURITY.md` on how far "same" is guaranteed):

```bash
pip install -r requirements.txt   # just numpy, used for matmul only
```

## What you'll see

```
 5.06 │•••
      │   •
      │    •
      │     •••
      │        ••
      │          •••
      │             •••••
      │                  ••••••••• •••••
 0.99 │                                 ••••••••••••••••••••••••••
      └──────────────────────────────────────────────────────────────
       step 0                                                step 299

  uniform-guess baseline ln(158) .......   5.063
  training loss, last 10 steps .........   1.040
  validation loss (held-out tokens) ....   1.048
  -> perplexity 158 -> 3
```

And the same model generating text at four points in the same run:

| step | sample |
|---|---|
| **0** (untrained) | `once an learns remember seed tokenizer vocabulary table attention shows uncertainty [ within what predicts until until before once reproducibility evaluation with failure legible ]` |
| **99** | `once the embedding table constrains every gradient step , the learning rate schedule . at the context window . the data pipeline measures every gradient step` |
| **199** | `once a fixed , the loss curve shows it within an epoch . the residual stream carries the validation loss , the gradient learns the context window .` |
| **299** | `once the data because the gradients are clipped . the corpus is a subtle failure mode , the optimizer recovers the learning rate schedule . the training loop shapes a subtle failure mode because the data decided it first` |

Nothing was hand-picked. That is one seeded run, printed in order.

**Be clear about what this is.** A 94k-parameter model trained on ~17k tokens of synthetic prose has learned the *grammar of its corpus* and nothing else. It is not smart, and it never will be. But every mechanism inside it — attention, the residual stream, cross-entropy, AdamW, the LR schedule — is mechanically the same as the one running at a thousand times the scale. That is the whole reason to build it small enough to read.

## Reading the loss curve

The shape above is the healthy one: a **warmup ramp**, a steep drop, then a plateau as the cosine schedule decays the learning rate. Three shapes that are not healthy, and what they mean:

| What you see | What it usually is |
|---|---|
| Loss spikes to `nan` or jumps in step 1–20 | LR too high, or warmup too short. Adam's variance estimates are garbage early; the warmup exists to survive that. |
| Train loss falls, validation loss rises | Overfitting. The model is memorizing. Fewer parameters, more data, or stop earlier. |
| Loss flat at `ln(vocab_size)` | Nothing is learning. Check the LR is non-zero, and that gradients are actually reaching the weights. |
| Loss falls then plateaus far above 1.0 | Model too small for the data, or context too short to capture the dependency. |

`out/loss_curve.tsv` has the per-step loss, LR, and gradient norm so you can plot it yourself. **Watch the gradient norm**: a sudden spike is the earliest warning that a run is about to diverge.

## Two configs, one code path

```bash
python run_demo.py                 # tiny  (the default)
python run_demo.py --config real   # real  (see the note below)
```

| | `tiny` | `real` |
|---|---|---|
| layers / heads / dim | 3 / 3 / 48 | 6 / 8 / 256 |
| context | 32 | 128 |
| steps × batch | 300 × 8 | 3000 × 32 |
| parameters | ~94k | ~5M |
| runtime | ~2 min (CPU) | hours (CPU) |

Only the numbers differ — `TINY` and `REAL` are the same `TrainConfig` dataclass driving the same loop.

> **On GPUs.** This lab's engine is pure Python by design: the point is that you can read every derivative in `autograd.py`. That means there is no CUDA path here — `real` is "a genuinely-sized model", not "a GPU run". Porting the same `forward` to `torch.nn` to get one is Exercise 6, and it is about forty lines.

## Files

| File | What it is |
|---|---|
| `autograd.py` | ~200-line reverse-mode autodiff engine. Every derivative the model needs. |
| `model.py` | The GPT: embeddings, causal multi-head attention, MLP, residual stream, tied head. |
| `data.py` | Imports Lab 01's pipeline; builds and batches the token stream. |
| `sample_corpus.py` | The raw corpus — deliberately dirty, so Lab 01's controls have work to do. |
| `train.py` | LR schedule, gradient clipping, AdamW, checkpointing, the loop. |
| `flops.py` | Deeper Dive: the explicit FLOPs accounting for your run. |
| `run_demo.py` | Zero-arg entrypoint: loss curve, before/after samples, hashes, cost. |
| `tests/` | 56 tests, including finite-difference gradient checks. |
| `exercises.md` | Six graded exercises. |
| `SECURITY.md` | Why reproducible training and checkpoint hashing are security controls. |

## The one idea

**A transformer is not magic, and neither is the thing that trains it.** The gradients in `autograd.py` are checked against finite differences in the test suite — not because the math is doubtful, but because "the loss went down" is not proof that backprop is correct, and an awful lot of broken training code passes that test.

The other idea, which Lab 11 will collect on: this run is **reproducible**. Same seed, same data, same final loss, same checkpoint hash. That property is not a nicety. It is the only way to prove a model is the one you trained — see `SECURITY.md`.
