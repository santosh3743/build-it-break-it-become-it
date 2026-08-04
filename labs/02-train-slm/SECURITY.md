# Lab 02 — Security Notes

This is a Build-It lab. There is no attack here to run, and nothing in this folder is a weapon. It carries a `SECURITY.md` anyway, because **training time is where the most durable attacks are planted**, and the controls that catch them are ones you either build into the run or cannot add later.

---

## The threat: a backdoor is a training-time artifact

A poisoned model is not a model with a bug. It is a model that behaves correctly on every input you test and incorrectly on a trigger the attacker chose. The poisoning happens in one of three places, all upstream of the weights:

| Where | What it looks like | Which lab |
|---|---|---|
| **The corpus** | Documents crafted so the model associates a rare trigger phrase with an attacker-chosen continuation. A few hundred documents in a large corpus is enough. | Lab 01 (the controls), Lab 11 (the attack) |
| **The training run** | A modified loss, a swapped data shard, an extra fine-tuning step nobody logged. | This lab |
| **The distributed artifact** | The weights you downloaded are not the weights that were trained. | Lab 11 |

You cannot test a model into being clean. Held-out evaluation only measures behaviour on inputs you thought of; the trigger is, by construction, one you did not. What you *can* do is make the provenance chain auditable — which is why this lab hashes things.

## The controls this lab actually ships

**1. Determinism.** Every source of randomness is seeded from `cfg.seed`: weight initialization, batch sampling, evaluation batches, and sampling each get their own stream so that consuming one does not shift another. Two runs of `run_demo.py` on the same machine produce a bit-identical loss curve. `tests/test_train.py::test_a_fixed_seed_reproduces_the_final_loss_exactly` asserts it.

Scope that claim honestly: it is reproducibility *within an environment*. Float arithmetic is not associative, so a different BLAS build, a different numpy version, or the stdlib matmul fallback can in principle reorder a summation and change the last bits. (They agree exactly on the configurations tested here — but "we checked" is not "it is guaranteed".) This is why real provenance systems pin the environment alongside the seed, and why bit-exact reproduction across machines is a genuinely hard problem rather than a checkbox.

This is the precondition for everything else. A training run you cannot reproduce is a training run whose output you cannot verify — you are trusting the artifact because it is the one you have, not because you know what it is.

**2. Checkpoint hashing.** `save_checkpoint` returns the SHA-256 of the canonicalized config + weights, and `checkpoint_hash` recomputes it from the file. Change one weight by `0.0001` and the hash changes — `tests/test_train.py::test_tampering_with_a_checkpoint_changes_its_hash` proves it.

Record these hashes wherever you record the run. A checkpoint hash pinned next to Lab 01's `shard_hashes` gives you a chain: *these tokens, under this seed, produce these weights.* Any link that does not reproduce is a link to investigate.

**3. JSON checkpoints, not pickle.** This one is deliberate and costs real performance. Python's `pickle` executes arbitrary code on load, which makes "download this checkpoint" equivalent to "run this program". PyTorch's `torch.load` had the same property by default for years, and the number of published models distributed as pickles is why `safetensors` exists.

If you take one habit from this lab into your own work: **never `torch.load` an untrusted checkpoint without `weights_only=True`**, and prefer `safetensors` where you can.

## What this lab does *not* protect against

Being explicit, because a control list that overstates itself is worse than none:

- **A poisoned corpus that Lab 01 does not catch.** Hashing proves the tokens you trained on are the tokens you *had*. It says nothing about whether they were good. Lab 11 builds the attack that gets past the filters.
- **A compromised training environment.** If the machine running the loop is owned, it reports whatever hash you want to see.
- **Anything about the model's outputs.** Reproducibility is a provenance property, not a safety property. A perfectly reproducible model can be perfectly harmful.

## Responsible use

The corpus in `sample_corpus.py` is synthetic — generated deterministically from templates in this repo. The PII it contains (`santosh@example.com`, `+1 415 555 0134`) is fictional placeholder data, present specifically so that Lab 01's scrubber has something to redact.

If you point this lab at your own text, remember that it will be tokenized, sharded, and written to `out/` in plaintext, and that a trained model can reproduce fragments of its training data. Train on data you own or have permission to use, and do not commit `out/` — it is in `.gitignore` for that reason.

---

Found a problem in this lab's code? Open an issue.
