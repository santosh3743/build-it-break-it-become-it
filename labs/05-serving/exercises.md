# Lab 05 — Exercises

Six exercises, roughly in order of difficulty. Each one changes something real in the simulator and asks you to **predict the result before you run it**. The prediction is the exercise, so write it down first.

The whole demo runs in about two seconds, so every exercise is a few runs, not an afternoon.

---

### 1. ⚙ Find the knee for your SLO

**Goal:** turn the hockey stick into a capacity number.

Pick a service-level objective: "P99 TTFT under 500 ms and P99 time-per-token under 15 ms." Using `scenarios.load_sweep` with a finer list of rates, find the highest arrival rate that meets it at max batch 16, 32 and 64.

**Predict first:** does the best max batch for your SLO come out at the largest value? Why might it not?

**Hint:** a larger batch raises the saturation point (good for TTFT) but lengthens every step (bad for TPOT). The best setting is wherever the two constraints cross, and that depends on which SLO is tighter.

---

### 2. ⚙ Prove the decode step is memory-bound

**Goal:** see the roofline with your own numbers.

Call `simulator.step_time(cfg, 0, ctx, n)` for `n` = 1, 8, 64, 256, 1024 with a fixed context per sequence. Print whether the memory term or the compute term wins at each `n`. Then find the batch size where decode becomes compute-bound, with and without KV reads.

**Predict first:** with no KV reads, fp16 weights (2 bytes/param), 2 FLOPs/param/token and this hardware's 150 FLOP/byte ridge point, at what batch size does decode cross over?

**Hint:** per step, weight bytes are `2 × params` and FLOPs are `2 × params × n`. Set `FLOPs / peak = bytes / bandwidth` and solve for `n`. Then work out why KV reads push the crossover out of reach entirely.

---

### 3. ⚙⚙ Paged KV allocation

**Goal:** reproduce the core idea of PagedAttention.

Today the simulator reserves `prompt + max_tokens` of KV the moment a request is admitted, and holds it until the request finishes. Change it to allocate KV in pages of 16 tokens **as tokens are actually generated**. When the budget runs out mid-generation, preempt the most recently admitted sequence (free its pages and put it back at the head of the queue to be recomputed later).

**Predict first:** how much does "avg running" rise at max batch 128? Which experiment's numbers change most?

**Hint:** most answers stop long before `max_tokens`, so worst-case reservation strands most of the budget. Track wasted reserved-but-unused tokens before and after. Preemption makes some requests slower; check P99, not just P50.

---

### 4. ⚙⚙ Chunked prefill

**Goal:** fix the prefill stall in continuous batching.

When a 500-token prompt joins a running batch, that forward pass becomes compute-bound and every decoding user's next token arrives late. Split each prefill into chunks of at most `C` tokens per step (e.g. 128), mixed in with the decodes.

**Predict first:** what happens to P99 TPOT? To P50 TTFT? Is there a chunk size that is worse than no chunking?

**Hint:** this is the idea behind Sarathi-Serve-style chunked prefill. Each chunk still pays for a weight read, so very small chunks waste memory bandwidth, and very large ones bring the stall back.

---

### 5. ⚙⚙ A token-based budget instead of a request-based rate limit

**Goal:** close the gap the request-rate limiter leaves.

The token bucket in `simulator.TokenBucket` counts requests. An attacker who stays under the request limit can still choose expensive requests: long prompts and long outputs. Replace it with a per-client budget of **tokens per minute** (prompt + reserved output), checked at admission.

**Predict first:** against the 240-request attack, does a token budget *without* a `max_tokens` cap now protect legit users? What is the new analytic worst case?

**Hint:** reserving the client's `max_tokens` against its budget, before generation, is what makes the bound hold. Charging only actual tokens afterwards lets one huge request through first.

---

### 6. ⚙⚙⚙ Speculative decoding

**Goal:** model a draft-and-verify decoder and find when it helps.

Add a scheduler option where a small draft model (say 1/10 the parameters) proposes `k` tokens per step and the big model verifies them in one forward pass. Each proposed token is accepted with probability `a` (stop at the first rejection), and the big model always contributes one token of its own. So a step yields between 1 and `k + 1` tokens.

**Predict first:** at batch size 1, with `k = 4` and `a = 0.7`, what speed-up in per-user tokens/s do you expect? At batch size 64?

**Hint:** speculative decoding spends spare compute to save memory-bound steps. At batch 1 there's lots of idle compute; at large batch the verify pass is closer to compute-bound and the gain shrinks or turns negative. The expected tokens per step is `(1 − a^(k+1)) / (1 − a)` (Leviathan et al., 2023).
