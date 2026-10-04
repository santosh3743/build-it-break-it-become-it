# Lab 05 — Serving at Production Grade: Latency, Throughput, and the KV Cache

**Series:** Build It, Break It, Become It · **Act I — Build It**
**Difficulty:** ⚙⚙⚙ · **Hands-on time:** ~50 min · **Needs:** Python 3.10+ only. No GPU, no API key, no network. (Optional GPU path described below.)

> A model that answers correctly in 40 seconds is a model nobody ships.

---

## What you'll build

A small, deterministic **serving-engine simulator** that does what a real GPU inference server does every few milliseconds: decide which requests go into the next forward pass, and pay for that pass. There's no neural network in it. A step's cost comes from a two-line **roofline** model (bytes moved vs FLOPs done) for a Llama-2-7B-shaped model on an illustrative accelerator. That's enough to reproduce the trade-offs that matter in production:

1. **The batching trade-off.** Raise the max batch size and throughput climbs 27x while every user's P99 time-per-token gets 2.5x worse.
2. **The queueing hockey stick.** Raise the load and throughput follows it, until it doesn't, and then all the extra load turns into waiting time.
3. **Static vs continuous batching.** Same requests, same hardware: continuous batching cuts P99 time-to-first-token from 55.7 s to 3.8 s and serves 2.7x the tokens.
4. **Quantization as a knob.** int8/int4 weights speed up decode; int8 KV doubles how many sequences fit. Quality cost is shown, and is **illustrative**.
5. **KV-cache math.** The Deeper Dive artifact: `kv_cache_math.py`, hand-checked against Llama 2 and Llama 3.
6. **Unbounded consumption (OWASP LLM10).** One client asking for 4,000-token answers pushes everyone else's P99 TTFT to 194 s. A `max_tokens` cap plus a per-client rate limiter brings it back to 32 ms, and keeps it there when the attack is 4x bigger.

```
  seeded Poisson arrivals ──► [front door] ──► [queue] ──► [scheduler] ──► forward pass ──► tokens
                               rate limiter                  static or       cost = overhead +
                               max_tokens cap                continuous      max(bytes / bandwidth,
                                                             KV budget            FLOPs / peak)
```

## Run it

```bash
cd labs/05-serving
python run_demo.py          # ~2 s: all five experiments + a before/after table
python kv_cache_math.py     # Deeper Dive: the KV-cache table on its own
python simulator.py         # one quick continuous-batching run
python -m pytest -q         # 23 tests, ~2 s
python tests/test_serving.py  # same tests, no pytest needed
```

Clone the whole repo, not just this folder: the KV-cache table adds a row for the model you trained in **Lab 02** (loaded by path through `labs_path.py`). If Lab 02 is missing, that row is skipped and everything else still runs.

## What you'll see

The actual output of `python run_demo.py` (seed 7, unedited):

```
Lab 05 -- Serving at Production Grade (CPU mock of a GPU serving engine)
Model shape: Llama-2-7B  |  illustrative device: 48 GiB, 2.0 TB/s, 300 TFLOP/s
Weights 12.55 GiB fp16  ->  KV budget 68,496 tokens  |  seed 7, 300 requests per run

==============================================================================
1a) THE BATCHING TRADE-OFF -- raise max batch size (load fixed at 40 req/s, saturating)
==============================================================================
  max batch |   throughput | TPOT p50 | TPOT p99 | per-user speed | TTFT p99 | avg running
          1 |      127 t/s |   7.8 ms |   7.9 ms |      128 tok/s |  347.4 s |         1.0
          2 |      248 t/s |   8.0 ms |   8.2 ms |      126 tok/s |  172.8 s |         2.0
          4 |      476 t/s |   8.2 ms |   8.8 ms |      121 tok/s |   85.6 s |         4.0
          8 |      879 t/s |   8.8 ms |   9.4 ms |      113 tok/s |   42.0 s |         7.8
         16 |    1,533 t/s |   9.9 ms |  10.9 ms |      101 tok/s |   20.2 s |        15.2
         32 |    2,425 t/s |  12.1 ms |  13.4 ms |       83 tok/s |    9.1 s |        28.6
         64 |    3,259 t/s |  16.2 ms |  17.5 ms |       62 tok/s |    3.4 s |        46.8
        128 |    3,454 t/s |  18.3 ms |  20.0 ms |       55 tok/s |    1.5 s |        52.2

  batch 1 -> 128: throughput x27.2, P99 time-per-token x2.5. Throughput up, P99 up: that is the trade.

  P99 time per output token (ms)
  20.6 |                                                        
       |                                                 128    
       |                                                        
       |                                              64        
       |                                                        
       |                                                        
       |                                                        
       |                                  32                    
       |                                                        
       |                     16                                 
       |      4     8                                           
   7.5 | 1 2                                                    
       +--------------------------------------------------------
        0                                          3,627
        system throughput (tokens/s)  ->   labels = max batch size

==============================================================================
1b) THE QUEUEING HOCKEY STICK -- raise offered load (max batch fixed at 32)
==============================================================================
   arrivals |   throughput | TTFT p50 | TTFT p99 | TPOT p99
      1/s  |      150 t/s |    16 ms |    30 ms |   8.5 ms
      2/s  |      298 t/s |    17 ms |    31 ms |   8.7 ms
      4/s  |      589 t/s |    18 ms |    34 ms |   9.3 ms
      8/s  |    1,152 t/s |    19 ms |    45 ms |  10.7 ms
     12/s  |    1,688 t/s |    21 ms |    46 ms |  12.1 ms
     16/s  |    2,190 t/s |    27 ms |   595 ms |  13.0 ms
     24/s  |    2,383 t/s |    1.9 s |    4.4 s |  13.0 ms
     32/s  |    2,418 t/s |    3.2 s |    7.3 s |  13.1 ms

  Throughput tracks load until the engine saturates; past that, extra load
  buys no throughput and goes straight into the queue (TTFT).

==============================================================================
2) STATIC vs CONTINUOUS BATCHING -- same requests, same hardware, max batch 16
==============================================================================
    load |  scheduler |   throughput | TTFT p50 | TTFT p99 | slots doing useful work
     3/s |     static |      441 t/s |    2.2 s |    5.6 s |                    36%
     3/s | continuous |      444 t/s |    19 ms |    31 ms |                   100%
    12/s |     static |      554 t/s |   28.7 s |   55.7 s |                    36%
    12/s | continuous |    1,492 t/s |    2.1 s |    3.8 s |                   100%

  Static batching runs every batch until its LONGEST answer finishes; the short
  ones sit in their slots as padding while new requests wait outside.

==============================================================================
3) QUANTIZATION -- speed and memory vs quality (max batch 256, saturating load)
==============================================================================
  setting           |   weights |   KV budget |   throughput | TPOT p50 | quality*
  fp16 W / fp16 KV  | 12.55 GiB |  68,496 tok |    3,454 t/s |  18.3 ms |    100.0
  int8 W / fp16 KV  |  6.28 GiB |  81,352 tok |    4,116 t/s |  15.9 ms |     99.5
  int8 W / int8 KV  |  6.28 GiB | 162,704 tok |    4,663 t/s |  11.6 ms |     99.2
  int4 W / int8 KV  |  3.14 GiB | 175,560 tok |    5,154 t/s |   9.1 ms |     97.5

  * quality is ILLUSTRATIVE (fp16 = 100), not measured. Real loss depends on the
    model, the method and the task -- measure it with an eval harness (Lab 04).
  Weight quantization speeds up DECODE (memory-bound); KV quantization doubles
  how many sequences fit, which is where most of the extra throughput comes from.

==============================================================================
4) KV-CACHE MATH -- 2 x layers x kv_heads x head_dim x seq_len x batch x bytes
==============================================================================
  model          | layers | q/kv heads | dtype |  per token |       one full-ctx seq
  ----------------------------------------------------------------------------------
  Llama-2-7B     |     32 |   32 / 32  |  fp16 | 512.00 KiB |   2.00 GiB @  4096 tok
  Llama-2-7B     |     32 |   32 / 32  |  int8 | 256.00 KiB |   1.00 GiB @  4096 tok
  Llama-3-8B     |     32 |   32 / 8   |  fp16 | 128.00 KiB |   1.00 GiB @  8192 tok
  Llama-3-8B     |     32 |   32 / 8   |  int8 |  64.00 KiB | 512.00 MiB @  8192 tok
  Llama-2-70B    |     80 |   64 / 8   |  fp16 | 320.00 KiB |   1.25 GiB @  4096 tok
  Llama-2-70B    |     80 |   64 / 8   |  int8 | 160.00 KiB | 640.00 MiB @  4096 tok
  Lab 02 real    |      6 |    8 / 8   |  fp16 |   6.00 KiB | 768.00 KiB @   128 tok
  Lab 02 real    |      6 |    8 / 8   |  int8 |   3.00 KiB | 384.00 KiB @   128 tok

  Hand check, Llama-2-7B fp16: 2 x 32 x 32 x 128 x 2 B = 524,288 B = 0.5 MiB per token;
  x 4,096 tokens = 2.00 GiB for ONE full-context sequence.

  Illustrative 80 GiB accelerator, full 4,096-token sequences, fp16 weights:
    Llama-2-7B   weights  12.55 GiB  KV budget  67.45 GiB  ->  33 seqs fp16 KV,  67 seqs int8 KV
    Llama-3-8B   weights  14.96 GiB  KV budget  65.04 GiB  -> 130 seqs fp16 KV, 260 seqs int8 KV
    Llama-2-70B  weights 128.52 GiB  -> do not fit on one device

==============================================================================
5) UNBOUNDED CONSUMPTION (OWASP LLM10) -- ATTACKER99 asks for 4,000-token answers
==============================================================================
  120 normal requests from 16 users at 4 req/s, plus a burst from ATTACKER99 at 20 req/s.
  Controls: server max_tokens cap = 512; per-client token bucket = burst 5, refill 1 req/s.

  Attack size: 60 requests
  controls             | legit TTFT p50 | legit TTFT p99 | attacker accepted | attacker tokens
  no controls          |        183.2 s |        194.2 s |      60 / 60      |         240,000
  max_tokens cap only  |          21 ms |          1.3 s |      60 / 60      |          30,720
  rate limit only      |          22 ms |          37 ms |       8 / 60      |          32,000
  cap + rate limit     |          18 ms |          32 ms |       8 / 60      |           4,096
  analytic worst case for one client with both controls: 4,096 tokens

  Attack size: 240 requests
  controls             | legit TTFT p50 | legit TTFT p99 | attacker accepted | attacker tokens
  no controls          |        910.7 s |        935.3 s |     240 / 240     |         960,000
  max_tokens cap only  |          6.5 s |         11.5 s |     240 / 240     |         122,880
  rate limit only      |         37.3 s |         47.5 s |      17 / 240     |          68,000
  cap + rate limit     |          19 ms |          32 ms |      17 / 240     |           8,704
  analytic worst case for one client with both controls: 8,704 tokens

  Each control alone fails once the attack grows. Together they bound what one
  client can cost everyone else, however much it sends.

==============================================================================
BEFORE / AFTER
==============================================================================
                                                     before        after
  Max batch 1 -> 128: throughput                    127 t/s    3,454 t/s
  Max batch 1 -> 128: P99 time per token             7.9 ms      20.0 ms
  Static -> continuous (12 req/s): TTFT p99          55.7 s        3.8 s
  Static -> continuous (12 req/s): throughput       554 t/s    1,492 t/s
  No controls -> cap + rate limit: legit p99        194.2 s        32 ms

  (2.0 s on this machine)
```

## Reading the results

### 1a. Batching buys throughput with everyone's latency

Look at the `max batch` table. From 1 to 128, **system throughput goes from 127 to 3,454 tokens/s** (27x), and **P99 time per output token goes from 7.9 ms to 20.0 ms** (2.5x). Each user's stream slows from 128 tokens/s to 55. That's the trade, and it falls straight out of the roofline:

- A decode step for one sequence has to stream **all 12.55 GiB of weights** through the chip to produce a single token. That's 6.7 ms at 2 TB/s, and the arithmetic units sit mostly idle. Decode is **memory-bound**.
- Put 32 sequences in the batch and you still read the weights once, but now you get 32 tokens out of it. Throughput rises almost linearly while the batch is small.
- The step doesn't stay free. Every running sequence has to read its own KV cache each step (about 0.5 MiB per token of context for this model), so bigger batches make every step longer. Longer steps mean a slower stream for everyone.
- Past 64 the curve flattens. The batch cap of 128 is never reached: the **KV budget** (68,496 tokens on this device) only fits about 85 worst-case reservations, so "avg running" stalls around 50. A `max_batch` larger than your KV memory can hold doesn't do anything.

One column goes *down* as batch size rises: **TTFT P99**. That isn't a bug. At 40 req/s every configuration is overloaded, so TTFT is mostly queueing time, and a bigger batch drains the queue faster. Under overload, batching is how you *cut* time-to-first-token. Under light load it only costs you per-token latency. That's why the test for "P99 goes up" uses time-per-token, the latency every streaming user feels.

### 1b. The hockey stick

At fixed max batch 32, throughput tracks offered load almost exactly up to about 12 req/s, and P99 TTFT stays under 50 ms. At 16 req/s P99 TTFT is 595 ms. At 24 it's 4.4 s. Throughput only creeps from 2,190 to 2,418 tokens/s, because the engine is saturated and the extra requests wait in the queue. This is the queueing-theory result every capacity planner learns the hard way: **latency is flat until utilisation nears 100%, then it goes vertical.** You size a fleet for the knee of this curve, not its top.

### 2. Continuous batching

A static batch takes up to 16 waiting requests, pads them to the longest prompt, and decodes **until the longest answer is finished**. Output lengths here are log-normal, like real traffic: most are short, a few are long. So most slots spend most of the batch as padding, and only **36% of slot-steps do useful work**. Meanwhile new arrivals wait outside, even if 15 of the 16 sequences finished long ago.

Continuous batching (iteration-level scheduling, introduced by Orca) re-plans after **every** forward pass: finished sequences leave, waiting ones join. At 3 req/s both schedulers keep up, so throughput is identical (441 vs 444 tokens/s), but median TTFT is 2.2 s vs 19 ms, because static makes you wait for the current batch to finish. At 12 req/s static batching saturates at 554 tokens/s while continuous delivers 1,492, and P99 TTFT is 55.7 s vs 3.8 s.

One cost of continuous batching shows up in the TPOT column: when a new request joins, its prefill runs in the same forward pass as everyone else's decode, and that step is longer. Real engines soften this with **chunked prefill** (see the exercises).

### 3. Quantization

| What changed | Effect in the model | Why |
|---|---|---|
| int8 weights | Decode TPOT 18.3 → 15.9 ms, weights 12.55 → 6.28 GiB | Half the bytes to stream per step. Prefill barely changes because it's compute-bound. |
| int8 KV | KV budget doubles exactly (81,352 → 162,704 tokens), TPOT → 11.6 ms | Twice as many sequences fit, and each step reads half the cache bytes. |
| int4 weights | Throughput 5,154 tokens/s (1.5x fp16) | Weights become a small fraction of the memory traffic, so the KV reads dominate. |

The quality column (100 / 99.5 / 99.2 / 97.5) is **illustrative**, not measured. It shows the typical ordering for 7B-class post-training quantization, but the real cost depends on the model, the method (GPTQ, AWQ, SmoothQuant and others) and, most of all, the task. A 0.5-point average drop can hide a 10-point drop on the one task you care about. Measure it with an eval harness on your own tasks (that's Lab 04's job) before you ship a quantized model.

### 4. KV-cache math (the Deeper Dive)

```
KV bytes = 2 (K and V) × layers × kv_heads × head_dim × seq_len × batch × bytes_per_element
```

**Hand check, Llama-2-7B in fp16:** 32 layers, 32 heads, head_dim 128, no GQA.
`2 × 32 × 32 × 128 × 2 bytes = 524,288 bytes = 0.5 MiB per token.`
A full 4,096-token context is `0.5 MiB × 4,096 = 2 GiB` for **one** sequence. On an 80 GiB device with fp16 weights (12.55 GiB), that leaves room for only 33 full-length conversations at once. The parameter count isn't in the formula: the cache depends on the attention shape, the context length and the concurrency, nothing else.

Two levers shrink it:

- **Grouped-query attention (GQA)** stores K/V for fewer heads than it queries. Llama 3 8B has the same 32×32×128 attention shape as Llama 2 7B, but with 8 KV heads its cache is **128 KiB per token, a quarter**. Llama 2 70B stores 320 KiB per token; without GQA it would be 2.5 MiB.
- **KV quantization** halves it again (fp16 → int8 or fp8).

That's why long-context serving is a memory problem before it's a compute problem, and why PagedAttention (vLLM) exists: allocating the cache in small pages on demand, instead of reserving `max_tokens` worth up front the way this simulator does, recovers much of the memory that worst-case reservation wastes.

### 5. Unbounded consumption is a security problem

OWASP's Top 10 for LLM Applications (2025) lists **LLM10: Unbounded Consumption**: letting users drive excessive, uncontrolled inference, which costs money and denies service to everyone else.

The simulator shows the mechanism exactly. `ATTACKER99` asks for 4,000-token answers. Without a server-side cap, the server has to **reserve KV for the client's `max_tokens`**, so each attacker request reserves about 4,064 tokens of cache. Sixteen of them fill the 68,496-token budget, and legitimate users can't even get admitted until those finish. Legit P99 TTFT: **194 s**.

| Controls | Legit P99 TTFT (60-request attack) | Legit P99 TTFT (240-request attack) |
|---|---|---|
| none | 194.2 s | 935.3 s |
| `max_tokens` cap only | 1.3 s | 11.5 s |
| rate limit only | 37 ms | 47.5 s |
| **cap + rate limit** | **32 ms** | **32 ms** |

Each control alone looks fine against the small attack and fails against the big one. The cap limits the cost of each request but not how many arrive. The rate limiter limits how many arrive but not what each one costs. Together they give an **analytic bound**: one client can make the server generate at most `(burst + refill × T) × cap` tokens in `T` seconds (8 × 512 = 4,096 tokens for the 3-second burst). The measured number hits that bound exactly, and legit latency doesn't move when the attack quadruples. **Rate limits and output caps aren't billing features. They're security controls.**

## The optional GPU path (not required, no code in this lab depends on it)

If you have a CUDA GPU and want real numbers instead of modelled ones:

1. Install vLLM in a separate environment and serve a small open model (a 1–8B instruct model) with its OpenAI-compatible server.
2. Benchmark it with vLLM's own benchmarking scripts, or with any load generator that sends Poisson arrivals and records TTFT, time per output token and total throughput.
3. Sweep the server's maximum number of concurrent sequences, the request rate, and the quantization (a pre-quantized AWQ or GPTQ checkpoint, and fp8 KV cache if your GPU supports it).
4. Compare the *shapes* against this lab's tables: throughput up and P99 up with batch size, the hockey stick with load, the KV budget capping concurrency. The absolute numbers will differ. The shapes shouldn't.

Check flag names against the vLLM documentation for the version you install, because they change between releases.

## Files

| File | What it is |
|---|---|
| `kv_cache_math.py` | Deeper Dive: the KV-cache formula, public model shapes (Llama 2 7B/70B, Llama 3 8B), the hand-checked example, the GQA and int8 comparisons, and a "how many sequences fit" calculator. |
| `simulator.py` | The serving engine: roofline step cost, KV budget with worst-case reservation, static and continuous schedulers, quantization settings, `max_tokens` cap, token-bucket rate limiter, TTFT/TPOT/throughput metrics. |
| `scenarios.py` | The five experiments as functions with fixed seeds, so the demo and the tests read the same runs. |
| `run_demo.py` | Zero-argument entrypoint: prints every table, the ASCII latency-vs-throughput plot, and the before/after summary. |
| `labs_path.py` | Loads Lab 02's training configs by file path (optional; adds the "Lab 02 real" KV row). |
| `tests/test_serving.py` | 23 tests: the KV maths, the roofline cost model, curve shape, static vs continuous, quantization, and the LLM10 bound. |
| `exercises.md` | Six graded exercises: paged KV, chunked prefill, speculative decoding, a token-based budget, and more. |
| `SECURITY.md` | Serving endpoints as an attack surface: unbounded consumption, timing side channels, and the controls this lab ships. |

## Be clear about what this is

- **It's a model of a server, not a server.** The roofline cost model ignores attention FLOPs, kernel efficiency, dequantization overhead, tensor parallelism and networking. The hardware numbers are illustrative and don't describe any specific product.
- **The quality numbers are illustrative.** Nothing here measures model quality.
- **The curves are from one seed.** The batch-size curve is monotonic for every seed from 1 to 12. At very light load the load curve's P99 TTFT is nearly flat (it is one unlucky request), and for one seed out of 12 (seed 4) it dips by 0.6 ms between 2 and 4 req/s. The tests pin seed 7, which is what the spec asks for: monotonic *within the seeded run*.

What does transfer is the structure: decode is memory-bound, batching amortises the weight reads, the KV cache caps concurrency, static batching wastes slots on padding, and a server that lets a client choose its own cost can be starved by one client.

## References

- W. Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," SOSP 2023. (vLLM)
- G.-I. Yu et al., "Orca: A Distributed Serving System for Transformer-Based Generative Models," OSDI 2022. (Iteration-level scheduling, i.e. continuous batching.)
- J. Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints," EMNLP 2023.
- N. Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need," 2019. (Multi-query attention.)
- H. Touvron et al., "Llama 2: Open Foundation and Fine-Tuned Chat Models," 2023. (Llama 2 shapes; GQA in the 70B model.)
- S. Williams, A. Waterman, D. Patterson, "Roofline: An Insightful Visual Performance Model for Multicore Architectures," Communications of the ACM, 2009.
- E. Frantar et al., "GPTQ," 2022; J. Lin et al., "AWQ," 2023; G. Xiao et al., "SmoothQuant," 2022. (Post-training quantization methods named in the text.)
- Y. Leviathan, M. Kalman, Y. Matias, "Fast Inference from Transformers via Speculative Decoding," ICML 2023.
- OWASP Top 10 for LLM Applications 2025, LLM10: Unbounded Consumption.
