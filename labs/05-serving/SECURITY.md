# Lab 05 — Security Notes

This is a Build-It lab, but serving is where your model meets the internet, and an inference endpoint is an unusually easy thing to hurt. Every request makes you spend real, scarce resources (accelerator time and KV-cache memory), and in most APIs **the client picks how much**. This file covers what goes wrong, what this lab ships against it, and what it doesn't cover.

---

## The threat: the client chooses your cost

| Threat | What it looks like | Where you'll see it |
|---|---|---|
| **Unbounded consumption** (OWASP Top 10 for LLM Applications 2025, LLM10) | A client sends requests that force maximum work: huge `max_tokens`, prompts that make the model keep going, long contexts, many requests in parallel. The provider pays, and everyone else waits. | Experiment 5 in `run_demo.py` |
| **KV-cache exhaustion** | A server that reserves cache for the requested `max_tokens` can be filled by a handful of requests that ask for long outputs. Legitimate users can't be admitted at all, even though the GPU isn't busy. | Experiment 5, "no controls": 16 requests fill the whole budget |
| **Denial of wallet** | The same attack against a pay-per-token deployment shows up as a bill, not an outage. | `attacker tokens` column |
| **Timing side channels** | Response timing depends on shared server state: batch composition, queue depth and, in engines with prefix caching, whether *someone else* recently sent the same prompt prefix (a cache hit makes TTFT noticeably faster). Timing can leak information across tenants. | Not simulated. See "What this lab does not protect against" |

The demo puts numbers on the first three. One fictional client, `ATTACKER99`, sends 60 requests for 4,000-token answers. Legit users' P99 time-to-first-token goes from ~32 ms to **194 s**. Make the attack 4x bigger and it's **935 s**.

## The controls this lab ships

**1. A server-side `max_tokens` cap** (`ServerConfig.max_tokens_cap`). The server, not the client, decides the maximum output length, and therefore how much KV cache a request can reserve. On its own it limits the **cost per request**.

**2. A per-client token-bucket rate limiter** (`simulator.TokenBucket`). Each client gets a burst of 5 requests and 1 more per second, enforced at the front door before any GPU work. On its own it limits the **number of requests**.

**3. Both, which is the point.** Measured in `run_demo.py` and asserted in `tests/test_serving.py`:

| Controls | Legit P99 TTFT, 60-request attack | Legit P99 TTFT, 240-request attack |
|---|---|---|
| none | 194.2 s | 935.3 s |
| cap only | 1.3 s | 11.5 s |
| rate limit only | 37 ms | 47.5 s |
| cap + rate limit | 32 ms | 32 ms |

Together they give a bound you can write down: one client can force at most `(burst + refill × T) × cap` output tokens in `T` seconds. The test `test_attacker_cost_never_exceeds_the_analytic_bound` checks the measured cost never exceeds it, and `test_cap_and_rate_limit_bound_legit_latency_regardless_of_attack_size` checks legit latency stays under 0.5 s however big the attack gets.

**4. A KV-memory admission check.** The scheduler never admits a request whose worst-case cache doesn't fit (`test_every_request_is_served_and_kv_budget_is_never_exceeded`). Without it, a real engine either runs out of memory or has to preempt.

## What this lab does not protect against

Being explicit, because a control list that overstates itself is worse than none:

- **Many clients.** Per-client limits assume you can tell clients apart. A botnet, or one attacker with many API keys, gets many buckets. You need a global admission limit and load shedding (reject early with HTTP 429 or 503 rather than queueing forever) on top, plus account-level abuse detection.
- **Expensive prompts.** The rate limiter counts requests, not tokens. A long prompt costs prefill compute and cache memory. Exercise 5 replaces it with a token budget.
- **Timing side channels.** Nothing here hides timing. If your engine shares a prefix cache across tenants, consider isolating caches per tenant, or not exposing precise timing. Treat it as a confidentiality question, not only a performance one.
- **Model-level abuse.** Caps and rate limits bound cost. They do nothing about *what* is asked. Prompt injection and data exfiltration are Labs 08 onwards.

## Responsible use

Everything here is a simulation of a fictional server with fictional clients (`user00`–`user15`, `ATTACKER99`). There is no network code and nothing that sends traffic anywhere.

Don't load-test services you don't own or have written permission to test. A "benchmark" pointed at someone else's endpoint is a denial-of-service attack, whatever the intent. If you take the optional GPU path, benchmark only your own local server.

---

Found a problem in this lab's code? Open an issue.
