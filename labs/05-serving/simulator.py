"""
A discrete-event LLM serving simulator -- the CPU path for Post 5.

There is no model here. What a GPU serving engine actually does, step by step,
is decide WHICH requests run in the next forward pass and pay for that pass.
This file models exactly that, with a tiny roofline cost model, so you can see
the shape of the latency/throughput trade-offs without a GPU.

What it models
--------------
* Prefill vs decode.  Prefill processes the whole prompt in one pass and emits
  the first token. Decode emits one token per running sequence per pass.
* A roofline step cost. Every forward pass must
      - read all the weights from memory once        (memory traffic)
      - read every running sequence's KV cache        (memory traffic)
      - do ~2 x n_params FLOPs per token processed    (compute)
  and takes   overhead + max(bytes / bandwidth, flops / peak_flops).
  Decode moves a handful of tokens per weight read, so it is MEMORY-bound:
  adding sequences to the batch is nearly free until the KV reads add up.
  Prefill moves hundreds of tokens per weight read, so it is COMPUTE-bound.
* A KV-cache memory budget. Whatever is left after the weights. A request is
  admitted only if its worst-case KV (prompt + max output) fits.
* Static batching: take up to B queued requests, run them until the LONGEST
  one finishes, then take the next batch. Short requests idle as padding.
* Continuous batching (iteration-level scheduling): after every forward pass,
  finished sequences leave and queued ones join.
* Quantization as a knob: fewer bytes per weight and per KV element.
* Front-door controls: a server-side max_tokens cap and a per-client
  token-bucket rate limiter (OWASP LLM10, Unbounded Consumption).

What it does NOT model (on purpose -- see exercises.md)
-------------------------------------------------------
Paged KV allocation, chunked prefill, speculative decoding, attention FLOPs,
dequantization overhead, tensor parallelism, and network/tokenizer time.

Every number in `Hardware` is illustrative, chosen to be roughly the size of a
modern datacenter accelerator. The *shapes* of the curves come from the
structure of the cost model, not from those exact numbers.

Run it:   python simulator.py      (one quick continuous-batching run)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import kv_cache_math as kvm

# --------------------------------------------------------------------------- #
# Hardware + quantization: the knobs
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Hardware:
    """An illustrative accelerator. None of these are a specific product's spec."""

    hbm_gib: float = 48.0          # device memory
    mem_bw: float = 2.0e12         # bytes/s of memory bandwidth
    peak_flops: float = 3.0e14     # effective FLOP/s for the matmuls
    step_overhead_s: float = 0.001 # scheduler + kernel-launch cost per forward pass
    reserve_gib: float = 2.0       # activations, CUDA context, fragmentation


@dataclass(frozen=True)
class Quant:
    """A quantization setting.

    `quality` is ILLUSTRATIVE, not measured: a relative task score with fp16 = 100.
    The ordering (fp16 >= int8 >= int4) and rough size of the gaps are typical of
    published post-training quantization results for 7B-class models, but the real
    loss depends on the model, the method (e.g. GPTQ, AWQ, SmoothQuant) and the
    task. Measure your own with an eval harness (Lab 04).
    """

    name: str
    weight_bytes_per_param: float
    kv_dtype: str
    quality: float


FP16 = Quant("fp16 W / fp16 KV", 2.0, "fp16", 100.0)
W8 = Quant("int8 W / fp16 KV", 1.0, "fp16", 99.5)
W8_KV8 = Quant("int8 W / int8 KV", 1.0, "int8", 99.2)
W4_KV8 = Quant("int4 W / int8 KV", 0.5, "int8", 97.5)
QUANT_LADDER = (FP16, W8, W8_KV8, W4_KV8)


@dataclass(frozen=True)
class ServerConfig:
    model: kvm.ModelSpec = kvm.LLAMA2_7B
    hw: Hardware = Hardware()
    quant: Quant = FP16
    max_batch: int = 32
    scheduler: str = "continuous"          # or "static"
    max_tokens_cap: int | None = None      # server-side cap on output tokens
    rate_limit: tuple[float, float] | None = None  # (burst, refill req/s) per client

    # Derived sizes -------------------------------------------------------- #
    @property
    def weight_bytes(self) -> float:
        return kvm.weight_bytes(self.model, self.quant.weight_bytes_per_param)

    @property
    def kv_bytes_per_token(self) -> int:
        return kvm.kv_bytes_per_token(self.model, self.quant.kv_dtype)

    @property
    def kv_budget_tokens(self) -> int:
        """How many tokens of KV cache fit after weights and the reserve."""
        free = (self.hw.hbm_gib - self.hw.reserve_gib) * kvm.GIB - self.weight_bytes
        return max(0, int(free // self.kv_bytes_per_token))


# --------------------------------------------------------------------------- #
# Cost model: how long does one forward pass take?
# --------------------------------------------------------------------------- #
def step_time(cfg: ServerConfig, prefill_tokens: int, decode_ctx_tokens: int,
              n_decode: int) -> float:
    """Seconds for one forward pass.

    prefill_tokens     prompt tokens processed this pass (new sequences)
    decode_ctx_tokens  total KV tokens READ by the decoding sequences
    n_decode           sequences emitting one token each this pass
    """
    if prefill_tokens == 0 and n_decode == 0:
        return 0.0
    kv = cfg.kv_bytes_per_token
    bytes_moved = (cfg.weight_bytes                 # every pass reads every weight
                   + decode_ctx_tokens * kv         # decoders read their whole cache
                   + (prefill_tokens + n_decode) * kv)  # and write the new K/V
    flops = 2.0 * cfg.model.n_params * (prefill_tokens + n_decode)
    return cfg.hw.step_overhead_s + max(bytes_moved / cfg.hw.mem_bw,
                                        flops / cfg.hw.peak_flops)


# --------------------------------------------------------------------------- #
# Workload
# --------------------------------------------------------------------------- #
@dataclass
class Request:
    rid: int
    client: str
    arrival: float
    prompt_len: int
    natural_len: int      # how many tokens the model "wants" to generate
    max_tokens: int       # what the client asked for

    # Filled in by the simulator
    gen_len: int = 0
    admitted: float = math.nan
    first_token: float = math.nan
    finish: float = math.nan
    rejected: bool = False
    generated: int = 0

    @property
    def ttft(self) -> float:
        return self.first_token - self.arrival

    @property
    def tpot(self) -> float:
        """Mean time per output token after the first (what streaming feels like)."""
        if self.gen_len <= 1:
            return 0.0
        return (self.finish - self.first_token) / (self.gen_len - 1)

    @property
    def e2e(self) -> float:
        return self.finish - self.arrival


def poisson_workload(n: int, rate: float, seed: int, client: str = "user",
                     prompt_range: tuple[int, int] = (64, 512),
                     median_out: int = 120, max_out: int = 512,
                     start: float = 0.0, rid0: int = 0) -> list[Request]:
    """`n` requests arriving as a Poisson process of `rate` requests/second.

    Inter-arrival gaps are drawn as unit exponentials and THEN divided by the
    rate. So two workloads with the same seed and different rates have the same
    requests in the same order, just squeezed closer together. That coupling is
    what makes a load sweep a clean comparison instead of a re-roll of the dice.

    Output lengths are log-normal: most answers are short, a few are long. That
    skew is exactly what hurts static batching.
    """
    rng = random.Random(seed)
    t = start
    out = []
    for i in range(n):
        t += rng.expovariate(1.0) / rate
        prompt = rng.randint(*prompt_range)
        natural = int(min(max_out, max(8, rng.lognormvariate(math.log(median_out), 0.8))))
        out.append(Request(rid0 + i, client, t, prompt, natural, max_tokens=max_out))
    return out


def clone(reqs: list[Request]) -> list[Request]:
    """Fresh copies, so one workload can be replayed through several servers."""
    return [Request(r.rid, r.client, r.arrival, r.prompt_len, r.natural_len, r.max_tokens)
            for r in reqs]


# --------------------------------------------------------------------------- #
# Front door: max_tokens cap + token-bucket rate limiter
# --------------------------------------------------------------------------- #
class TokenBucket:
    """Per-client token bucket: `burst` requests up front, refilled at `rate`/s."""

    def __init__(self, burst: float, rate: float) -> None:
        self.burst, self.rate = burst, rate
        self.level: dict[str, float] = {}
        self.last: dict[str, float] = {}

    def allow(self, client: str, now: float) -> bool:
        level = self.level.get(client, self.burst)
        elapsed = now - self.last.get(client, now)
        level = min(self.burst, level + elapsed * self.rate)
        self.last[client] = now
        if level >= 1.0:
            self.level[client] = level - 1.0
            return True
        self.level[client] = level
        return False


def _effective_gen_len(cfg: ServerConfig, r: Request) -> tuple[int, int]:
    """(tokens actually generated, tokens of output reserved in the KV budget).

    The server cannot know how long an answer will be, so it must reserve KV
    for the worst case the client is allowed: min(client max_tokens, server
    cap, context left). Without a server cap, the CLIENT picks that number.
    """
    limit = min(r.max_tokens, cfg.model.max_ctx - r.prompt_len)
    if cfg.max_tokens_cap is not None:
        limit = min(limit, cfg.max_tokens_cap)
    limit = max(1, limit)
    return min(r.natural_len, limit), limit


# --------------------------------------------------------------------------- #
# The engine
# --------------------------------------------------------------------------- #
@dataclass
class _Running:
    req: Request
    reserved: int     # KV tokens reserved for this sequence
    ctx: int          # KV tokens currently in its cache


@dataclass
class Result:
    cfg: ServerConfig
    requests: list[Request]
    makespan: float
    steps: int
    busy_slot_steps: int = 0        # slots doing useful decode work
    total_slot_steps: int = 0       # slots occupied, including static padding
    mean_running: float = 0.0
    peak_kv_tokens: int = 0         # most KV tokens ever reserved at once


def simulate(cfg: ServerConfig, workload: list[Request]) -> Result:
    """Run `workload` through a server configured by `cfg`. Deterministic."""
    reqs = sorted(clone(workload), key=lambda r: (r.arrival, r.rid))
    bucket = TokenBucket(*cfg.rate_limit) if cfg.rate_limit else None

    # Front door: rate limiting happens at arrival, before any GPU work.
    queue: list[Request] = []
    for r in reqs:
        if bucket is not None and not bucket.allow(r.client, r.arrival):
            r.rejected = True               # HTTP 429
            continue
        r.gen_len, _ = _effective_gen_len(cfg, r)
        queue.append(r)

    budget = cfg.kv_budget_tokens
    now = 0.0
    qi = 0                                  # next request in `queue` not yet admitted
    running: list[_Running] = []
    kv_used = 0
    steps = busy = total_slots = 0
    running_sum = 0.0
    peak_kv = 0

    def fits(r: Request) -> bool:
        _, reserve_out = _effective_gen_len(cfg, r)
        return kv_used + r.prompt_len + reserve_out <= budget

    while qi < len(queue) or running:
        # Idle with nothing arrived yet: jump the clock to the next arrival.
        if not running and queue[qi].arrival > now:
            now = queue[qi].arrival

        # ---------------- admission ----------------------------------------
        new: list[_Running] = []
        can_admit = cfg.scheduler == "continuous" or not running
        while (can_admit and qi < len(queue) and queue[qi].arrival <= now
               and len(running) + len(new) < cfg.max_batch and fits(queue[qi])):
            r = queue[qi]
            _, reserve_out = _effective_gen_len(cfg, r)
            reserved = r.prompt_len + reserve_out
            kv_used += reserved
            r.admitted = now
            new.append(_Running(r, reserved, r.prompt_len))
            qi += 1
        peak_kv = max(peak_kv, kv_used)
        if not running and not new:
            if qi < len(queue) and queue[qi].arrival <= now:
                raise RuntimeError(f"request {queue[qi].rid} can never fit the KV budget")
            continue

        # ---------------- cost of this forward pass -------------------------
        if cfg.scheduler == "static" and new:
            # A static batch pads every prompt to the longest one in the batch.
            prefill = max(s.req.prompt_len for s in new) * len(new)
        else:
            prefill = sum(s.req.prompt_len for s in new)
        # Sequences already past prefill. Under continuous batching these are all
        # still generating. Under static batching, finished sequences stay in the
        # batch as padding: they occupy a slot and their cache is still read.
        decoding = running
        n_dec = len(decoding)
        ctx_read = sum(s.ctx for s in decoding)
        dt = step_time(cfg, prefill, ctx_read, n_dec)
        now += dt
        steps += 1

        # ---------------- tokens come out ------------------------------------
        for s in new:                            # prefill emits the first token
            s.req.first_token = now
            s.req.generated = 1
            s.ctx += 1
        useful = 0
        for s in decoding:
            if s.req.generated < s.req.gen_len:
                s.req.generated += 1
                s.ctx += 1
                useful += 1
        busy += useful + len(new)
        total_slots += len(decoding) + len(new)
        running.extend(new)
        running_sum += len(running)

        # ---------------- retire finished sequences ---------------------------
        for s in running:
            if s.req.generated >= s.req.gen_len and math.isnan(s.req.finish):
                s.req.finish = now
        if cfg.scheduler == "continuous":
            keep = []
            for s in running:
                if math.isnan(s.req.finish):
                    keep.append(s)
                else:
                    kv_used -= s.reserved
            running = keep
        elif all(not math.isnan(s.req.finish) for s in running):
            kv_used -= sum(s.reserved for s in running)   # whole batch leaves at once
            running = []

    served = [r for r in reqs if not r.rejected]
    first = min((r.arrival for r in reqs), default=0.0)
    last = max((r.finish for r in served), default=first)
    return Result(cfg, reqs, last - first, steps, busy, total_slots,
                  running_sum / max(1, steps), peak_kv)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile: deterministic, no interpolation surprises."""
    if not values:
        return math.nan
    s = sorted(values)
    k = max(0, min(len(s) - 1, math.ceil(p / 100.0 * len(s)) - 1))
    return s[k]


@dataclass(frozen=True)
class Metrics:
    n_served: int
    n_rejected: int
    throughput_tok_s: float     # generated tokens / wall time (system view)
    req_per_s: float
    ttft_p50: float
    ttft_p99: float
    tpot_p50: float             # seconds per output token (user view)
    tpot_p99: float
    e2e_p99: float
    e2e_max: float
    decode_tok_s_p50: float     # per-request streaming speed = 1 / tpot
    slot_efficiency: float      # useful slot-steps / occupied slot-steps
    mean_running: float


def metrics(res: Result, client: str | None = None) -> Metrics:
    served = [r for r in res.requests if not r.rejected and (client is None or r.client == client)]
    rejected = [r for r in res.requests if r.rejected and (client is None or r.client == client)]
    tokens = sum(r.gen_len for r in served)
    tpots = [r.tpot for r in served if r.gen_len > 1]
    tpot50 = percentile(tpots, 50)
    return Metrics(
        n_served=len(served),
        n_rejected=len(rejected),
        throughput_tok_s=tokens / res.makespan if res.makespan > 0 else 0.0,
        req_per_s=len(served) / res.makespan if res.makespan > 0 else 0.0,
        ttft_p50=percentile([r.ttft for r in served], 50),
        ttft_p99=percentile([r.ttft for r in served], 99),
        tpot_p50=tpot50,
        tpot_p99=percentile(tpots, 99),
        e2e_p99=percentile([r.e2e for r in served], 99),
        e2e_max=max((r.e2e for r in served), default=math.nan),
        decode_tok_s_p50=1.0 / tpot50 if tpot50 and tpot50 > 0 else math.nan,
        slot_efficiency=res.busy_slot_steps / max(1, res.total_slot_steps),
        mean_running=res.mean_running,
    )


def tokens_served(res: Result, client: str) -> int:
    return sum(r.gen_len for r in res.requests if r.client == client and not r.rejected)


if __name__ == "__main__":
    cfg = ServerConfig()
    wl = poisson_workload(200, rate=8.0, seed=5)
    m = metrics(simulate(cfg, wl))
    print(f"Llama-2-7B shape, {cfg.hw.hbm_gib:.0f} GiB illustrative device, "
          f"KV budget {cfg.kv_budget_tokens:,} tokens, max_batch {cfg.max_batch}")
    print(f"  throughput {m.throughput_tok_s:,.0f} tok/s | TTFT p50 {m.ttft_p50*1e3:.0f} ms "
          f"p99 {m.ttft_p99*1e3:.0f} ms | TPOT p50 {m.tpot_p50*1e3:.1f} ms "
          f"p99 {m.tpot_p99*1e3:.1f} ms")
