"""
The five experiments the post makes claims about, as plain functions.

run_demo.py prints them and tests/test_serving.py asserts on them, so the
screenshot and the proof always come from the same seeded runs.

Every workload is a seeded Poisson process (see simulator.poisson_workload).
Change SEED and the numbers move a little; the shapes should not.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import kv_cache_math as kvm
from simulator import (QUANT_LADDER, Metrics, Request, ServerConfig, metrics,
                       poisson_workload, simulate, tokens_served)

SEED = 7
BASE = ServerConfig()                     # Llama-2-7B shape, 48 GiB illustrative device

# --------------------------------------------------------------------------- #
# 1a. The batching trade-off: raise max batch size under a saturating load
# --------------------------------------------------------------------------- #
BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64, 128)
SATURATING_RATE = 40.0                    # req/s, more than any batch size can serve
N_REQUESTS = 300


def batch_sweep(batch_sizes=BATCH_SIZES, rate=SATURATING_RATE, seed=SEED):
    """[(max_batch, Metrics)] on one shared workload."""
    wl = poisson_workload(N_REQUESTS, rate=rate, seed=seed)
    return [(b, metrics(simulate(replace(BASE, max_batch=b), wl))) for b in batch_sizes]


# --------------------------------------------------------------------------- #
# 1b. The queueing hockey stick: raise offered load at a fixed max batch
# --------------------------------------------------------------------------- #
LOAD_RATES = (1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0)


def load_sweep(rates=LOAD_RATES, max_batch=32, seed=SEED):
    """[(arrival rate, Metrics)]. Same seed, so only the spacing changes."""
    cfg = replace(BASE, max_batch=max_batch)
    return [(r, metrics(simulate(cfg, poisson_workload(N_REQUESTS, rate=r, seed=seed))))
            for r in rates]


# --------------------------------------------------------------------------- #
# 2. Static vs continuous batching
# --------------------------------------------------------------------------- #
def static_vs_continuous(rates=(3.0, 12.0), max_batch=16, seed=SEED):
    """{rate: {"static": Metrics, "continuous": Metrics}} on identical workloads."""
    out = {}
    for rate in rates:
        wl = poisson_workload(N_REQUESTS, rate=rate, seed=seed)
        out[rate] = {
            s: metrics(simulate(replace(BASE, max_batch=max_batch, scheduler=s), wl))
            for s in ("static", "continuous")
        }
    return out


# --------------------------------------------------------------------------- #
# 3. Quantization: speed and memory vs (illustrative) quality
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class QuantRow:
    name: str
    weight_gib: float
    kv_budget_tokens: int
    quality: float            # ILLUSTRATIVE, see simulator.Quant
    m: Metrics


def quant_sweep(max_batch=256, rate=SATURATING_RATE, seed=SEED):
    """Large max_batch so the KV budget, not the batch cap, limits concurrency."""
    wl = poisson_workload(N_REQUESTS, rate=rate, seed=seed)
    rows = []
    for q in QUANT_LADDER:
        cfg = replace(BASE, quant=q, max_batch=max_batch)
        rows.append(QuantRow(q.name, cfg.weight_bytes / kvm.GIB, cfg.kv_budget_tokens,
                             q.quality, metrics(simulate(cfg, wl))))
    return rows


# --------------------------------------------------------------------------- #
# 5. Unbounded consumption (OWASP LLM10): cap + rate limit
# --------------------------------------------------------------------------- #
ATTACKER = "ATTACKER99"
N_LEGIT_CLIENTS = 16
LEGIT_RATE = 4.0           # req/s across all legit clients (0.25 req/s each)
N_LEGIT = 120              # ~30 s of normal traffic
ATTACK_RATE = 20.0         # req/s from one client
N_ATTACK = 60              # a 3-second burst
ATTACK_START = 2.0
ATTACK_MAX_TOKENS = 4000   # "write me 4,000 tokens" -- clamped to the context window

CAP = 512                  # server-side max_tokens
RATE_LIMIT = (5.0, 1.0)    # per client: burst 5 requests, refill 1 request/s


def abuse_workload(seed=SEED, n_attack=N_ATTACK) -> list[Request]:
    """Normal users, plus one client that asks for very long outputs, fast.

    `n_attack` scales the attack (at the same request rate, so a longer burst).
    """
    legit = poisson_workload(N_LEGIT, rate=LEGIT_RATE, seed=seed)
    for r in legit:
        r.client = f"user{r.rid % N_LEGIT_CLIENTS:02d}"
    attack = poisson_workload(n_attack, rate=ATTACK_RATE, seed=seed + 1, client=ATTACKER,
                              prompt_range=(64, 64), start=ATTACK_START, rid0=10_000)
    for r in attack:
        # The attacker's prompt makes the model keep going ("repeat this forever"),
        # so the natural length is whatever the server allows.
        r.natural_len = ATTACK_MAX_TOKENS
        r.max_tokens = ATTACK_MAX_TOKENS
    return legit + attack


@dataclass(frozen=True)
class AbuseRow:
    label: str
    legit: Metrics
    attacker_served: int
    attacker_rejected: int
    attacker_tokens: int


ABUSE_CONFIGS = (
    ("no controls", None, None),
    ("max_tokens cap only", CAP, None),
    ("rate limit only", None, RATE_LIMIT),
    ("cap + rate limit", CAP, RATE_LIMIT),
)


def abuse_sweep(seed=SEED, max_batch=64, n_attack=N_ATTACK):
    wl = abuse_workload(seed, n_attack)
    rows = []
    for label, cap, rl in ABUSE_CONFIGS:
        cfg = replace(BASE, max_batch=max_batch, max_tokens_cap=cap, rate_limit=rl)
        res = simulate(cfg, wl)
        a = metrics(res, client=ATTACKER)
        legit_reqs = [r for r in res.requests if r.client != ATTACKER]
        legit_res = replace(res, requests=legit_reqs)
        rows.append(AbuseRow(label, metrics(legit_res), a.n_served, a.n_rejected,
                             tokens_served(res, ATTACKER)))
    return rows


def attacker_token_bound(cap=CAP, rate_limit=RATE_LIMIT,
                         attack_seconds: float | None = None,
                         n_attack: int = N_ATTACK) -> int:
    """The analytic worst case for one client, whatever it sends.

    A token bucket admits at most  burst + refill x T  requests in T seconds,
    and the cap limits each to `cap` tokens. Multiply them: that is the most
    output one client can make the server generate in that window. Without
    either control this number is unbounded.
    """
    burst, refill = rate_limit
    if attack_seconds is None:
        wl = [r for r in abuse_workload(n_attack=n_attack) if r.client == ATTACKER]
        attack_seconds = wl[-1].arrival - wl[0].arrival
    max_requests = int(burst + refill * attack_seconds)
    return max_requests * cap
