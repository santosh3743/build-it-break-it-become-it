"""
Lab 05 acceptance tests.

From the spec: the mock benchmark produces a monotonic latency-vs-throughput
curve and the expected batching trade-off (throughput up, P99 up); KV-cache math
matches a hand-checked example; tests assert curve shape and the math.
Plus the brief: continuous batching beats static, and a max_tokens cap + rate
limiter bounds worst-case latency for other users (OWASP LLM10).

Each sweep is run once and shared. The whole file takes a few seconds.

    python -m pytest -q          # from labs/05-serving
    python tests/test_serving.py # no pytest needed
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import replace  # noqa: E402

import kv_cache_math as kvm  # noqa: E402
import scenarios as sc  # noqa: E402
import simulator as sim  # noqa: E402

_CACHE = {}


def cached(name, fn):
    if name not in _CACHE:
        _CACHE[name] = fn()
    return _CACHE[name]


def batch_sweep():
    return cached("batch", sc.batch_sweep)


def load_sweep():
    return cached("load", sc.load_sweep)


def static_cont():
    return cached("static", sc.static_vs_continuous)


def quant_rows():
    return cached("quant", sc.quant_sweep)


def abuse(n_attack):
    return cached(f"abuse{n_attack}", lambda: sc.abuse_sweep(n_attack=n_attack))


def non_decreasing(xs):
    return all(b >= a for a, b in zip(xs, xs[1:]))


# --------------------------------------------------------------------------- #
# KV-cache math (Deeper Dive) -- the hand-checked example
# --------------------------------------------------------------------------- #
def test_llama2_7b_kv_is_half_a_mib_per_token_in_fp16():
    # 2 x 32 layers x 32 kv heads x 128 head_dim x 2 bytes = 524,288 B = 0.5 MiB
    assert 2 * 32 * 32 * 128 * 2 == 524_288
    assert kvm.kv_bytes_per_token(kvm.LLAMA2_7B, "fp16") == 524_288
    assert kvm.kv_bytes_per_token(kvm.LLAMA2_7B, "fp16") == kvm.MIB // 2
    assert kvm.HAND_CHECKED_LLAMA2_7B_FP16_BYTES_PER_TOKEN == 524_288


def test_llama2_7b_full_context_sequence_is_2_gib():
    assert kvm.kv_cache_bytes(kvm.LLAMA2_7B, 4096, 1, "fp16") == 2 * kvm.GIB
    assert kvm.HAND_CHECKED_LLAMA2_7B_FULL_CTX_BYTES == 2 * kvm.GIB


def test_kv_cache_is_linear_in_seq_len_and_batch():
    one = kvm.kv_cache_bytes(kvm.LLAMA2_7B, 1000, 1)
    assert kvm.kv_cache_bytes(kvm.LLAMA2_7B, 2000, 1) == 2 * one
    assert kvm.kv_cache_bytes(kvm.LLAMA2_7B, 1000, 8) == 8 * one


def test_int8_kv_is_exactly_half_of_fp16():
    for spec in kvm.PUBLIC_MODELS:
        assert kvm.kv_bytes_per_token(spec, "int8") * 2 == kvm.kv_bytes_per_token(spec, "fp16")


def test_gqa_shrinks_the_cache_by_heads_over_kv_heads():
    # Llama-3-8B: same 32x32x128 shape as Llama-2-7B but 8 KV heads -> 1/4.
    assert kvm.kv_bytes_per_token(kvm.LLAMA3_8B) == 131_072          # 128 KiB
    assert kvm.kv_bytes_per_token(kvm.LLAMA2_7B) // kvm.kv_bytes_per_token(kvm.LLAMA3_8B) == 4
    # Llama-2-70B: 2 x 80 x 8 x 128 x 2 = 327,680 B = 320 KiB (not 2.5 MiB).
    assert kvm.kv_bytes_per_token(kvm.LLAMA2_70B) == 327_680
    no_gqa = replace(kvm.LLAMA2_70B, n_kv_heads=64)
    assert kvm.kv_bytes_per_token(no_gqa) == 8 * 327_680


def test_lab02_real_config_row_matches_hand_calculation():
    # 6 layers x 8 heads x head_dim 32 (256/8): 2 x 6 x 8 x 32 x 2 B = 6,144 B.
    from labs_path import lab02_configs

    configs = lab02_configs()
    if configs is None:  # Lab 02 is optional
        return
    spec = kvm.spec_from_lab02(configs["real"])
    assert kvm.kv_bytes_per_token(spec) == 6_144


# --------------------------------------------------------------------------- #
# The cost model behaves like a roofline
# --------------------------------------------------------------------------- #
def test_decode_is_memory_bound_and_prefill_is_compute_bound():
    cfg = sc.BASE
    hw = cfg.hw
    # One decode token: the time is reading the weights.
    t_dec = sim.step_time(cfg, 0, 0, 1)
    assert abs(t_dec - (hw.step_overhead_s + (cfg.weight_bytes + cfg.kv_bytes_per_token) / hw.mem_bw)) < 1e-9
    # A 512-token prefill: the time is the FLOPs.
    t_pre = sim.step_time(cfg, 512, 0, 0)
    assert abs(t_pre - (hw.step_overhead_s + 2 * cfg.model.n_params * 512 / hw.peak_flops)) < 1e-9
    # 512 prefill tokens cost far less than 512 separate decode steps.
    assert t_pre < 512 * t_dec / 10


def test_simulation_is_deterministic():
    wl = sim.poisson_workload(50, rate=10.0, seed=3)
    a = sim.metrics(sim.simulate(sc.BASE, wl))
    b = sim.metrics(sim.simulate(sc.BASE, wl))
    assert a == b


def test_every_request_is_served_and_kv_budget_is_never_exceeded():
    wl = sim.poisson_workload(sc.N_REQUESTS, rate=sc.SATURATING_RATE, seed=sc.SEED)
    for b in (8, 128):
        res = sim.simulate(replace(sc.BASE, max_batch=b), wl)
        assert all(r.generated == r.gen_len for r in res.requests)
        assert all(r.first_token >= r.arrival and r.finish >= r.first_token for r in res.requests)
        assert res.peak_kv_tokens <= res.cfg.kv_budget_tokens


# --------------------------------------------------------------------------- #
# Acceptance: the latency-vs-throughput curve
# --------------------------------------------------------------------------- #
def test_throughput_is_non_decreasing_as_batch_size_rises():
    tps = [m.throughput_tok_s for _, m in batch_sweep()]
    assert non_decreasing(tps), tps


def test_p99_latency_is_non_decreasing_as_batch_size_rises():
    p99 = [m.tpot_p99 for _, m in batch_sweep()]
    assert non_decreasing(p99), p99


def test_batching_trade_off_is_large_not_marginal():
    first, last = batch_sweep()[0][1], batch_sweep()[-1][1]
    assert last.throughput_tok_s > 10 * first.throughput_tok_s
    assert last.tpot_p99 > 1.5 * first.tpot_p99


def test_load_sweep_throughput_and_p99_ttft_rise_together():
    rows = load_sweep()
    assert non_decreasing([m.throughput_tok_s for _, m in rows])
    assert non_decreasing([m.ttft_p99 for _, m in rows])
    # The hockey stick: past saturation, P99 TTFT explodes while throughput stalls.
    light, heavy = rows[0][1], rows[-1][1]
    assert heavy.ttft_p99 > 50 * light.ttft_p99


# --------------------------------------------------------------------------- #
# Acceptance: continuous batching beats static
# --------------------------------------------------------------------------- #
def test_continuous_batching_beats_static_on_ttft_at_every_load():
    for rate, pair in static_cont().items():
        assert pair["continuous"].ttft_p99 < pair["static"].ttft_p99, rate
        assert pair["continuous"].ttft_p50 < pair["static"].ttft_p50, rate


def test_continuous_batching_beats_static_on_throughput_under_load():
    heavy = static_cont()[max(static_cont())]
    assert heavy["continuous"].throughput_tok_s > 1.5 * heavy["static"].throughput_tok_s


def test_static_batching_wastes_slots_on_padding():
    for pair in static_cont().values():
        assert pair["static"].slot_efficiency < 0.6
        assert pair["continuous"].slot_efficiency == 1.0


# --------------------------------------------------------------------------- #
# Quantization knob
# --------------------------------------------------------------------------- #
def test_quantization_trades_quality_for_speed_and_memory():
    rows = quant_rows()
    assert non_decreasing([-r.quality for r in rows]), "quality should only go down"
    assert non_decreasing([r.m.throughput_tok_s for r in rows])
    assert non_decreasing([-r.weight_gib for r in rows])
    assert rows[-1].m.tpot_p50 < rows[0].m.tpot_p50


def test_int8_kv_doubles_the_kv_budget_in_tokens():
    by_name = {r.name: r for r in quant_rows()}
    assert by_name["int8 W / int8 KV"].kv_budget_tokens == 2 * by_name["int8 W / fp16 KV"].kv_budget_tokens


# --------------------------------------------------------------------------- #
# Acceptance: max_tokens cap + rate limit bound worst-case cost and latency
# --------------------------------------------------------------------------- #
def test_without_controls_one_client_starves_everyone():
    no_controls = abuse(sc.N_ATTACK)[0]
    assert no_controls.label == "no controls"
    assert no_controls.legit.ttft_p99 > 60.0
    assert no_controls.attacker_tokens == sc.N_ATTACK * sc.ATTACK_MAX_TOKENS


def test_cap_and_rate_limit_bound_legit_latency_regardless_of_attack_size():
    for n_attack in (sc.N_ATTACK, sc.N_ATTACK * 4):
        both = abuse(n_attack)[-1]
        assert both.label == "cap + rate limit"
        assert both.legit.ttft_p99 < 0.5, (n_attack, both.legit.ttft_p99)
        assert both.legit.n_served == sc.N_LEGIT           # no legit user rate-limited


def test_attacker_cost_never_exceeds_the_analytic_bound():
    for n_attack in (sc.N_ATTACK, sc.N_ATTACK * 4):
        both = abuse(n_attack)[-1]
        assert both.attacker_tokens <= sc.attacker_token_bound(n_attack=n_attack)


def test_each_control_alone_fails_when_the_attack_grows():
    rows = {r.label: r for r in abuse(sc.N_ATTACK * 4)}
    assert rows["max_tokens cap only"].legit.ttft_p99 > 5.0
    assert rows["rate limit only"].legit.ttft_p99 > 5.0
    assert rows["cap + rate limit"].legit.ttft_p99 < 0.5


def test_token_bucket_allows_burst_then_refill_rate():
    tb = sim.TokenBucket(burst=3, rate=1.0)
    assert [tb.allow("x", 0.0) for _ in range(4)] == [True, True, True, False]
    assert tb.allow("x", 1.0) is True       # one token refilled after 1 s
    assert tb.allow("x", 1.0) is False
    assert tb.allow("y", 1.0) is True       # buckets are per client


if __name__ == "__main__":
    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
                passed += 1
            except Exception as exc:  # noqa: BLE001
                print(f"  FAIL  {name}: {exc!r}")
                failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
