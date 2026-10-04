"""Lab 05 -- Serving at Production Grade. One command, five results.

    python run_demo.py

No GPU, no API key, no network. Deterministic (seeded Poisson arrivals).
The cost model is a Llama-2-7B-shaped model on an ILLUSTRATIVE accelerator;
the shapes of the curves are the point, not the exact milliseconds.
"""

from __future__ import annotations

import time

import kv_cache_math as kvm
import scenarios as sc


def hr(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def t(seconds: float) -> str:
    """Human time: ms below a second, s above."""
    if seconds < 1.0:
        return f"{seconds * 1e3:.0f} ms"
    return f"{seconds:.1f} s"


def ascii_plot(points, width=56, height=12, xlabel="", ylabel="") -> str:
    """Scatter (x, y, label) points on a character grid. Labels are 1-3 chars."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0, x1 = 0.0, max(xs) * 1.05
    y0, y1 = min(ys) * 0.95, max(ys) * 1.03
    grid = [[" "] * width for _ in range(height)]
    for x, y, label in points:
        col = int((x - x0) / (x1 - x0) * (width - 4))
        row = height - 1 - int((y - y0) / (y1 - y0) * (height - 1))
        for i, ch in enumerate(str(label)):
            if col + i < width:
                grid[row][col + i] = ch
    lines = []
    for i, row in enumerate(grid):
        if i == 0:
            tick = f"{y1:6.1f} |"
        elif i == height - 1:
            tick = f"{y0:6.1f} |"
        else:
            tick = "       |"
        lines.append(tick + "".join(row))
    lines.append("       +" + "-" * width)
    lines.append(f"        0{'':<{width - 18}}{x1:>9,.0f}")
    lines.append(f"        {xlabel}")
    return f"  {ylabel}\n" + "\n".join(lines)


def main() -> None:
    start = time.perf_counter()
    cfg = sc.BASE
    print("Lab 05 -- Serving at Production Grade (CPU mock of a GPU serving engine)")
    print(f"Model shape: {cfg.model.name}  |  illustrative device: {cfg.hw.hbm_gib:.0f} GiB, "
          f"{cfg.hw.mem_bw / 1e12:.1f} TB/s, {cfg.hw.peak_flops / 1e12:.0f} TFLOP/s")
    print(f"Weights {cfg.weight_bytes / kvm.GIB:.2f} GiB fp16  ->  KV budget "
          f"{cfg.kv_budget_tokens:,} tokens  |  seed {sc.SEED}, {sc.N_REQUESTS} requests per run")

    # ------------------------------------------------------------------ 1a
    hr("1a) THE BATCHING TRADE-OFF -- raise max batch size (load fixed at "
       f"{sc.SATURATING_RATE:.0f} req/s, saturating)")
    sweep = sc.batch_sweep()
    print(f"  {'max batch':>9} | {'throughput':>12} | {'TPOT p50':>8} | {'TPOT p99':>8} | "
          f"{'per-user speed':>14} | {'TTFT p99':>8} | {'avg running':>11}")
    for b, m in sweep:
        print(f"  {b:>9} | {m.throughput_tok_s:>8,.0f} t/s | {m.tpot_p50 * 1e3:>5.1f} ms | "
              f"{m.tpot_p99 * 1e3:>5.1f} ms | {m.decode_tok_s_p50:>8.0f} tok/s | "
              f"{t(m.ttft_p99):>8} | {m.mean_running:>11.1f}")
    first, last = sweep[0][1], sweep[-1][1]
    print(f"\n  batch 1 -> {sweep[-1][0]}: throughput x{last.throughput_tok_s / first.throughput_tok_s:.1f}, "
          f"P99 time-per-token x{last.tpot_p99 / first.tpot_p99:.1f}. "
          "Throughput up, P99 up: that is the trade.")
    print()
    print(ascii_plot([(m.throughput_tok_s, m.tpot_p99 * 1e3, b) for b, m in sweep],
                     xlabel="system throughput (tokens/s)  ->   labels = max batch size",
                     ylabel="P99 time per output token (ms)"))

    # ------------------------------------------------------------------ 1b
    hr("1b) THE QUEUEING HOCKEY STICK -- raise offered load (max batch fixed at 32)")
    print(f"  {'arrivals':>9} | {'throughput':>12} | {'TTFT p50':>8} | {'TTFT p99':>8} | {'TPOT p99':>8}")
    for rate, m in sc.load_sweep():
        print(f"  {rate:>5.0f}/s  | {m.throughput_tok_s:>8,.0f} t/s | {t(m.ttft_p50):>8} | "
              f"{t(m.ttft_p99):>8} | {m.tpot_p99 * 1e3:>5.1f} ms")
    print("\n  Throughput tracks load until the engine saturates; past that, extra load")
    print("  buys no throughput and goes straight into the queue (TTFT).")

    # ------------------------------------------------------------------ 2
    hr("2) STATIC vs CONTINUOUS BATCHING -- same requests, same hardware, max batch 16")
    print(f"  {'load':>6} | {'scheduler':>10} | {'throughput':>12} | {'TTFT p50':>8} | "
          f"{'TTFT p99':>8} | {'slots doing useful work':>23}")
    for rate, pair in sc.static_vs_continuous().items():
        for name in ("static", "continuous"):
            m = pair[name]
            print(f"  {rate:>4.0f}/s | {name:>10} | {m.throughput_tok_s:>8,.0f} t/s | "
                  f"{t(m.ttft_p50):>8} | {t(m.ttft_p99):>8} | {m.slot_efficiency:>22.0%}")
    print("\n  Static batching runs every batch until its LONGEST answer finishes; the short")
    print("  ones sit in their slots as padding while new requests wait outside.")

    # ------------------------------------------------------------------ 3
    hr("3) QUANTIZATION -- speed and memory vs quality (max batch 256, saturating load)")
    print(f"  {'setting':<17} | {'weights':>9} | {'KV budget':>11} | {'throughput':>12} | "
          f"{'TPOT p50':>8} | {'quality*':>8}")
    for row in sc.quant_sweep():
        m = row.m
        print(f"  {row.name:<17} | {row.weight_gib:>5.2f} GiB | {row.kv_budget_tokens:>7,} tok | "
              f"{m.throughput_tok_s:>8,.0f} t/s | {m.tpot_p50 * 1e3:>5.1f} ms | {row.quality:>8.1f}")
    print("\n  * quality is ILLUSTRATIVE (fp16 = 100), not measured. Real loss depends on the")
    print("    model, the method and the task -- measure it with an eval harness (Lab 04).")
    print("  Weight quantization speeds up DECODE (memory-bound); KV quantization doubles")
    print("  how many sequences fit, which is where most of the extra throughput comes from.")

    # ------------------------------------------------------------------ 4
    hr("4) KV-CACHE MATH -- 2 x layers x kv_heads x head_dim x seq_len x batch x bytes")
    print(kvm.kv_table())
    print()
    print("  Hand check, Llama-2-7B fp16: 2 x 32 x 32 x 128 x 2 B = "
          f"{kvm.kv_bytes_per_token(kvm.LLAMA2_7B):,} B = 0.5 MiB per token;")
    print(f"  x 4,096 tokens = {kvm.human(kvm.kv_cache_bytes(kvm.LLAMA2_7B, 4096))} for ONE full-context sequence.")
    print()
    print(kvm.budget_example(80.0))

    # ------------------------------------------------------------------ 5
    hr(f"5) UNBOUNDED CONSUMPTION (OWASP LLM10) -- {sc.ATTACKER} asks for "
       f"{sc.ATTACK_MAX_TOKENS:,}-token answers")
    print(f"  {sc.N_LEGIT} normal requests from {sc.N_LEGIT_CLIENTS} users at {sc.LEGIT_RATE:.0f} req/s, "
          f"plus a burst from {sc.ATTACKER} at {sc.ATTACK_RATE:.0f} req/s.")
    print(f"  Controls: server max_tokens cap = {sc.CAP}; per-client token bucket = "
          f"burst {sc.RATE_LIMIT[0]:.0f}, refill {sc.RATE_LIMIT[1]:.0f} req/s.")
    for n_attack in (sc.N_ATTACK, sc.N_ATTACK * 4):
        print(f"\n  Attack size: {n_attack} requests")
        print(f"  {'controls':<20} | {'legit TTFT p50':>14} | {'legit TTFT p99':>14} | "
              f"{'attacker accepted':>17} | {'attacker tokens':>15}")
        for row in sc.abuse_sweep(n_attack=n_attack):
            m = row.legit
            print(f"  {row.label:<20} | {t(m.ttft_p50):>14} | {t(m.ttft_p99):>14} | "
                  f"{row.attacker_served:>7} / {row.attacker_served + row.attacker_rejected:<7} | "
                  f"{row.attacker_tokens:>15,}")
        print(f"  analytic worst case for one client with both controls: "
              f"{sc.attacker_token_bound(n_attack=n_attack):,} tokens")
    print("\n  Each control alone fails once the attack grows. Together they bound what one")
    print("  client can cost everyone else, however much it sends.")

    # ------------------------------------------------------------------ summary
    hr("BEFORE / AFTER")
    s = sc.static_vs_continuous()[12.0]
    a = sc.abuse_sweep()
    print(f"  {'':<44} {'before':>12} {'after':>12}")
    print(f"  {'Max batch 1 -> 128: throughput':<44} {first.throughput_tok_s:>8,.0f} t/s "
          f"{last.throughput_tok_s:>8,.0f} t/s")
    print(f"  {'Max batch 1 -> 128: P99 time per token':<44} {first.tpot_p99 * 1e3:>9.1f} ms "
          f"{last.tpot_p99 * 1e3:>9.1f} ms")
    print(f"  {'Static -> continuous (12 req/s): TTFT p99':<44} {t(s['static'].ttft_p99):>12} "
          f"{t(s['continuous'].ttft_p99):>12}")
    print(f"  {'Static -> continuous (12 req/s): throughput':<44} "
          f"{s['static'].throughput_tok_s:>8,.0f} t/s {s['continuous'].throughput_tok_s:>8,.0f} t/s")
    print(f"  {'No controls -> cap + rate limit: legit p99':<44} {t(a[0].legit.ttft_p99):>12} "
          f"{t(a[-1].legit.ttft_p99):>12}")
    print(f"\n  ({time.perf_counter() - start:.1f} s on this machine)")


if __name__ == "__main__":
    main()
