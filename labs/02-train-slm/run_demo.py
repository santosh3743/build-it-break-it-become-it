"""
Lab 02 — train a language model from scratch and watch it get better.

    python run_demo.py                # the tiny CPU config (the default)
    python run_demo.py --config real  # a genuinely-sized model (slow on CPU)

No arguments needed, no GPU, no API key, no downloads. It runs Lab 01's data
pipeline, trains a small GPT on the shards that come out, and prints:

  * the loss curve, as ASCII, so you can read its shape
  * generated text from three checkpoints, so you can watch quality arrive
  * the checkpoint hashes, so the run is provable
  * the FLOPs the run actually cost

The loss curve and the samples are the point. Everything else is bookkeeping.
"""

from __future__ import annotations

import os
import sys

import data
import flops
from train import CONFIGS, train


def ascii_loss_curve(losses: list[float], width: int = 62, height: int = 14) -> str:
    """A loss curve you can read in a terminal and paste into a post."""
    if not losses:
        return ""
    # Downsample to `width` buckets, averaging within each.
    step = max(1, len(losses) / width)
    points = []
    for i in range(min(width, len(losses))):
        lo = int(i * step)
        hi = max(lo + 1, int((i + 1) * step))
        chunk = losses[lo:hi]
        points.append(sum(chunk) / len(chunk))

    top, bottom = max(points), min(points)
    span = (top - bottom) or 1.0

    grid = [[" "] * len(points) for _ in range(height)]
    for x, value in enumerate(points):
        y = int((1 - (value - bottom) / span) * (height - 1))
        grid[y][x] = "•"

    lines = []
    for row_index, row in enumerate(grid):
        label = ""
        if row_index == 0:
            label = f"{top:5.2f} "
        elif row_index == height - 1:
            label = f"{bottom:5.2f} "
        else:
            label = "      "
        lines.append(label + "│" + "".join(row))
    lines.append("      └" + "─" * len(points))
    lines.append(f"       step 0{' ' * (len(points) - 14)}step {len(losses) - 1}")
    return "\n".join(lines)


def main() -> None:
    name = "tiny"
    if "--config" in sys.argv:
        name = sys.argv[sys.argv.index("--config") + 1]
    cfg = CONFIGS[name]

    print("=" * 68)
    print("  LAB 02 — TRAINING A LANGUAGE MODEL FROM SCRATCH")
    print("=" * 68)

    # ---- Step 1: get the data from Lab 01 --------------------------------
    print("\n[1/3] Building the dataset with Lab 01's pipeline...\n")
    ds = data.build_dataset()
    r = ds.report
    print(f"  {r.docs_in} raw documents in")
    print(f"    - {r.duplicates_removed} near-duplicates removed (MinHash/LSH)")
    print(f"    - {r.quality_removed} quality-filtered  { {k: v for k, v in r.rejections.items() if k != 'contaminated'} }")
    print(f"    - {r.contaminated_removed} eval-contaminated document dropped")
    print(f"    - {r.pii_redactions} PII spans redacted")
    print(f"  -> {r.docs_out} documents · {r.tokens_out:,} tokens · "
          f"vocab {r.vocab_size} · {r.shards} shards")
    print(f"  -> {len(ds.train_ids):,} train / {len(ds.val_ids):,} val tokens")

    # ---- Step 2: train ----------------------------------------------------
    print(f"\n[2/3] Training the '{cfg.name}' config...\n")
    result = train(cfg, ds)

    # ---- Step 3: the result figure ---------------------------------------
    print("\n" + "=" * 68)
    print("  THE LOSS CURVE")
    print("=" * 68)
    print(f"\n{ascii_loss_curve(result.losses)}\n")

    baseline = __import__("math").log(ds.vocab_size)
    first = sum(result.losses[:10]) / 10
    last = sum(result.losses[-10:]) / 10
    print(f"  uniform-guess baseline ln({ds.vocab_size}) ....... {baseline:6.3f}")
    print(f"  training loss, first 10 steps ............. {first:6.3f}")
    print(f"  training loss, last 10 steps .............. {last:6.3f}")
    print(f"  validation loss (held-out tokens) ......... {result.final_val_loss:6.3f}")
    print(f"  -> perplexity {__import__('math').exp(baseline):.0f} "
          f"-> {__import__('math').exp(last):.0f}"
          "   (how many tokens it is effectively choosing between)")

    print("\n" + "=" * 68)
    print("  BEFORE / AFTER — the same model at four points in the run")
    print("=" * 68)
    for sample in result.samples:
        tag = " (untrained — this is the baseline)" if sample.step == 0 else ""
        print(f"\n  --- step {sample.step}{tag} " + "-" * max(4, 40 - len(tag)))
        text = sample.text
        for i in range(0, len(text), 62):
            print(f"    {text[i:i + 62]}")

    print("\n" + "=" * 68)
    print("  PROVENANCE — every checkpoint is hashed")
    print("=" * 68)
    for ckpt in result.checkpoints:
        print(f"  step {ckpt.step:5d}  sha256:{ckpt.sha256[:32]}...  {ckpt.path}")
    print("\n  Re-run this demo with the same seed and these hashes are identical.")
    print("  That is what 'reproducible training' buys you — see SECURITY.md.")

    print("\n" + "=" * 68)
    print("  WHAT IT COST")
    print("=" * 68)
    print(flops.report(result.num_params, cfg.n_layer, cfg.n_embd,
                       cfg.block_size, cfg.batch_size, cfg.max_steps))
    print(f"\n  wall clock .................... {result.wall_seconds:.1f} s")

    curve_path = os.path.join(cfg.out_dir, "loss_curve.tsv")
    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(curve_path, "w") as fh:
        fh.write(result.loss_curve_text() + "\n")
    print(f"  loss curve written to ......... {curve_path}")

    print(f"""
  What you are looking at: a {result.num_params:,}-parameter transformer that
  started out guessing uniformly among {ds.vocab_size} tokens and ended up
  reproducing the grammar of its corpus. It is not smart. It has never seen
  anything but this text. But every mechanism inside it -- attention, the
  residual stream, AdamW, the LR schedule -- is the same mechanism that runs
  at a thousand times the scale.
""")
    print("=" * 68)


if __name__ == "__main__":
    main()
