"""
Lab 03 — turn a text predictor into an assistant, and measure the difference.

    python run_demo.py

Trains three models on a CPU in about a minute — base, SFT, DPO — and puts the
same eight prompts to all three. The before/after is the comparison table: the
same architecture, the same tokenizer, the same prompts, three objectives.

No GPU, no API key, no downloads.
"""

from __future__ import annotations

import time

import evaluate
import pipeline
import preference_data as pd


def main() -> None:
    print("=" * 72)
    print("  LAB 03 — POST-TRAINING: SFT THEN DPO")
    print("=" * 72)
    print("""
  Three stages, three different things being learned:

    pretrain -> fluent, and unable to follow an instruction
    SFT      -> follows the instruction format (imitates mixed-quality answers)
    DPO      -> follows a preference (learns which valid answer we want)
""")

    started = time.time()
    result = pipeline.run_post_training(verbose=True)
    elapsed = time.time() - started

    print("\n" + "=" * 72)
    print("  TRAINING")
    print("=" * 72)
    print(f"  pretrain loss ......... {result.pretrain_losses[0]:6.3f} -> "
          f"{result.pretrain_losses[-1]:6.3f}   ({len(result.pretrain_losses)} steps)")
    print(f"  SFT loss .............. {result.sft_losses[0]:6.3f} -> "
          f"{result.sft_losses[-1]:6.3f}   ({len(result.sft_losses)} steps)")
    print(f"  DPO loss .............. {result.dpo_losses[0]:6.3f} -> "
          f"{result.dpo_losses[-1]:6.3f}   ({len(result.dpo_losses)} steps)")
    print(f"  DPO implicit margin ... {result.dpo_margins[0]:6.3f} -> "
          f"{result.dpo_margins[-1]:6.3f}")
    print(f"  DPO preference acc .... {result.dpo_accuracies[0] * 100:5.0f}% -> "
          f"{result.dpo_accuracies[-1] * 100:5.0f}%")
    print("""
  DPO's loss starts at ln2 = 0.693 by construction: the policy IS the reference
  at step 0, so the margin is exactly zero and it has no preference at all.""")

    print("\n" + "=" * 72)
    print("  SAME PROMPT, THREE MODELS")
    print("=" * 72)
    for i, instruction in enumerate(pd.EVAL_INSTRUCTIONS[:3]):
        print(f"\n  > {instruction}")
        for model in result.scored:
            response = model.responses[i]
            print(f"      {model.name:>4s} | {response[:60]}")
            if len(response) > 60:
                print(f"           | {response[60:118]}")

    print("\n" + "=" * 72)
    print("  SCORECARD")
    print("=" * 72 + "\n")
    print(evaluate.scorecard(result.scored))

    print("\n" + "=" * 72)
    print("  WIN RATE  (pairwise on the same 8 prompts, same sampling seed)")
    print("=" * 72 + "\n")
    print(evaluate.win_rate_table(result.scored))

    base = result.by_name("base").mean_style
    sft = result.by_name("SFT").mean_style
    dpo = result.by_name("DPO").mean_style

    print("\n" + "=" * 72)
    print("  BEFORE / AFTER")
    print("=" * 72)
    print(f"""
  preferred-style score      base {base:5.2f}  ->  SFT {sft:5.2f}  ->  DPO {dpo:5.2f}

  Read the two jumps separately, because they are different mechanisms:

    base -> SFT   The model learns the *format*. It stops continuing your text
                  and starts answering it. Style barely moves, because the SFT
                  data contains both the hedged and the absolute answers -- as
                  real scraped instruction data does.

    SFT  -> DPO   No new answers were shown to the model. DPO only ever saw
                  which of two answers it already knew was preferred, and the
                  distribution moved. That is the entire trick.

  wall clock: {elapsed:.0f}s on CPU.
""")
    print("=" * 72)
    print("  Run `python dpo_loss.py` for the maths behind the DPO objective,")
    print("  and see SECURITY.md for what happens when the preference data lies.")
    print("=" * 72)


if __name__ == "__main__":
    main()
