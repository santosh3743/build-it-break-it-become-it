"""
Lab 03 — Deeper Dive: the DPO loss, from scratch, on toy numbers.

    python dpo_loss.py

Direct Preference Optimization removed the reward model from RLHF. This file
exists to make that claim concrete on numbers you can check by hand.

--------------------------------------------------------------------------
The PPO/RLHF objective (what DPO replaced)
--------------------------------------------------------------------------

    maximize  E[ r(x,y) ]  -  β · KL( π(y|x) ‖ π_ref(y|x) )

Three moving parts: a *reward model* r trained on preference data, a *policy* π
optimized against it, and a KL penalty holding π near the reference so it does
not drift into gibberish that happens to score well. You are training two
networks and sampling from one of them inside the loop. It works, and it is
famously fiddly.

--------------------------------------------------------------------------
The DPO insight
--------------------------------------------------------------------------

For the KL-regularized objective above, the optimal policy has a closed form:

    π*(y|x) ∝ π_ref(y|x) · exp( r(x,y) / β )

Rearrange for r:

    r(x,y) = β · log( π*(y|x) / π_ref(y|x) )  +  β·log Z(x)

So *the reward is already implied by the policy*. Substitute that into the
Bradley-Terry model of a pairwise preference — P(y_w ≻ y_l) = σ(r_w - r_l) —
and log Z(x) cancels, because it depends only on x. What survives is:

    L_DPO = -log σ( β · [ (log π(y_w|x) - log π_ref(y_w|x))
                        - (log π(y_l|x) - log π_ref(y_l|x)) ] )

No reward model. No sampling loop. Just a classification loss on pairs you
already have. The quantity in brackets is the *implicit reward margin*.

--------------------------------------------------------------------------
Reading the gradient
--------------------------------------------------------------------------

    ∇L = -β · σ(-margin) · [ ∇log π(y_w|x) - ∇log π(y_l|x) ]

The σ(-margin) factor is the interesting part: it is an automatic difficulty
weight. Pairs the policy already gets right (large positive margin) contribute
almost nothing; pairs it gets wrong dominate the update. Nobody designed that —
it falls out of the algebra.

--------------------------------------------------------------------------
GRPO, in one paragraph
--------------------------------------------------------------------------

DPO needs a human preference pair. When correctness is *checkable* — code that
compiles, math that evaluates, a test that passes — you do not need a human or a
reward model. GRPO samples a group of G answers to the same prompt, scores each
with the verifier, and uses the group's mean as the baseline instead of a
learned value network:

    advantage_i = (score_i - mean(scores)) / std(scores)

That is why GRPO is the reasoning-model workhorse: the reward is free and
exact. See `grpo_advantages` below.
"""

from __future__ import annotations

import math


def sigmoid(x: float) -> float:
    # Numerically stable both ways: exp of a large positive x would overflow.
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def log_sigmoid(x: float) -> float:
    if x >= 0:
        return -math.log1p(math.exp(-x))
    return x - math.log1p(math.exp(x))


def implicit_reward(policy_logprob: float, ref_logprob: float, beta: float = 0.1) -> float:
    """β · log(π/π_ref) — the reward DPO never has to train a model for."""
    return beta * (policy_logprob - ref_logprob)


def dpo_margin(policy_chosen: float, policy_rejected: float,
               ref_chosen: float, ref_rejected: float, beta: float = 0.1) -> float:
    """The bracketed term: how much more the policy prefers chosen over rejected,
    *relative to where the reference model already stood*.

    Subtracting the reference is what stops the loss from rewarding a policy for
    preferences the reference already had.
    """
    return (implicit_reward(policy_chosen, ref_chosen, beta)
            - implicit_reward(policy_rejected, ref_rejected, beta))


def dpo_loss(policy_chosen: float, policy_rejected: float,
             ref_chosen: float, ref_rejected: float, beta: float = 0.1) -> float:
    """-log σ(margin). Zero when the policy strongly prefers the chosen answer."""
    return -log_sigmoid(dpo_margin(policy_chosen, policy_rejected,
                                   ref_chosen, ref_rejected, beta))


def dpo_grad_weight(policy_chosen: float, policy_rejected: float,
                    ref_chosen: float, ref_rejected: float,
                    beta: float = 0.1) -> float:
    """β·σ(-margin): the automatic difficulty weight on this pair's gradient."""
    margin = dpo_margin(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta)
    return beta * sigmoid(-margin)


def ppo_style_objective(reward: float, policy_logprob: float, ref_logprob: float,
                        beta: float = 0.1) -> float:
    """r - β·KL, with the single-sample KL estimate log π - log π_ref.

    Shown for contrast only: note that it needs `reward` handed to it from
    somewhere. That somewhere is a second trained network.
    """
    return reward - beta * (policy_logprob - ref_logprob)


def grpo_advantages(scores: list[float]) -> list[float]:
    """Group-relative advantages: (score - mean) / std, no value network."""
    n = len(scores)
    mean = sum(scores) / n
    var = sum((s - mean) ** 2 for s in scores) / n
    std = math.sqrt(var)
    if std == 0.0:
        # Every answer scored the same -- no signal, so no update.
        return [0.0] * n
    return [(s - mean) / std for s in scores]


if __name__ == "__main__":
    beta = 0.1
    ref_c, ref_r = -10.0, -10.0        # reference model is indifferent

    print("=" * 70)
    print("  LAB 03 — DEEPER DIVE: the DPO loss on toy numbers")
    print("=" * 70)
    print(f"\n  Reference model log-probs: chosen {ref_c}, rejected {ref_r} "
          "(indifferent)\n")
    print("  policy logπ(chosen) | logπ(rejected) | margin  |  loss  | grad weight")
    print("  " + "-" * 66)
    for pc, pr in [(-10.0, -10.0), (-9.0, -11.0), (-5.0, -15.0),
                   (-11.0, -9.0), (-15.0, -5.0)]:
        m = dpo_margin(pc, pr, ref_c, ref_r, beta)
        loss = dpo_loss(pc, pr, ref_c, ref_r, beta)
        w = dpo_grad_weight(pc, pr, ref_c, ref_r, beta)
        print(f"  {pc:18.1f} | {pr:14.1f} | {m:7.3f} | {loss:6.3f} | {w:.5f}")

    print("""
  Read the table top to bottom:
    row 1  policy matches the reference -> margin 0, loss ln2 = 0.693. The
           starting point of every DPO run.
    row 3  policy strongly prefers chosen -> loss near 0, gradient weight
           collapses to ~0.037. This pair is done; it stops pulling.
    row 5  policy prefers the REJECTED answer -> loss 1.31 and the largest
           gradient weight. The loss spends its budget where it is wrong.

  Nobody hand-designed that weighting. It is σ(-margin) falling out of the
  algebra, and it is why DPO trains stably without a reward model.
""")

    print("=" * 70)
    print("  GRPO: advantages from a group, no value network")
    print("=" * 70)
    scores = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0]   # 8 sampled answers, 4 correct
    print(f"\n  verifier scores ... {scores}")
    print(f"  advantages ........ {[round(a, 3) for a in grpo_advantages(scores)]}")
    print("\n  Correct answers get pushed up, incorrect pushed down, and the")
    print("  group mean is the baseline. The reward is a unit test, not a model.\n")
    print("=" * 70)
