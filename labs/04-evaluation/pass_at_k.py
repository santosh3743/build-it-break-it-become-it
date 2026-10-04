"""
Lab 04 — pass@k, and why the obvious way to compute it is wrong.

For code (and anything else with an automatic checker), you often let the
model try more than once. pass@k is "the probability that at least one of k
samples passes the checker".

The obvious estimate: measure the per-sample pass rate p = c/n, then say
pass@k = 1 - (1 - p)^k. That's biased: it **underestimates** pass@k (Chen et
al., 2021, the HumanEval paper, point this out and give an unbiased one).
Draw n >= k samples, count the c that pass, and ask: if I picked k of these n
at random without replacement, what's the chance at least one is correct?

    pass@k = 1 - C(n - c, k) / C(n, k)

C(n - c, k) / C(n, k) is the chance that all k picks come from the n - c
failures. Average this over problems to get the benchmark's pass@k.

The HumanEval paper computes it as a running product rather than with big
binomials, which avoids overflow for large n:

    1 - prod_{i = n-c+1}^{n} (1 - k / i)

Both forms are implemented below and the tests check they agree.

The toy task: write an integer expression that computes a described
quantity. The verifier really evaluates each sample, using a tiny AST
evaluator that accepts only integer arithmetic. Never run model-written code
with `eval` or `exec`. Even a toy verifier is a sandbox, and Act II shows what
happens when one isn't.
"""

from __future__ import annotations

import ast
import operator
from dataclasses import dataclass
from math import comb

from models import unit


# --------------------------------------------------------------------------- #
# The estimators
# --------------------------------------------------------------------------- #
def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k from n samples of which c passed (Chen et al., 2021)."""
    if not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError(f"need 0 <= c <= n and 1 <= k <= n, got n={n} c={c} k={k}")
    if n - c < k:          # fewer failures than picks: some pick must be a pass
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def pass_at_k_product(n: int, c: int, k: int) -> float:
    """The same quantity as a running product (the HumanEval paper's form)."""
    if n - c < k:
        return 1.0
    prob_all_fail = 1.0
    for i in range(n - c + 1, n + 1):
        prob_all_fail *= 1.0 - k / i
    return 1.0 - prob_all_fail


def naive_pass_at_k(n: int, c: int, k: int) -> float:
    """The biased plug-in estimate 1 - (1 - c/n)^k. Shown only to compare."""
    return 1.0 - (1.0 - c / n) ** k


# --------------------------------------------------------------------------- #
# A safe verifier: integer arithmetic only
# --------------------------------------------------------------------------- #
_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}


def safe_eval(expr: str) -> int:
    """Evaluate an integer arithmetic expression, rejecting everything else.

    No names, no calls, no attributes, no huge powers. Anything outside the
    allowlist raises ValueError, and the verifier counts that as a fail.
    """
    def ev(node: ast.AST) -> int:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -ev(node.operand)
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 64:
                raise ValueError("exponent too large")
            return _BIN[type(node.op)](left, right)
        raise ValueError(f"disallowed syntax: {type(node).__name__}")

    try:
        return ev(ast.parse(expr, mode="eval"))
    except (SyntaxError, ZeroDivisionError) as exc:
        raise ValueError(str(exc)) from exc


def verify(problem: "Problem", sample: str) -> bool:
    try:
        return safe_eval(sample) == safe_eval(problem.reference)
    except ValueError:
        return False


# --------------------------------------------------------------------------- #
# The toy problems and two mock samplers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Problem:
    id: str
    prompt: str
    reference: str        # a correct expression
    correct_forms: tuple  # other correct ways a model might write it
    wrong_forms: tuple    # typical bugs: wrong operator, off by one, wrong unit


PROBLEMS = [
    Problem("p01", "seconds in a day", "24*60*60",
            ("60*60*24", "86400", "24*3600"), ("24*60", "24+60+60", "24*60*60*7")),
    Problem("p02", "sum of the integers 1..100", "100*101//2",
            ("5050", "101*50", "(1+100)*100//2"), ("100*100//2", "100*99//2", "100*101")),
    Problem("p03", "bytes in a kibibyte", "2**10",
            ("1024", "4**5", "32*32"), ("1000", "2**8", "10**3")),
    Problem("p04", "minutes in a week", "7*24*60",
            ("10080", "60*24*7", "168*60"), ("7*24", "7*60", "5*24*60")),
    Problem("p05", "even numbers from 1 to 50 inclusive", "50//2",
            ("25", "(50-0)//2"), ("50//2+1", "24", "50%2")),
    Problem("p06", "handshakes among 10 people", "10*9//2",
            ("45", "(10*10-10)//2"), ("10*9", "10*10//2", "10*11//2")),
    Problem("p07", "days in a leap year", "366",
            ("365+1", "12*30+6"), ("365", "12*30", "52*7")),
    Problem("p08", "edges of a cube", "12",
            ("6*4//2", "3*4"), ("6", "8", "6*4")),
    Problem("p09", "leap years from 2001 to 2100 inclusive", "24",
            ("100//4-1", "25-1"), ("25", "100//4", "100//4+1")),
    Problem("p10", "last digit of 7**4", "7**4%10",
            ("1", "2401%10"), ("7", "9", "7*4%10")),
]


@dataclass(frozen=True)
class Sampler:
    """A mock model that writes `n` attempts per problem.

    `solve_rate[i]` is the chance one sample of problem i is a correct form.
    A focused (low-temperature) model is very reliable on what it knows and
    hopeless on the rest. A diverse (high-temperature) model is unreliable
    everywhere, but sometimes right everywhere.
    """
    name: str
    solve_rate: tuple

    def sample(self, problem_index: int, j: int) -> str:
        p = PROBLEMS[problem_index]
        u = unit(f"{self.name}:{p.id}:{j}")
        pool = p.correct_forms + (p.reference,) if u < self.solve_rate[problem_index] \
            else p.wrong_forms
        return pool[int(unit(f"{self.name}:{p.id}:{j}:form") * len(pool))]


FOCUSED = Sampler("focused (low temp)", (0.95, 0.90, 0.90, 0.85, 0.80, 0.0, 0.0, 0.05, 0.0, 0.0))
DIVERSE = Sampler("diverse (high temp)", (0.35, 0.30, 0.30, 0.25, 0.30, 0.20, 0.25, 0.20, 0.15, 0.20))
SAMPLERS = [FOCUSED, DIVERSE]
N_SAMPLES = 20
KS = (1, 5, 10)


def count_correct(sampler: Sampler, n: int = N_SAMPLES) -> list[int]:
    """c for each problem: how many of n samples the verifier accepts."""
    return [sum(verify(p, sampler.sample(i, j)) for j in range(n))
            for i, p in enumerate(PROBLEMS)]


def benchmark(sampler: Sampler, n: int = N_SAMPLES, ks=KS) -> dict:
    """Mean unbiased and naive pass@k over all problems."""
    cs = count_correct(sampler, n)
    out = {"c": cs}
    for k in ks:
        out[f"pass@{k}"] = sum(pass_at_k(n, c, k) for c in cs) / len(cs)
        out[f"naive@{k}"] = sum(naive_pass_at_k(n, c, k) for c in cs) / len(cs)
    return out


def table(n: int = N_SAMPLES, ks=KS) -> str:
    lines = [f"  sampler             | c per problem (of n={n})              | "
             + " | ".join(f"pass@{k:<2d}" for k in ks),
             "  " + "-" * 92]
    for s in SAMPLERS:
        b = benchmark(s, n, ks)
        cs = " ".join(f"{c:2d}" for c in b["c"])
        lines.append(f"  {s.name:<19s} | {cs:<37s} | "
                     + " | ".join(f"{b[f'pass@{k}'] * 100:6.1f}%" for k in ks))
    return "\n".join(lines)
