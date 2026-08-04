"""
Lab 02 — a mini reverse-mode autodiff engine, in the standard library.

Why write this instead of importing torch? Because the point of the lab is that
a transformer is not magic, and neither is the thing that trains it. Roughly 200
lines below give you every derivative the model needs. You can read all of them.

Design choices that keep it small:

  * Every tensor is 2-D: `rows x cols`, a plain list of lists of floats. The
    model handles one sequence at a time, so a "batch" is just a loop that
    accumulates gradients -- which is exactly what a batch mathematically is.
  * Each op returns a new Tensor holding a closure that knows how to push
    gradient back to its inputs. `backward()` walks those closures in reverse
    topological order.
  * Gradients ACCUMULATE (`+=`). That is what makes a tensor used twice in the
    graph get the sum of both paths, and it is why you must call `zero_grad()`
    between optimizer steps.

If `numpy` happens to be installed we use it for matmul only -- the hot loop --
and the numerics are unchanged. Nothing here requires it.
"""

from __future__ import annotations

import math
import random

try:  # Optional: a ~10x faster matmul. Everything works without it.
    import numpy as _np
except ImportError:  # pragma: no cover - exercised on stdlib-only machines
    _np = None

Matrix = list


# --------------------------------------------------------------------------- #
# The tensor
# --------------------------------------------------------------------------- #
class Tensor:
    """A 2-D array of floats that remembers how it was computed."""

    __slots__ = ("data", "grad", "rows", "cols", "_backward", "_prev")

    def __init__(self, data: Matrix, _prev: tuple["Tensor", ...] = ()):
        self.data = data
        self.rows = len(data)
        self.cols = len(data[0]) if data else 0
        self.grad = [[0.0] * self.cols for _ in range(self.rows)]
        self._prev = _prev
        self._backward = lambda: None

    # -- construction helpers ------------------------------------------------
    @staticmethod
    def zeros(rows: int, cols: int) -> "Tensor":
        return Tensor([[0.0] * cols for _ in range(rows)])

    @staticmethod
    def randn(rows: int, cols: int, rng: random.Random, std: float = 0.02) -> "Tensor":
        """Small normal init. std=0.02 is the GPT-2 initialization."""
        return Tensor([[rng.gauss(0.0, std) for _ in range(cols)] for _ in range(rows)])

    def zero_grad(self) -> None:
        for row in self.grad:
            for i in range(len(row)):
                row[i] = 0.0

    # -- the engine ----------------------------------------------------------
    def backward(self) -> None:
        """Seed this (scalar) node with dL/dL = 1 and push gradient backwards."""
        topo: list[Tensor] = []
        seen: set[int] = set()
        # Iterative DFS -- a recursive one blows the stack on deep graphs.
        stack = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                topo.append(node)
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.append((node, True))
            for child in node._prev:
                if id(child) not in seen:
                    stack.append((child, False))

        self.grad[0][0] += 1.0
        for node in reversed(topo):
            node._backward()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Tensor({self.rows}x{self.cols})"


# --------------------------------------------------------------------------- #
# Ops. Each one: compute forward, then define how gradient flows back.
# --------------------------------------------------------------------------- #
def matmul(a: Tensor, b: Tensor) -> Tensor:
    """(n,k) @ (k,m) -> (n,m).  dA = dOut @ Bᵀ,  dB = Aᵀ @ dOut."""
    assert a.cols == b.rows, f"matmul shape mismatch: {a.cols} vs {b.rows}"

    if _np is not None:
        out_data = (_np.array(a.data) @ _np.array(b.data)).tolist()
    else:
        b_t = list(zip(*b.data))  # column-major once, so the inner loop is tight
        out_data = [[sum(x * y for x, y in zip(row, col)) for col in b_t] for row in a.data]

    out = Tensor(out_data, (a, b))

    def _backward() -> None:
        if _np is not None:
            g = _np.array(out.grad)
            da = g @ _np.array(b.data).T
            db = _np.array(a.data).T @ g
            for i in range(a.rows):
                a_row, d_row = a.grad[i], da[i]
                for j in range(a.cols):
                    a_row[j] += float(d_row[j])
            for i in range(b.rows):
                b_row, d_row = b.grad[i], db[i]
                for j in range(b.cols):
                    b_row[j] += float(d_row[j])
        else:
            for i in range(a.rows):
                g_row = out.grad[i]
                a_grad_row = a.grad[i]
                a_row = a.data[i]
                for k in range(a.cols):
                    b_row, b_grad_row = b.data[k], b.grad[k]
                    acc = 0.0
                    a_ik = a_row[k]
                    for j in range(b.cols):
                        acc += g_row[j] * b_row[j]
                        b_grad_row[j] += a_ik * g_row[j]
                    a_grad_row[k] += acc

    out._backward = _backward
    return out


def add(a: Tensor, b: Tensor) -> Tensor:
    """Elementwise add. `b` may be a (1, cols) row vector, broadcast down rows."""
    broadcast = b.rows == 1 and a.rows != 1
    out = Tensor(
        [[a.data[i][j] + b.data[0 if broadcast else i][j] for j in range(a.cols)]
         for i in range(a.rows)],
        (a, b),
    )

    def _backward() -> None:
        for i in range(a.rows):
            g_row = out.grad[i]
            a_row = a.grad[i]
            b_row = b.grad[0 if broadcast else i]
            for j in range(a.cols):
                a_row[j] += g_row[j]
                b_row[j] += g_row[j]

    out._backward = _backward
    return out


def scale(a: Tensor, s: float) -> Tensor:
    """Multiply by a constant (used for the 1/sqrt(head_dim) attention scaling)."""
    out = Tensor([[v * s for v in row] for row in a.data], (a,))

    def _backward() -> None:
        for i in range(a.rows):
            g_row, a_row = out.grad[i], a.grad[i]
            for j in range(a.cols):
                a_row[j] += g_row[j] * s

    out._backward = _backward
    return out


def transpose(a: Tensor) -> Tensor:
    out = Tensor([list(col) for col in zip(*a.data)], (a,))

    def _backward() -> None:
        for i in range(out.rows):
            g_row = out.grad[i]
            for j in range(out.cols):
                a.grad[j][i] += g_row[j]

    out._backward = _backward
    return out


def gelu(a: Tensor) -> Tensor:
    """GELU (tanh approximation) -- the activation GPT-2 uses.

        g(x) = 0.5x(1 + tanh(c(x + 0.044715x³))),   c = sqrt(2/pi)
    """
    c = math.sqrt(2.0 / math.pi)
    cache: list[list[float]] = []
    out_data = []
    for row in a.data:
        o_row, d_row = [], []
        for x in row:
            u = c * (x + 0.044715 * x ** 3)
            t = math.tanh(u)
            o_row.append(0.5 * x * (1.0 + t))
            # dg/dx = 0.5(1+t) + 0.5x(1-t²)·c(1 + 3·0.044715x²)
            d_row.append(0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * c * (1 + 0.134145 * x * x))
        out_data.append(o_row)
        cache.append(d_row)

    out = Tensor(out_data, (a,))

    def _backward() -> None:
        for i in range(a.rows):
            g_row, a_row, d_row = out.grad[i], a.grad[i], cache[i]
            for j in range(a.cols):
                a_row[j] += g_row[j] * d_row[j]

    out._backward = _backward
    return out


def layernorm(a: Tensor, eps: float = 1e-5) -> Tensor:
    """Normalize each row to zero mean / unit variance (no affine params here;
    the model applies its own gain and bias afterwards)."""
    n = a.cols
    norm_data, inv_std_cache = [], []
    for row in a.data:
        mean = sum(row) / n
        var = sum((v - mean) ** 2 for v in row) / n
        inv_std = 1.0 / math.sqrt(var + eps)
        norm_data.append([(v - mean) * inv_std for v in row])
        inv_std_cache.append(inv_std)

    out = Tensor(norm_data, (a,))

    def _backward() -> None:
        # dx = (1/std)·(dy - mean(dy) - y·mean(dy·y))
        for i in range(a.rows):
            g_row, y_row = out.grad[i], out.data[i]
            inv_std = inv_std_cache[i]
            mean_g = sum(g_row) / n
            mean_gy = sum(g_row[j] * y_row[j] for j in range(n)) / n
            a_row = a.grad[i]
            for j in range(n):
                a_row[j] += inv_std * (g_row[j] - mean_g - y_row[j] * mean_gy)

    out._backward = _backward
    return out


def softmax_rows(a: Tensor) -> Tensor:
    """Row-wise softmax.  dx_i = y_i(dy_i - Σ_j dy_j y_j)."""
    out_data = []
    for row in a.data:
        m = max(row)
        exps = [math.exp(v - m) for v in row]
        total = sum(exps)
        out_data.append([e / total for e in exps])

    out = Tensor(out_data, (a,))

    def _backward() -> None:
        for i in range(a.rows):
            g_row, y_row, a_row = out.grad[i], out.data[i], a.grad[i]
            dot = sum(g_row[j] * y_row[j] for j in range(a.cols))
            for j in range(a.cols):
                a_row[j] += y_row[j] * (g_row[j] - dot)

    out._backward = _backward
    return out


def causal_softmax(a: Tensor) -> Tensor:
    """Softmax over each row, but position t may only attend to positions <= t.

    This single op is the whole "a language model cannot read the future" rule.
    Masked entries are dropped before the exponential, so they get exactly zero
    probability and exactly zero gradient.
    """
    out_data = []
    for i, row in enumerate(a.data):
        limit = i + 1  # row i is query position i; keys 0..i are visible
        visible = row[:limit]
        m = max(visible)
        exps = [math.exp(v - m) for v in visible]
        total = sum(exps)
        out_data.append([e / total for e in exps] + [0.0] * (a.cols - limit))

    out = Tensor(out_data, (a,))

    def _backward() -> None:
        for i in range(a.rows):
            limit = i + 1
            g_row, y_row, a_row = out.grad[i], out.data[i], a.grad[i]
            dot = sum(g_row[j] * y_row[j] for j in range(limit))
            for j in range(limit):
                a_row[j] += y_row[j] * (g_row[j] - dot)

    out._backward = _backward
    return out


def slice_cols(a: Tensor, start: int, end: int) -> Tensor:
    """Take columns [start, end) -- how the attention heads are split."""
    out = Tensor([row[start:end] for row in a.data], (a,))

    def _backward() -> None:
        for i in range(a.rows):
            g_row, a_row = out.grad[i], a.grad[i]
            for j in range(end - start):
                a_row[start + j] += g_row[j]

    out._backward = _backward
    return out


def concat_cols(parts: list[Tensor]) -> Tensor:
    """Glue the per-head outputs back into one (rows, n_embd) tensor."""
    rows = parts[0].rows
    out = Tensor([[v for p in parts for v in p.data[i]] for i in range(rows)], tuple(parts))

    def _backward() -> None:
        for i in range(rows):
            g_row = out.grad[i]
            offset = 0
            for p in parts:
                p_row = p.grad[i]
                for j in range(p.cols):
                    p_row[j] += g_row[offset + j]
                offset += p.cols

    out._backward = _backward
    return out


def embed(weight: Tensor, ids: list[int]) -> Tensor:
    """Row lookup. The gradient of a lookup is a scatter-add back into the table."""
    out = Tensor([weight.data[i][:] for i in ids], (weight,))

    def _backward() -> None:
        for pos, token_id in enumerate(ids):
            g_row, w_row = out.grad[pos], weight.grad[token_id]
            for j in range(weight.cols):
                w_row[j] += g_row[j]

    out._backward = _backward
    return out


def cross_entropy(logits: Tensor, targets: list[int]) -> Tensor:
    """Mean negative log-likelihood of the target tokens.

    The gradient is famously clean: (softmax(logits) - one_hot(target)) / N.
    That subtraction is the entire learning signal -- "push down what you
    predicted, push up what was actually next".
    """
    assert len(targets) == logits.rows, "one target per row"
    n = logits.rows
    total = 0.0
    probs: list[list[float]] = []
    for row, t in zip(logits.data, targets):
        m = max(row)
        exps = [math.exp(v - m) for v in row]
        s = sum(exps)
        probs.append([e / s for e in exps])
        total += -(row[t] - m - math.log(s))

    out = Tensor([[total / n]], (logits,))

    def _backward() -> None:
        g = out.grad[0][0] / n
        for i, t in enumerate(targets):
            p_row, l_row = probs[i], logits.grad[i]
            for j in range(logits.cols):
                l_row[j] += g * (p_row[j] - (1.0 if j == t else 0.0))

    out._backward = _backward
    return out
