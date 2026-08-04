"""
Tests for the mini-autograd engine.

The whole trainer rests on these gradients being right, so we check them the
only way that actually proves it: against finite differences. If d(loss)/dw
computed by backprop disagrees with (loss(w+h) - loss(w-h)) / 2h, backprop is
wrong -- no amount of "the loss went down" can rescue that.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autograd import Tensor, cross_entropy, gelu, layernorm, matmul, softmax_rows  # noqa: E402


def numeric_grad(build_loss, tensor, i, j, h=1e-5):
    """Central-difference estimate of d(loss)/d(tensor[i][j])."""
    original = tensor.data[i][j]

    tensor.data[i][j] = original + h
    plus = build_loss().data[0][0]

    tensor.data[i][j] = original - h
    minus = build_loss().data[0][0]

    tensor.data[i][j] = original
    return (plus - minus) / (2 * h)


def check_grads(build_loss, params, tol=1e-4):
    """Backprop every param, then compare each entry against finite differences."""
    for p in params:
        p.zero_grad()
    loss = build_loss()
    loss.backward()

    for p in params:
        for i in range(p.rows):
            for j in range(p.cols):
                analytic = p.grad[i][j]
                numeric = numeric_grad(build_loss, p, i, j)
                assert abs(analytic - numeric) < tol, (
                    f"grad mismatch at [{i}][{j}]: backprop={analytic:.6f} "
                    f"finite-diff={numeric:.6f}"
                )


# --------------------------------------------------------------------------- #
# Shapes and basic values
# --------------------------------------------------------------------------- #
def test_matmul_shape_and_value():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[5.0], [6.0]])
    out = matmul(a, b)
    assert (out.rows, out.cols) == (2, 1)
    assert out.data == [[17.0], [39.0]]


def test_softmax_rows_sums_to_one():
    out = softmax_rows(Tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]))
    for row in out.data:
        assert abs(sum(row) - 1.0) < 1e-9
    # Uniform input -> uniform output.
    assert all(abs(v - 1 / 3) < 1e-9 for v in out.data[1])


def test_layernorm_zero_mean_unit_variance():
    out = layernorm(Tensor([[1.0, 2.0, 3.0, 4.0]]))
    row = out.data[0]
    mean = sum(row) / len(row)
    var = sum((v - mean) ** 2 for v in row) / len(row)
    assert abs(mean) < 1e-6
    assert abs(var - 1.0) < 1e-3


def test_cross_entropy_of_uniform_logits_is_log_vocab():
    import math

    logits = Tensor([[0.0, 0.0, 0.0, 0.0]])
    loss = cross_entropy(logits, [2])
    assert abs(loss.data[0][0] - math.log(4)) < 1e-9


def test_cross_entropy_is_lower_when_target_is_favoured():
    confident = cross_entropy(Tensor([[10.0, 0.0, 0.0]]), [0])
    wrong = cross_entropy(Tensor([[10.0, 0.0, 0.0]]), [1])
    assert confident.data[0][0] < wrong.data[0][0]


# --------------------------------------------------------------------------- #
# Gradient correctness -- the tests that actually matter
# --------------------------------------------------------------------------- #
def test_matmul_gradients_match_finite_differences():
    a = Tensor([[0.3, -0.7, 0.5], [1.1, 0.2, -0.4]])
    b = Tensor([[0.2, 0.9], [-0.5, 0.1], [0.7, -0.3]])
    check_grads(lambda: cross_entropy(matmul(a, b), [0, 1]), [a, b])


def test_gelu_gradients_match_finite_differences():
    w = Tensor([[0.5, -1.2, 0.8], [0.1, 0.4, -0.9]])
    check_grads(lambda: cross_entropy(gelu(w), [1, 2]), [w])


def test_layernorm_gradients_match_finite_differences():
    w = Tensor([[0.5, -1.2, 0.8, 0.3], [0.1, 0.4, -0.9, 1.5]])
    check_grads(lambda: cross_entropy(layernorm(w), [1, 2]), [w])


def test_softmax_gradients_match_finite_differences():
    w = Tensor([[0.4, -0.6, 0.9], [1.2, 0.1, -0.2]])
    check_grads(lambda: cross_entropy(softmax_rows(w), [2, 0]), [w])


def test_gradients_accumulate_through_a_reused_tensor():
    """A tensor used twice must receive the sum of both paths' gradients."""
    w = Tensor([[0.3, 0.6], [-0.2, 0.4]])
    check_grads(lambda: cross_entropy(matmul(w, w), [0, 1]), [w])


def test_zero_grad_clears_previous_backward():
    a = Tensor([[0.3, -0.7], [1.1, 0.2]])
    build = lambda: cross_entropy(a, [0, 1])  # noqa: E731
    build().backward()
    first = [row[:] for row in a.grad]
    build().backward()
    doubled = a.grad
    # Without zero_grad, gradients accumulate.
    assert doubled[0][0] == first[0][0] * 2
    a.zero_grad()
    assert a.grad == [[0.0, 0.0], [0.0, 0.0]]


if __name__ == "__main__":
    # No-pytest fallback: python tests/test_autograd.py
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
            passed += 1
    print(f"\n{passed} passed")
