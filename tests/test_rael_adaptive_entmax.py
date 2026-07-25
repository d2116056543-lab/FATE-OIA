from __future__ import annotations

import inspect
import time

import pytest
import torch

from fate_oia.models.rael_relation_contributions import (
    RAELUnaryContribution,
    entmax_bisect,
)


def _entmax15_closed_form_reference(scores: torch.Tensor) -> torch.Tensor:
    """Independent alpha=1.5 oracle via its sorted quadratic support equation."""
    values = scores.double()
    ordered, _ = values.sort(dim=-1, descending=True)
    u = ordered * 0.5
    output = torch.zeros_like(values)
    for row in range(values.shape[0]):
        chosen_threshold = None
        for support_size in range(1, values.shape[-1] + 1):
            active = u[row, :support_size]
            total = active.sum()
            square_total = active.square().sum()
            discriminant = total.square() - support_size * (square_total - 1.0)
            threshold = (total - discriminant.clamp_min(0.0).sqrt()) / support_size
            next_value = u[row, support_size] if support_size < values.shape[-1] else None
            if bool((active[-1] > threshold) and (next_value is None or next_value <= threshold)):
                chosen_threshold = threshold
                break
        if chosen_threshold is None:
            raise AssertionError("reference support resolution failed")
        output[row] = (values[row] * 0.5 - chosen_threshold).clamp_min(0.0).square()
    return output


def _implicit_score_vjp(probabilities: torch.Tensor, alpha: torch.Tensor, upstream: torch.Tensor) -> torch.Tensor:
    """Independent fixed-support alpha-entmax score VJP."""
    alpha_last = alpha.reshape(-1, 1)
    active = probabilities > 1e-14
    gppr = torch.where(active, probabilities.pow(2.0 - alpha_last), torch.zeros_like(probabilities))
    centered = upstream - (gppr * upstream).sum(dim=-1, keepdim=True) / gppr.sum(dim=-1, keepdim=True)
    return gppr * centered


def test_entmax_bisect_normalizes_broadcast_alpha_is_sparse_and_differentiable() -> None:
    scores = torch.tensor(
        [
            [[7.0, 0.1, -2.0, -5.0, -8.0], [0.2, 0.1, 0.0, -0.1, -0.2]],
            [[4.0, -1.0, -2.0, -3.0, -4.0], [0.3, 0.2, 0.1, 0.0, -0.1]],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    alpha = torch.tensor([[[1.10], [1.45]]], dtype=torch.float32, requires_grad=True)

    probabilities = entmax_bisect(scores, alpha=alpha, dim=-1)

    assert probabilities.shape == scores.shape
    assert torch.allclose(
        probabilities.sum(dim=-1),
        torch.ones_like(probabilities[..., 0]),
        atol=1e-5,
        rtol=1e-5,
    )
    assert bool((probabilities >= 0).all())
    assert int((probabilities == 0).sum()) > 0

    objective = (probabilities * torch.linspace(-1.0, 1.0, scores.shape[-1])).sum()
    objective.backward()
    assert scores.grad is not None and bool(torch.isfinite(scores.grad).all())
    assert alpha.grad is not None and bool(torch.isfinite(alpha.grad).all())


def test_entmax_operates_on_last_dimension_and_is_low_precision_safe() -> None:
    with pytest.raises(ValueError, match="last dimension"):
        entmax_bisect(torch.randn(2, 3, 4), alpha=1.2, dim=1)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    scores = torch.tensor(
        [[15.0, 2.0, -9.0, -30.0], [0.0, -0.5, -1.0, -9.0]],
        dtype=torch.bfloat16,
        device=device,
        requires_grad=True,
    )
    probabilities = entmax_bisect(scores, alpha=torch.tensor([1.05, 1.5], device=device))
    assert probabilities.dtype == torch.bfloat16
    assert bool(torch.isfinite(probabilities).all())
    assert bool((probabilities >= 0).all())
    assert torch.allclose(
        probabilities.float().sum(dim=-1),
        torch.ones(2, device=device),
        atol=2e-2,
        rtol=2e-2,
    )
    probabilities.float().square().sum().backward()
    assert scores.grad is not None and bool(torch.isfinite(scores.grad).all())


def test_target_alpha_is_learned_bounded_and_starts_near_one_tenth() -> None:
    unary = RAELUnaryContribution(num_targets=4, dim=16, attribute_dim=5)
    alpha = unary.adaptive_alpha()
    assert alpha.shape == (4,)
    assert torch.allclose(alpha, torch.full_like(alpha, 1.10), atol=2e-3, rtol=0.0)

    with torch.no_grad():
        unary.eta.copy_(torch.tensor([-100.0, -3.0, 3.0, 100.0]))
    alpha = unary.adaptive_alpha()
    assert bool((alpha >= 1.05).all())
    assert bool((alpha <= 1.50).all())
    assert alpha[0].item() == pytest.approx(1.05, abs=1e-5)
    assert alpha[-1].item() == pytest.approx(1.50, abs=1e-5)


def test_entmax_is_pure_pytorch_without_external_entmax_or_softmax_substitution() -> None:
    source = inspect.getsource(entmax_bisect)
    assert "softmax" not in source.lower()
    assert "entmax." not in source.lower()
    assert "torch." in source


def test_entmax_uses_custom_implicit_backward_not_autograd_through_bisection() -> None:
    scores = torch.tensor([[0.2, 0.1, 0.0, -0.1]], dtype=torch.float64, requires_grad=True)
    probabilities = entmax_bisect(scores, alpha=torch.tensor([1.1], dtype=torch.float64))
    assert probabilities.grad_fn is not None
    assert "EntmaxBisectFunctionBackward" in type(probabilities.grad_fn).__name__


def test_entmax_is_shift_invariant_for_extreme_common_offsets_and_normalized() -> None:
    # The nontrivial row is intentionally separated by representable double-precision gaps at 1e20.
    base = torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.0, -1.0e6, -2.0e6, -4.0e6]], dtype=torch.float64)
    alpha = torch.tensor([1.1, 1.5], dtype=torch.float64)
    expected = entmax_bisect(base, alpha=alpha)
    for offset in (1.0e8, 1.0e10, 1.0e20):
        actual = entmax_bisect(base + offset, alpha=alpha)
        assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)
        assert torch.allclose(actual.sum(dim=-1), torch.ones(2, dtype=torch.float64), atol=1e-10)


def test_entmax_matches_independent_entmax15_closed_form_and_softmax_limit() -> None:
    scores = torch.tensor([[0.45, 0.20, -0.10, -0.25], [0.10, 0.05, -0.10, -0.20]], dtype=torch.float64)
    alpha15 = entmax_bisect(scores, alpha=torch.tensor([1.5, 1.5], dtype=torch.float64))
    assert torch.allclose(alpha15, _entmax15_closed_form_reference(scores), atol=2e-8, rtol=2e-8)

    nearly_softmax = entmax_bisect(scores, alpha=torch.tensor([1.0001, 1.0001], dtype=torch.float64))
    assert torch.allclose(nearly_softmax, torch.softmax(scores, dim=-1), atol=8e-4, rtol=8e-4)


@pytest.mark.parametrize("alpha_value", [1.05, 1.0501, 1.1, 1.5])
def test_entmax_implicit_score_jacobian_and_alpha_gradient_match_references(alpha_value: float) -> None:
    scores = torch.tensor([[0.20, 0.10, 0.00, -0.10]], dtype=torch.float64, requires_grad=True)
    alpha = torch.tensor([alpha_value], dtype=torch.float64, requires_grad=True)
    upstream = torch.tensor([[0.70, -0.40, 0.20, 0.10]], dtype=torch.float64)
    probabilities = entmax_bisect(scores, alpha=alpha)
    loss = (probabilities * upstream).sum()
    loss.backward()

    expected_score = _implicit_score_vjp(probabilities.detach(), alpha.detach(), upstream)
    assert torch.allclose(scores.grad, expected_score, atol=3e-7, rtol=3e-5)

    step = 2e-4
    with torch.no_grad():
        plus = (entmax_bisect(scores.detach(), alpha=torch.tensor([alpha_value + step], dtype=torch.float64)) * upstream).sum()
        minus = (entmax_bisect(scores.detach(), alpha=torch.tensor([alpha_value - step], dtype=torch.float64)) * upstream).sum()
    finite_difference = (plus - minus) / (2.0 * step)
    assert torch.allclose(alpha.grad, finite_difference.reshape_as(alpha.grad), atol=3e-5, rtol=3e-3)


@pytest.mark.parametrize("slot_count", [4, 21, 25])
def test_entmax_profile_k4_k21_k25_stays_below_thirty_milliseconds(slot_count: int) -> None:
    scores = torch.randn(16, slot_count, dtype=torch.float32)
    alpha = torch.full((16,), 1.1, dtype=torch.float32)
    for _ in range(8):
        entmax_bisect(scores, alpha=alpha)
    started = time.perf_counter()
    for _ in range(64):
        probabilities = entmax_bisect(scores, alpha=alpha)
    elapsed_per_call = (time.perf_counter() - started) / 64.0
    assert bool(torch.isfinite(probabilities).all())
    assert elapsed_per_call < 0.030, (slot_count, elapsed_per_call)
