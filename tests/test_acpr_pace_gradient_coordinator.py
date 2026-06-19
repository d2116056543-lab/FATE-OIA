import torch

from fate_oia.utils.acpr_pace_gradient_coordinator import common_descent_gradient


def test_common_descent_aligned_returns_sum():
    a = torch.tensor([1.0, 0.0])
    b = torch.tensor([0.5, 0.0])
    c, s = common_descent_gradient(a, b)
    assert torch.allclose(c, a + b)
    assert s["gradient_conflict"] == 0.0


def test_common_descent_conflict_is_finite():
    a = torch.tensor([1.0, 0.0])
    b = torch.tensor([-0.5, 1.0])
    c, s = common_descent_gradient(a, b)
    assert torch.isfinite(c).all()
    assert 0.0 <= s["gradient_alpha"] <= 1.0
