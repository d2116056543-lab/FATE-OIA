import torch

from fate_oia.engine.fit_tida_traffic_boundary_cv import (
    _balanced_boundary_loss,
    _concatenate_rows,
    _fit_thresholds,
    _macro_f1,
)


def test_cv_threshold_fit_improves_simple_macro_f1():
    target = torch.tensor([[0.0], [0.0], [1.0], [1.0]])
    logits = torch.tensor([[-2.0], [-1.0], [0.2], [0.4]])
    threshold = _fit_thresholds(logits, target)
    assert _macro_f1(logits, target, threshold) == 1.0


def test_balanced_boundary_loss_has_finite_action_specific_gradient():
    base = torch.randn(8, 4)
    target = torch.tensor([[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]] * 4)
    delta = torch.zeros_like(base, requires_grad=True)
    loss = _balanced_boundary_loss(base - delta, base, target, torch.full((4,), 0.5), delta)
    loss.backward()
    assert torch.isfinite(loss)
    assert delta.grad is not None and delta.grad.abs().sum() > 0


def test_cv_row_concatenation_preserves_all_sources():
    rows = _concatenate_rows([
        {"x": torch.ones(2, 3)}, {"x": torch.zeros(4, 3)}
    ])
    assert rows["x"].shape == (6, 3)
    assert rows["x"][:2].sum() == 6
