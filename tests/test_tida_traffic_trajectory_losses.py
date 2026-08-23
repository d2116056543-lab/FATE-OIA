import torch

from fate_oia.losses.tida_traffic_trajectory_losses import (
    trajectory_boundary_correction_loss,
    trajectory_selected_control_loss,
)


def test_boundary_correction_prefers_gt_aligned_trajectory_delta():
    base = torch.tensor([[-0.1, 0.1], [0.2, -0.2]])
    target = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    support = torch.full_like(base, 0.8)
    good = torch.tensor([[0.3, -0.3], [0.3, -0.3]], requires_grad=True)
    bad = -good.detach()
    good_loss = trajectory_boundary_correction_loss(base, good, target, support)
    bad_loss = trajectory_boundary_correction_loss(base, bad, target, support)
    assert good_loss < bad_loss
    good_loss.backward()
    assert good.grad is not None and torch.isfinite(good.grad).all()


def test_selected_control_requires_ordered_trajectory_to_improve_gt_margin():
    base = torch.zeros(2, 2)
    target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    support = torch.ones_like(base)
    selected = torch.tensor([[0.01, -0.01], [-0.01, 0.01]], requires_grad=True)
    control_good = torch.zeros_like(selected)
    control_bad = 2.0 * selected.detach()
    good_loss = trajectory_selected_control_loss(base, selected, control_good, target, support)
    bad_loss = trajectory_selected_control_loss(base, selected, control_bad, target, support)
    assert good_loss < bad_loss
    good_loss.backward()
    assert selected.grad is not None and selected.grad.abs().sum() > 0
