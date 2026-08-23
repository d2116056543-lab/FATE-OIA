import torch

from fate_oia.losses.tida_traffic_trajectory_losses import (
    trajectory_boundary_correction_loss,
    trajectory_selected_control_loss,
    trajectory_utility_calibration_loss,
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
    selected = torch.tensor([[0.004, -0.004], [-0.004, 0.004]], requires_grad=True)
    control_good = torch.zeros_like(selected)
    control_bad = 2.0 * selected.detach()
    good_loss = trajectory_selected_control_loss(base, selected, control_good, target, support)
    bad_loss = trajectory_selected_control_loss(base, selected, control_bad, target, support)
    assert good_loss < bad_loss
    good_loss.backward()
    assert selected.grad is not None and selected.grad.abs().sum() > 0


def test_boundary_correction_uses_train_calib_deploy_boundary():
    base = torch.tensor([[0.55, -0.55]])
    deploy_boundary = torch.tensor([0.60, -0.60])
    target = torch.tensor([[1.0, 0.0]])
    support = torch.ones_like(base)
    aligned = torch.tensor([[0.20, -0.20]])
    opposed = -aligned
    aligned_loss = trajectory_boundary_correction_loss(
        base, aligned, target, support, deploy_boundary_logits=deploy_boundary
    )
    opposed_loss = trajectory_boundary_correction_loss(
        base, opposed, target, support, deploy_boundary_logits=deploy_boundary
    )
    assert aligned_loss < opposed_loss


def test_selected_control_margin_is_reachable_at_low_support():
    base = torch.zeros(1, 1)
    target = torch.ones_like(base)
    support = torch.full_like(base, 0.10)
    trust = torch.full_like(base, 0.25)
    selected = torch.tensor([[0.004]], requires_grad=True)
    control = torch.zeros_like(selected)
    loss = trajectory_selected_control_loss(
        base, selected, control, target, support,
        trajectory_trust=trust, trajectory_cap=0.08,
    )
    assert loss.item() == 0.0


def test_selected_control_uses_same_saturating_support_gate_as_forward():
    base = torch.zeros(1, 1)
    target = torch.ones_like(base)
    support = torch.full_like(base, 0.10)
    trust = torch.full_like(base, 0.25)
    selected = torch.tensor([[0.0003]], requires_grad=True)
    loss = trajectory_selected_control_loss(
        base, selected, torch.zeros_like(selected), target, support,
        trajectory_trust=trust, trajectory_cap=0.08,
    )
    assert loss.item() > 0.001
    loss.backward()
    assert selected.grad is not None and selected.grad.item() < 0


def test_selected_control_reachable_margin_includes_order_gate():
    base = torch.zeros(1, 1)
    target = torch.ones_like(base)
    support = torch.ones_like(base)
    trust = torch.full_like(base, 0.5)
    order_gate = torch.full_like(base, 0.25)
    # 0.25 * 0.08 * 1.0 * 0.5 * 0.25 = 0.0025
    selected = torch.tensor([[0.0026]], requires_grad=True)
    loss = trajectory_selected_control_loss(
        base, selected, torch.zeros_like(selected), target, support,
        trajectory_trust=trust, trajectory_order_gate=order_gate, trajectory_cap=0.08,
    )
    assert loss.item() == 0.0


def test_boundary_correction_is_invariant_to_negative_replication_per_action():
    base = torch.zeros(2, 1)
    target = torch.tensor([[1.0], [0.0]])
    support = torch.ones_like(base)
    delta = torch.tensor([[-0.01], [-0.01]])
    reference = trajectory_boundary_correction_loss(base, delta, target, support)

    repeated = trajectory_boundary_correction_loss(
        torch.cat((base[:1], base[1:].repeat(20, 1))),
        torch.cat((delta[:1], delta[1:].repeat(20, 1))),
        torch.cat((target[:1], target[1:].repeat(20, 1))),
        torch.cat((support[:1], support[1:].repeat(20, 1))),
    )
    assert torch.allclose(reference, repeated, atol=1e-7)


def test_selected_control_is_invariant_to_negative_replication_per_action():
    base = torch.zeros(2, 1)
    target = torch.tensor([[1.0], [0.0]])
    support = torch.ones_like(base)
    selected = torch.tensor([[-0.001], [-0.001]])
    control = torch.zeros_like(selected)
    reference = trajectory_selected_control_loss(
        base, selected, control, target, support,
    )

    repeated = trajectory_selected_control_loss(
        torch.cat((base[:1], base[1:].repeat(20, 1))),
        torch.cat((selected[:1], selected[1:].repeat(20, 1))),
        torch.cat((control[:1], control[1:].repeat(20, 1))),
        torch.cat((target[:1], target[1:].repeat(20, 1))),
        torch.cat((support[:1], support[1:].repeat(20, 1))),
    )
    assert torch.allclose(reference, repeated, atol=1e-7)


def test_utility_calibration_opens_for_helpful_and_closes_for_harmful_candidates():
    target = torch.tensor([[1.0, 0.0]])
    candidate = torch.tensor([[0.02, 0.02]])
    correct_logits = torch.tensor([[5.0, -5.0]], requires_grad=True)
    reversed_logits = -correct_logits.detach()
    correct_loss = trajectory_utility_calibration_loss(correct_logits, candidate, target)
    reversed_loss = trajectory_utility_calibration_loss(reversed_logits, candidate, target)
    assert correct_loss < reversed_loss
    correct_loss.backward()
    assert correct_logits.grad is not None and torch.isfinite(correct_logits.grad).all()
