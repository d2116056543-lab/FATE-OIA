import torch

from fate_oia.losses.tida_losses import (
    target_conditioned_geometric_correction_loss,
    target_conditioned_geometric_ranking_loss,
)


def test_target_conditioned_correction_rewards_gt_margin_and_has_finite_gradient():
    base = torch.tensor([[-0.2, 0.2], [0.1, -0.1]])
    target = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    good = torch.tensor([[0.3, -0.3], [0.3, -0.3]], requires_grad=True)
    bad = -good.detach()
    motion = torch.ones(2, 4) * 0.1
    good_loss = target_conditioned_geometric_correction_loss(base, good, target, motion)
    bad_loss = target_conditioned_geometric_correction_loss(base, bad, target, motion)
    assert good_loss < bad_loss
    good_loss.backward()
    assert torch.isfinite(good.grad).all()


def test_confident_correct_base_is_protected_from_harmful_delta():
    base = torch.tensor([[2.0, -2.0]])
    target = torch.tensor([[1.0, 0.0]])
    motion = torch.ones(1, 3)
    safe = target_conditioned_geometric_correction_loss(base, torch.zeros_like(base), target, motion)
    harmful = target_conditioned_geometric_correction_loss(base, torch.tensor([[-1.0, 1.0]]), target, motion)
    assert harmful > safe


def test_action_specific_motion_focuses_correction_gradient_on_supported_action():
    base = torch.zeros(2, 2)
    target = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    delta = torch.zeros_like(base, requires_grad=True)
    motion = torch.zeros(2, 3, 2)
    motion[:, :, 0] = 0.2
    loss = target_conditioned_geometric_correction_loss(base, delta, target, motion)
    loss.backward()
    assert delta.grad[:, 0].abs().mean() > delta.grad[:, 1].abs().mean()


def test_geometric_ranking_loss_rewards_correct_pair_order_and_is_differentiable():
    base = torch.tensor([[0.1], [0.2], [-0.2], [0.0]])
    target = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    motion = torch.ones(4, 3) * 0.1
    good = torch.tensor([[0.3], [-0.3], [0.3], [-0.3]], requires_grad=True)
    bad = -good.detach()
    good_loss = target_conditioned_geometric_ranking_loss(base, good, target, motion)
    bad_loss = target_conditioned_geometric_ranking_loss(base, bad, target, motion)
    assert good_loss < bad_loss
    good_loss.backward()
    assert good.grad is not None and torch.isfinite(good.grad).all()


def test_geometric_ranking_uses_detached_cross_batch_reference_pairs():
    base = torch.tensor([[0.0], [0.1]])
    target = torch.tensor([[1.0], [1.0]])
    delta = torch.zeros_like(base, requires_grad=True)
    reference_logits = torch.tensor([[0.2], [0.3]])
    reference_target = torch.tensor([[0.0], [0.0]])
    loss = target_conditioned_geometric_ranking_loss(
        base,
        delta,
        target,
        torch.ones(2, 2) * 0.1,
        reference_logits=reference_logits,
        reference_target=reference_target,
    )
    assert loss > 0
    loss.backward()
    assert delta.grad is not None and delta.grad.abs().sum() > 0
