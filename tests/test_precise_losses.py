import torch

from fate_oia.losses.precise_losses import (
    _soft_cldice_loss,
    _symmetric_soft_distance_transform_loss,
    semantic_reliability_weight,
)


def test_semantic_reliability_uses_detached_paired_view_consistency():
    direct = torch.zeros(1, 2, requires_grad=True)
    semantic = torch.zeros(1, 2, requires_grad=True)
    attention = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], requires_grad=True)
    evidence_reliability = torch.ones(1, 2, requires_grad=True)
    view = torch.tensor([0.0, 1.0], requires_grad=True)
    weight = semantic_reliability_weight(direct, semantic, attention, evidence_reliability, view)
    assert weight[0, 0].item() == 0.25
    assert weight[0, 1].item() == 1.0
    assert weight.requires_grad is False


def test_curve_losses_reward_matching_thin_structures_and_are_differentiable():
    target = torch.zeros(1, 1, 15, 15)
    target[:, :, 2:13, 7] = 1.0
    matching = target.clone().requires_grad_(True)
    shifted = torch.zeros_like(target)
    shifted[:, :, 2:13, 11] = 1.0
    matching_loss = _soft_cldice_loss(matching, target) + _symmetric_soft_distance_transform_loss(matching, target)
    shifted_loss = _soft_cldice_loss(shifted, target) + _symmetric_soft_distance_transform_loss(shifted, target)
    assert matching_loss < shifted_loss
    matching_loss.backward()
    assert matching.grad is not None and torch.isfinite(matching.grad).all()
