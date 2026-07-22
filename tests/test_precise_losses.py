import torch

from fate_oia.losses.precise_losses import semantic_reliability_weight


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
