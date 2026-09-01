import torch

from fate_oia.losses.tida_losses import (
    reason_geometric_supervision_weight,
    reason_partial_asl_loss,
    reason_pu_weight,
)


def test_reason_pu_weight_preserves_positives_and_distinguishes_unknown_negatives():
    target = torch.tensor([[1.0, 0.0, 0.0]])
    contradiction = torch.tensor([[0.0, 0.0, 1.0]], requires_grad=True)

    weight = reason_pu_weight(target, contradiction, negative_floor=0.2)

    torch.testing.assert_close(weight, torch.tensor([[1.0, 0.2, 1.0]]))
    assert not weight.requires_grad


def test_reliable_reason_negative_receives_stronger_partial_asl_gradient():
    logits = torch.tensor([[0.0], [1.0], [1.0]], requires_grad=True)
    target = torch.tensor([[1.0], [0.0], [0.0]])
    contradiction = torch.tensor([[0.0], [0.0], [1.0]])

    loss = reason_partial_asl_loss(logits, target, contradiction)
    loss.backward()

    assert logits.grad[2].abs().item() > logits.grad[1].abs().item()


def test_geometric_reason_supervision_excludes_unknown_negatives():
    target = torch.tensor([[1.0, 0.0, 0.0]])
    contradiction = torch.tensor([[0.0, 0.4, 0.8]], requires_grad=True)

    weight = reason_geometric_supervision_weight(target, contradiction, threshold=0.55)

    expected_certified = (0.8 - 0.55) / (1.0 - 0.55)
    torch.testing.assert_close(weight, torch.tensor([[1.0, 0.0, expected_certified]]))
    assert not weight.requires_grad
