import torch
import torch.nn.functional as F

from fate_oia.losses.aie_losses import proper_binary_calibration_loss


def test_proper_calibration_matches_bce_for_fully_observed_action_labels():
    logits = torch.tensor([[1.2, -0.7], [-0.2, 0.9]], requires_grad=True)
    target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    actual = proper_binary_calibration_loss(logits, target)
    expected = F.binary_cross_entropy_with_logits(logits, target)

    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_proper_calibration_preserves_positive_weight_and_downweights_pu_zeros():
    logits = torch.zeros((1, 3), requires_grad=True)
    target = torch.tensor([[1.0, 0.0, 0.0]])
    negative_weight = torch.tensor([[1.0, 0.25, 0.75]])

    loss = proper_binary_calibration_loss(logits, target, negative_weight)
    expected_raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    expected = (expected_raw * negative_weight).sum() / negative_weight.sum()

    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert logits.grad[0, 0].abs() > logits.grad[0, 1].abs()
    assert torch.isfinite(logits.grad).all()
