import torch

from fate_oia.losses.aie_losses import evidence_censored_reason_asl_loss, reason_negative_weight


def test_reason_negative_weight_preserves_positive_and_bounds_zero():
    target = torch.tensor([[1.0, 0.0, 0.0]])
    counter = torch.tensor([[0.2, 0.0, 1.0]], requires_grad=True)
    weight = reason_negative_weight(target, counter)
    torch.testing.assert_close(weight, torch.tensor([[1.0, 0.25, 1.0]]))
    loss = evidence_censored_reason_asl_loss(torch.zeros_like(target, requires_grad=True), target, counter)
    loss.backward()
    assert counter.grad is None


