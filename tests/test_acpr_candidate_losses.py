import torch

from fate_oia.losses.acpr_candidate_losses import (
    action_candidate_nonregression_loss,
    all_candidate_probe_loss,
)


def test_candidate_nonregression_loss_zero_when_equal_and_positive_when_worse():
    labels = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    fallback = torch.tensor([[4.0, -4.0, 4.0, -4.0]])
    same = fallback.clone().requires_grad_(True)
    worse = -fallback.clone().requires_grad_(True)

    assert action_candidate_nonregression_loss(same, fallback, labels).item() == 0.0
    loss = action_candidate_nonregression_loss(worse, fallback, labels)
    assert loss.item() > 0.0


def test_all_candidate_probe_loss_backprops_to_candidate_tensor():
    labels = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
    fallback = torch.zeros(2, 4)
    blend = torch.zeros(2, 4, requires_grad=True)
    candidates = {
        "visual": torch.zeros(2, 4, requires_grad=True),
        "reason": torch.zeros(2, 4, requires_grad=True),
        "blend": blend,
        "predicate": torch.zeros(2, 4, requires_grad=True),
        "blend_predicate": torch.zeros(2, 4, requires_grad=True),
    }
    total, parts = all_candidate_probe_loss(candidates, fallback, labels, tuple(candidates))
    total.backward()
    assert blend.grad is not None
    assert float(blend.grad.abs().sum()) > 0.0
    assert "loss_candidate_blend" in parts
    assert "nonreg_candidate_blend" in parts

