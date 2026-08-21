import torch

from fate_oia.losses.tida_losses import action_smooth_ap_loss


def test_action_smooth_ap_uses_detached_reference_from_same_accumulation_window():
    current = torch.tensor([[0.0]], requires_grad=True)
    target = torch.tensor([[1.0]])
    reference = torch.tensor([[0.5]], requires_grad=True)
    reference_target = torch.tensor([[0.0]])

    loss = action_smooth_ap_loss(current, target, reference, reference_target)
    loss.backward()

    assert loss.item() > 0.0
    assert current.grad is not None and current.grad.abs().sum().item() > 0.0
    assert reference.grad is None
