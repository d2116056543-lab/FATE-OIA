import torch

from fate_oia.losses.acpr_losses import partial_label_reason_loss, reason_soft_f1_loss


def test_acpr_partial_label_loss_grad():
    logits = torch.randn(3, 21, requires_grad=True)
    target = torch.randint(0, 2, (3, 21)).float()
    loss = partial_label_reason_loss(logits, target) + reason_soft_f1_loss(logits, target)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
