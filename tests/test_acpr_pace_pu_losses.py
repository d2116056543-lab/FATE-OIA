import torch

from fate_oia.losses.acpr_losses import (
    partial_label_reason_loss,
    pu_predicate_reason_alignment_loss,
    pu_reason_soft_f1_loss,
)


def test_partial_label_loss_keeps_positive_full_and_low_contradiction_unlabeled_small():
    logits = torch.tensor([[0.0, 0.0]], requires_grad=True)
    target = torch.tensor([[1.0, 0.0]])
    contradiction = torch.tensor([[0.0, 0.0]])
    loss = partial_label_reason_loss(logits, target, contradiction, neg_min_weight=0.2)
    loss.backward()
    assert logits.grad is not None
    assert abs(logits.grad[0, 0]) > abs(logits.grad[0, 1])


def test_pu_soft_f1_and_alignment_are_finite():
    logits = torch.randn(4, 3)
    target = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 0], [1, 0, 1]], dtype=torch.float32)
    contradiction = torch.rand(4, 3)
    loss = pu_reason_soft_f1_loss(logits, target, contradiction)
    assert torch.isfinite(loss)
    pred = torch.sigmoid(torch.randn(4, 5))
    pos = torch.rand(3, 5)
    neg = torch.rand(3, 5)
    align = pu_predicate_reason_alignment_loss(pred, target, pos, neg, contradiction)
    assert torch.isfinite(align)
