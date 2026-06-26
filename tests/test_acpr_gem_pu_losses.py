import torch

from fate_oia.losses import acpr_losses as L


def test_pu_soft_f1_downweights_unlabeled_negatives_by_contradiction():
    logits = torch.tensor([[2.0, 2.0], [1.0, -1.0]])
    target = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    low = torch.zeros_like(target)
    high = torch.ones_like(target)

    loss_low = L.pu_reason_soft_f1_loss(logits, target, low, neg_min_weight=0.2)
    loss_high = L.pu_reason_soft_f1_loss(logits, target, high, neg_min_weight=0.2)

    assert loss_high > loss_low


def test_pu_predicate_alignment_has_no_target_sign_hard_negative_shortcut():
    predicate_probs = torch.rand(3, 5)
    target = torch.zeros(3, 2)
    pos = torch.rand(2, 5)
    neg = torch.rand(2, 5)
    contradiction = torch.zeros_like(target)

    loss = L.pu_predicate_reason_alignment_loss(predicate_probs, target, pos, neg, contradiction)

    assert torch.isfinite(loss)
