import torch

from fate_oia.losses.aie_losses import classwise_pu_dr_loss


def test_classwise_dr_pushes_positive_above_negative_for_each_reason():
    logits = torch.tensor([[0.0], [0.2], [-0.1]], requires_grad=True)
    target = torch.tensor([[1.0], [0.0], [0.0]])
    negative_weight = torch.ones_like(target)
    loss = classwise_pu_dr_loss(logits, target, negative_weight, gamma_pair=2.0, gamma_negative=1.0)
    loss.backward()
    assert logits.grad[0, 0] < 0
    assert logits.grad[1, 0] > 0
    assert logits.grad[2, 0] > 0


def test_classwise_dr_downweights_uncertain_unlabeled_negatives():
    logits_full = torch.tensor([[0.0], [1.0]], requires_grad=True)
    logits_weak = logits_full.detach().clone().requires_grad_(True)
    target = torch.tensor([[1.0], [0.0]])
    classwise_pu_dr_loss(logits_full, target, torch.ones_like(target)).backward()
    classwise_pu_dr_loss(logits_weak, target, torch.tensor([[1.0], [0.1]])).backward()
    assert logits_weak.grad[1, 0].abs() < logits_full.grad[1, 0].abs()
