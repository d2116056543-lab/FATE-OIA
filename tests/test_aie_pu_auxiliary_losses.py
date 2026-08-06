import torch

from fate_oia.losses.aie_losses import predicate_reason_alignment_pu_loss, soft_f1_loss


def test_pu_weights_reduce_unconfirmed_zero_penalty_in_auxiliary_losses():
    logits = torch.tensor([[3.0, 3.0]])
    target = torch.zeros_like(logits)
    full = soft_f1_loss(logits, target, torch.ones_like(target))
    censored = soft_f1_loss(logits, target, torch.full_like(target, 0.25))
    assert censored < full

    predicate_probs = torch.tensor([[0.9, 0.1]])
    positive = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    contradictory = torch.zeros_like(positive)
    full_align = predicate_reason_alignment_pu_loss(
        predicate_probs, target, positive, contradictory, torch.ones_like(target)
    )
    censored_align = predicate_reason_alignment_pu_loss(
        predicate_probs, target, positive, contradictory, torch.full_like(target, 0.25)
    )
    assert censored_align < full_align
