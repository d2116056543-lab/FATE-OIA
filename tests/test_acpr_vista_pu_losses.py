from __future__ import annotations

import torch

from fate_oia.losses.acpr_losses import predicate_reason_alignment_loss, reason_soft_f1_loss


def test_reason_soft_f1_uses_pu_negative_weights():
    logits = torch.tensor([[3.0, 3.0], [3.0, -3.0]], requires_grad=True)
    target = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    low_contradiction = torch.zeros_like(target)
    high_contradiction = torch.ones_like(target)
    low_loss = reason_soft_f1_loss(logits, target, contradiction_scores=low_contradiction, neg_min_weight=0.2)
    high_loss = reason_soft_f1_loss(logits, target, contradiction_scores=high_contradiction, neg_min_weight=0.2)
    assert high_loss > low_loss
    low_loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_predicate_reason_alignment_is_pu_consistent():
    predicate_probs = torch.tensor([[0.9, 0.1]], requires_grad=True)
    reason_targets = torch.tensor([[1.0, 0.0]])
    grammar_pos = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    grammar_neg = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    contradiction = torch.tensor([[0.0, 0.0]])
    weak_neg_loss = predicate_reason_alignment_loss(
        predicate_probs,
        reason_targets,
        grammar_pos,
        grammar_neg,
        contradiction_scores=contradiction,
        neg_min_weight=0.2,
    )
    strong_neg_loss = predicate_reason_alignment_loss(
        predicate_probs,
        reason_targets,
        grammar_pos,
        grammar_neg,
        contradiction_scores=torch.ones_like(contradiction),
        neg_min_weight=0.2,
    )
    assert strong_neg_loss > weak_neg_loss
    weak_neg_loss.backward()
    assert torch.isfinite(predicate_probs.grad).all()
