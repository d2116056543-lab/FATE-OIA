from __future__ import annotations

import torch

from fate_oia.losses.acpr_interactflow_losses import exp29_pu_loss, predicate_pu_loss


def test_predicate_pu_and_exp29_pu_use_different_logit_contracts() -> None:
    predicate_logits = torch.randn(2, 15, 48, requires_grad=True)
    predicate_targets = torch.zeros_like(predicate_logits)
    predicate_mask = torch.ones_like(predicate_logits)
    exp29_logits = torch.randn(2, 29, requires_grad=True)
    exp29_targets = torch.zeros_like(exp29_logits)
    exp29_mask = torch.ones_like(exp29_logits)

    pred_loss = predicate_pu_loss(predicate_logits, predicate_targets, predicate_mask)
    exp_loss = exp29_pu_loss(exp29_logits, exp29_targets, exp29_mask)
    pred_loss.backward(retain_graph=True)
    exp_loss.backward()

    assert predicate_logits.grad is not None
    assert exp29_logits.grad is not None
    assert predicate_logits.grad.shape == predicate_logits.shape
    assert exp29_logits.grad.shape == exp29_logits.shape
