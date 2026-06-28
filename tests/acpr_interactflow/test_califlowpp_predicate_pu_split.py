from __future__ import annotations

import torch

from fate_oia.losses.acpr_interactflow_losses import (
    exp29_pu_loss,
    predicate_pu_loss,
    predicate_structural_weak_loss,
)


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



def test_predicate_pu_without_targets_does_not_train_everything_as_negative() -> None:
    logits = torch.zeros(2, 15, 48, requires_grad=True)

    loss = predicate_pu_loss(logits)
    loss.backward()

    assert torch.isfinite(loss)
    assert loss.item() == 0.0
    assert logits.grad is not None
    assert torch.allclose(logits.grad, torch.zeros_like(logits.grad))


def test_predicate_structural_weak_pushes_low_logits_up_without_all_negative_pu() -> None:
    logits = torch.full((2, 15, 48), -4.0, requires_grad=True)
    probs = torch.sigmoid(logits)

    loss = predicate_structural_weak_loss(logits, probs)
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert logits.grad.mean() < 0
