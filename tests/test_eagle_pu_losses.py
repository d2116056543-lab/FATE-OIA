from __future__ import annotations

import torch
from fate_oia.losses.eagle_pu_losses import action_direct_asl_loss, action_set_ce_loss, cardinality_loss, calibration_regularizer, evidence_margin_loss, positive_unlabeled_reason_loss, reason_direct_asl_loss, reason_soft_f1_loss

def test_eagle_pu_loss_terms_are_finite_and_named():
    action_logits = torch.randn(3, 4, requires_grad=True)
    reason_logits = torch.randn(3, 21, requires_grad=True)
    action_targets = torch.tensor([[1,0,0,0],[1,0,1,0],[0,1,0,1]], dtype=torch.float32)
    reason_targets = torch.zeros(3, 21)
    reason_targets[:, [2, 5, 12]] = 1
    reliability = torch.sigmoid(torch.randn(3, 21))
    action_set_logits = torch.randn(3, 16, requires_grad=True)
    losses = [action_direct_asl_loss(action_logits, action_targets), reason_direct_asl_loss(reason_logits, reason_targets), reason_soft_f1_loss(reason_logits, reason_targets), positive_unlabeled_reason_loss(reason_logits, reason_targets, reliability), action_set_ce_loss(action_set_logits, torch.tensor([1, 5, 10])), cardinality_loss(torch.randn(3, 5, requires_grad=True), action_targets), calibration_regularizer(torch.ones(25), torch.zeros(25)), evidence_margin_loss(torch.tensor([0.4]), torch.tensor([0.3]), active=True)]
    total = sum(losses)
    assert torch.isfinite(total)
    total.backward()
    assert action_logits.grad is not None and torch.isfinite(action_logits.grad).all()
    assert reason_logits.grad is not None and torch.isfinite(reason_logits.grad).all()
