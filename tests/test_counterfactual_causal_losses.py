from __future__ import annotations

import torch

from fate_oia.losses.counterfactual_causal_losses import counterfactual_direct_effect_loss, counterfactual_replacement_loss
from fate_oia.losses.tail_causal_ranking import hard_logit_pairwise_ranking_loss


def test_counterfactual_loss_nonzero_when_valid():
    gt = torch.zeros(3, 21)
    gt[:, 5] = 1
    factual = torch.zeros(3, 21)
    deleted = torch.zeros(3, 21) - 0.2
    cf = {"reason_logits_factual": factual, "reason_logits_target_deleted": deleted}
    loss, stats = counterfactual_direct_effect_loss(cf, gt, None, (5,))
    assert loss.item() > 0
    assert stats["cf_valid_count"] == 3


def test_replacement_loss_penalizes_false_activation():
    gt = torch.zeros(2, 21)
    cf = {"reason_logits_replaced": torch.ones(2, 21)}
    loss, stats = counterfactual_replacement_loss(cf, gt)
    assert loss.item() > 0
    assert stats["replacement_false_activation_rate"] > 0


def test_tail_ranking_gradients():
    logits = torch.randn(4, 21, requires_grad=True)
    targets = torch.zeros(4, 21)
    targets[0, 5] = 1
    loss = hard_logit_pairwise_ranking_loss(logits, targets, (5,))
    loss.backward()
    assert logits.grad is not None

