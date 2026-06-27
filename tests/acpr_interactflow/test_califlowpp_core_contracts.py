from __future__ import annotations

import torch

from fate_oia.acpr_interactflow.decision_ledger import DecisionLedgerHead
from fate_oia.acpr_interactflow.exp29_head import Exp29Head
from fate_oia.acpr_interactflow.timing import StepTimer
from fate_oia.losses.acpr_interactflow_losses import (
    compute_interactflow_losses,
    non_degradation_soft_kl_hinge_loss,
    predicate_pu_loss,
)


def test_decision_ledger_is_three_action_exact_additive() -> None:
    ledger = DecisionLedgerHead(dim=16, num_actions=3)
    visual = torch.randn(2, 16)
    motion = torch.randn(2, 16)
    predicate = torch.randn(2, 16)
    factors = torch.randn(2, 5, 16)
    flow = torch.randn(2, 5, 3)
    out = ledger(visual, motion, predicate, factors, flow)

    reconstructed = out.global_logits + out.gated_state_contributions.sum(1) + out.calibration_delta
    assert out.final_logits.shape == (2, 3)
    assert out.raw_state_contributions.shape == (2, 5, 3)
    assert torch.allclose(out.final_logits, reconstructed)
    assert float(out.identity_error) < 1e-6


def test_exp29_head_uses_ledger_contributions_and_calibrated_logits() -> None:
    head = Exp29Head(dim=16, exp_dim=4)
    factors = torch.randn(2, 5, 16)
    predicates = torch.randn(2, 7, 16)
    contrib = torch.randn(2, 5, 3)
    global_hidden = torch.randn(2, 16)
    action_logits = torch.randn(2, 3)
    out = head(
        factor_tokens_lag=factors,
        predicate_tokens_summary=predicates,
        gated_state_contributions=contrib,
        global_decision_hidden=global_hidden,
        action_logits=action_logits,
    )

    assert out.logits_raw.shape == (2, 4)
    assert out.logits_calibrated.shape == (2, 4)
    assert out.cluster_attention_to_factors.shape == (2, 4, 5)
    assert out.cluster_reliability.shape == (4,)
    assert out.cluster_to_state_prior.shape == (4, 5)


def test_soft_kl_safety_uses_soft_targets_not_hard_ce() -> None:
    final = torch.tensor([[2.0, 0.0, -1.0]], requires_grad=True)
    base = torch.tensor([[1.0, 0.5, -0.5]])
    soft = torch.tensor([[0.6, 0.4, 0.0]])
    loss = non_degradation_soft_kl_hinge_loss(final, base, soft, margin=0.01)
    loss.backward()
    assert torch.isfinite(loss)
    assert final.grad is not None


def test_predicate_pu_loss_is_separate_from_exp29_shape() -> None:
    logits = torch.randn(2, 15, 48)
    targets = torch.zeros(2, 15, 48)
    mask = torch.ones(2, 15, 48)
    loss = predicate_pu_loss(logits, targets, mask)
    assert torch.isfinite(loss)


def test_step_timer_reports_required_sections() -> None:
    timer = StepTimer()
    with timer.section("data_gap"):
        pass
    with timer.section("visual_dino"):
        pass
    row = timer.summary(reset=False)
    assert "data_gap_time" in row
    assert "visual_dino_time" in row
    assert "total_profiled_time" in row
