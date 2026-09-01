import torch

from fate_oia.losses.tida_losses import (
    reason_local_deletion_credit_loss,
    reason_local_order_credit_loss,
)
from fate_oia.losses.tida_loss_registry import TIDALossRegistry


def _inputs():
    target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    contradiction = torch.tensor([[0.0, 0.9], [0.9, 0.0]])
    motion = torch.ones(2, 1, 2)
    return target, contradiction, motion


def test_reason_local_order_credit_prefers_label_correct_ordered_evidence():
    target, contradiction, motion = _inputs()
    helpful = torch.tensor([[0.03, -0.03], [-0.03, 0.03]])
    shuffled = torch.zeros_like(helpful)
    harmful = -helpful

    good = reason_local_order_credit_loss(
        helpful, shuffled, target, motion, contradiction
    )
    bad = reason_local_order_credit_loss(
        harmful, shuffled, target, motion, contradiction
    )

    assert good < bad
    assert torch.isfinite(good)


def test_reason_local_deletion_credit_prefers_selected_over_control_effect():
    target, contradiction, motion = _inputs()
    candidate = torch.tensor([[0.03, -0.03], [-0.03, 0.03]])
    selected_good = torch.zeros_like(candidate)
    control_good = candidate * 0.9
    selected_bad = candidate * 0.9
    control_bad = torch.zeros_like(candidate)

    good = reason_local_deletion_credit_loss(
        candidate, selected_good, control_good, target, motion, contradiction
    )
    bad = reason_local_deletion_credit_loss(
        candidate, selected_bad, control_bad, target, motion, contradiction
    )

    assert good < bad
    assert torch.isfinite(good)


def test_site_loss_weights_are_registered_for_real_training():
    registry = TIDALossRegistry({
        "reason_local_order": 0.05,
        "reason_local_deletion": 0.05,
        "logit_flow_action_rank": 0.20,
        "logit_flow_reason_rank": 0.08,
    })
    assert registry.weights["reason_local_order"] == 0.05
    assert registry.weights["reason_local_deletion"] == 0.05
    assert registry.weights["logit_flow_action_rank"] == 0.20
    assert registry.weights["logit_flow_reason_rank"] == 0.08
