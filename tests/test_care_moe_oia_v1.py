from __future__ import annotations

import argparse

import torch

from fate_oia.engine.train_care_moe_oia import is_full_goal_run
from fate_oia.models.care_moe_oia_model import CAREMoEOIAModel
from fate_oia.losses.care_moe_losses import evidence_bag_loss


def test_gt_positive_reasons_active_and_top2_experts():
    model = CAREMoEOIAModel()
    model.train()
    tokens = torch.randn(2, 17, 384)
    reason = torch.zeros(2, 21)
    reason[0, [5, 9, 12]] = 1
    reason[1, [6, 11, 14]] = 1
    out = model(tokens, batch={"reason": reason}, structured=[None, None], epoch=3)
    assert out["active_reason_recall_train"].item() == 1.0
    assert int((out["expert_route_mask"].sum(-1) > 2).sum().item()) == 0
    assert set(out["expert_usage"].keys()) == {"object", "lane", "drivable", "traffic_control", "global_context"}


def test_action_cap_and_primary_test_image_only():
    model = CAREMoEOIAModel(action_cap_max=0.04)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 17, 384), batch=None, structured=[{"objects": [{"category": "car"}]}, None], epoch=10)
    assert out["action_delta"].abs().max().item() <= 0.040001
    assert out["diagnostics"]["primary_test_uses_bdd100k_gt"] is False


def test_evidence_bag_loss_has_gradient():
    model = CAREMoEOIAModel()
    model.train()
    reason = torch.zeros(2, 21)
    reason[:, [5, 6]] = 1
    out = model(torch.randn(2, 17, 384), batch={"reason": reason}, structured=[None, None], epoch=3)
    loss = evidence_bag_loss(out, reason, out["active_reason_mask"])
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.expert_router.parameters())


def test_goal_completed_only_for_full_24_epoch_test_run():
    full = argparse.Namespace(
        epochs=24,
        max_train_samples=0,
        max_test_samples=0,
        test_only_evaluation=True,
        best_selection_split="test",
    )
    smoke = argparse.Namespace(
        epochs=1,
        max_train_samples=16,
        max_test_samples=16,
        test_only_evaluation=True,
        best_selection_split="test",
    )
    wrong_split = argparse.Namespace(
        epochs=24,
        max_train_samples=0,
        max_test_samples=0,
        test_only_evaluation=True,
        best_selection_split="val",
    )
    assert is_full_goal_run(full)
    assert not is_full_goal_run(smoke)
    assert not is_full_goal_run(wrong_split)
