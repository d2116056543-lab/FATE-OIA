from __future__ import annotations

import argparse

import torch

from fate_oia.losses.care_act_losses import care_act_training_loss
from fate_oia.models.care_act_model import CAREActOIAModel


def test_care_act_loss_reaches_action_evidence_and_action_set():
    args = argparse.Namespace(
        asl_gamma_pos=0.0,
        asl_gamma_neg=4.0,
        asl_clip=0.05,
        loss_base_action=0.5,
        loss_base_reason=0.2,
        loss_action_visual=0.05,
        loss_r2a_gt=0.1,
        loss_action_agree=0.01,
        loss_action_evidence=0.5,
        loss_action_set=0.25,
        loss_evidence_bag=0.1,
        loss_reason_delta_reg=0.001,
        loss_action_delta_reg=0.001,
    )
    model = CAREActOIAModel()
    model.train()
    action = torch.randint(0, 2, (2, 4)).float()
    reason = torch.randint(0, 2, (2, 21)).float()
    out = model(torch.randn(2, 45 * 80 + 1, 384), batch={"reason": reason}, structured=[None, None], epoch=8)
    loss, parts = care_act_training_loss(out, action, reason, args)
    loss.backward()
    assert parts["action_evidence_loss"] > 0
    assert parts["action_set_loss"] > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.action_evidence_bank.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.action_set_head.parameters())
