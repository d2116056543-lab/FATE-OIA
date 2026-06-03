from __future__ import annotations

import argparse

import torch

from fate_oia.losses.psr_train_losses import psr_train_loss
from fate_oia.models.psr_train_oia_model import PSRTrainOIAFeatureModel


def _args():
    return argparse.Namespace(
        asl_gamma_pos=0.0,
        asl_gamma_neg=4.0,
        asl_clip=0.05,
        pareto_margin_action=0.005,
        pareto_margin_reason=0.005,
        loss_final_action=1.0,
        loss_final_reason=1.0,
        loss_a_action=0.4,
        loss_e_reason=0.4,
        loss_a_reason=0.05,
        loss_e_action=0.01,
        loss_calibration_reason=0.05,
        loss_pareto=0.2,
        loss_gradient_budget=0.001,
    )


def test_psr_train_model_warmup_and_router_shapes():
    torch.manual_seed(11)
    model = PSRTrainOIAFeatureModel(dim=32, action_dim=4, reason_dim=21, dropout=0.0)
    tokens = torch.randn(3, 20, 32)
    warm = model(tokens, epoch=0)
    assert warm["final_action_logits"].shape == (3, 4)
    assert warm["final_reason_logits"].shape == (3, 21)
    assert torch.allclose(warm["final_action_logits"], warm["a_action_logits"])
    assert torch.allclose(warm["final_reason_logits"], warm["e_reason_logits"])
    routed = model(tokens, epoch=8)
    assert routed["reason_router_gate"].shape == (3, 21)
    assert routed["action_router_gate"].shape == (3, 4)
    assert routed["router_scale"].item() > 0
    assert (routed["reason_router_gate"] >= 0).all()
    assert (routed["reason_router_gate"] <= 1).all()


def test_psr_train_loss_gives_router_and_calibration_gradients():
    torch.manual_seed(12)
    model = PSRTrainOIAFeatureModel(dim=32, action_dim=4, reason_dim=21, dropout=0.0)
    tokens = torch.randn(2, 24, 32)
    action = torch.randint(0, 2, (2, 4)).float()
    reason = torch.randint(0, 2, (2, 21)).float()
    out = model(tokens, epoch=8)
    loss, parts = psr_train_loss(out, action, reason, _args())
    loss.backward()
    assert parts["pareto_action_loss"] >= 0
    assert parts["pareto_reason_loss"] >= 0
    assert model.action_router[-1].weight.grad is not None
    assert model.action_router[-1].weight.grad.abs().sum().item() > 0
    assert model.reason_router[-1].weight.grad is not None
    assert model.reason_router[-1].weight.grad.abs().sum().item() > 0
    assert model.reason_calibration_bias.grad is not None
    assert model.reason_calibration_bias.grad.abs().sum().item() > 0
