from __future__ import annotations

import torch

from fate_oia.engine.train_fate_oia import action_branch_losses
from fate_oia.models.fate_oia_model import FATEOIAFeatureModel


def test_fate_oia_model_outputs_visual_reason_and_fused_action_logits():
    torch.manual_seed(0)
    model = FATEOIAFeatureModel(dim=16, action_dim=4, reason_dim=21, use_label_query=True)
    out = model(torch.randn(2, 12, 16))
    assert out["action_visual_logits"].shape == (2, 4)
    assert out["action_reason_logits"].shape == (2, 4)
    assert out["action_fused_logits"].shape == (2, 4)
    assert out["action_logits"].shape == (2, 4)
    assert out["fusion_gate"].shape == (2, 4)
    assert torch.all(out["fusion_gate"] >= 0)
    assert torch.all(out["fusion_gate"] <= 1)


def test_reason_branch_changes_fused_action_logits():
    torch.manual_seed(1)
    model = FATEOIAFeatureModel(dim=8, action_dim=4, reason_dim=21, use_label_query=True)
    tokens = torch.randn(1, 7, 8)
    out_a = model(tokens)
    with torch.no_grad():
        model.reason_to_action.net[-1].bias.add_(2.0)
    out_b = model(tokens)
    assert not torch.allclose(out_a["action_reason_logits"], out_b["action_reason_logits"])
    assert not torch.allclose(out_a["action_fused_logits"], out_b["action_fused_logits"])


def test_r2a_gt_loss_has_gradient_to_reason_branch():
    torch.manual_seed(2)
    model = FATEOIAFeatureModel(dim=8, action_dim=4, reason_dim=21, use_label_query=True)
    out = model(torch.randn(2, 9, 8))
    action_target = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=torch.float32)
    losses = action_branch_losses(out, action_target, loss_r2a_gt=0.5, loss_action_agree=0.1)
    losses["action_total"].backward()
    grad = sum(float(p.grad.abs().sum()) for p in model.reason_to_action.parameters() if p.grad is not None)
    assert grad > 0
