from __future__ import annotations

import torch

from fate_oia.models.acpr_oia_model import ACPROIAModel


def test_action_reason_predicate_losses_reach_vista_gate():
    model = ACPROIAModel(use_mock_dino=True, vista_enabled=True, threshold_enabled=False)
    x = torch.randn(1, 3, 360, 640)
    out = model(x, epoch=0)
    loss = out["action_logits_base"].mean() + out["reason_logits_base"].mean() + out["predicate_logits"].mean()
    loss.backward()
    up_grad = sum(
        float(block.up.weight.grad.abs().sum())
        for block in model.visual_adapter.blocks
        if block.up.weight.grad is not None
    )
    assert up_grad > 0
