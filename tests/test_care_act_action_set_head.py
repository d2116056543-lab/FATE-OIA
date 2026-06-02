from __future__ import annotations

import torch

from fate_oia.models.care_action_set_head import ActionSetConsistencyHead


def test_action_set_head_has_prototypes_and_gradients():
    head = ActionSetConsistencyHead(dim=384, action_dim=4)
    base = torch.randn(2, 4)
    context = torch.randn(2, 4, 384)
    uncertainty = torch.rand(2, 4)
    out = head(base, context, uncertainty)
    assert out["action_set_logits"].shape == (2, 4)
    assert out["action_set_delta"].abs().max().item() <= 0.060001
    assert head.action_set_prototypes.shape[1] == 4
    loss = out["action_set_logits"].sum()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())
