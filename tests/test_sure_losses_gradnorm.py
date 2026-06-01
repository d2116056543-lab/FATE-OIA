from __future__ import annotations

import torch

from fate_oia.losses.gradnorm import GradNormBalancer
from fate_oia.losses.sure_losses import compute_sure_losses, make_sure_criterion
from fate_oia.models.sure_oia_model import SUREOIAFeatureModel


def test_sure_losses_and_gradnorm_backward() -> None:
    model = SUREOIAFeatureModel(dim=32, action_dim=4, reason_dim=21, relation_queries=4)
    balancer = GradNormBalancer()
    out = model(torch.randn(2, 16, 32), structured=[{}, {}])
    action = torch.randint(0, 2, (2, 4)).float()
    reason = torch.randint(0, 2, (2, 21)).float()
    losses = compute_sure_losses(out, action, reason, make_sure_criterion("bce"))
    total, stats = balancer(losses)
    total.backward()
    assert stats["min_weight"] == 0.7
    assert 0.7 <= stats["action_weight"] <= 1.5
    assert any(p.grad is not None for p in model.parameters())
