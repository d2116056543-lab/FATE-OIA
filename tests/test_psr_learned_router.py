from __future__ import annotations

import torch

from fate_oia.models.psr_oia_router import LearnedPSRRouter, PSRFeatureBuilder


def test_learned_router_has_gradients_and_bounded_alpha():
    torch.manual_seed(2)
    action_a, action_e = torch.randn(7, 4), torch.randn(7, 4)
    reason_a, reason_e = torch.randn(7, 21), torch.randn(7, 21)
    af, rf = PSRFeatureBuilder().build(action_a, reason_a, action_e, reason_e)
    router = LearnedPSRRouter()
    out = router(af, rf, action_a, reason_a, action_e, reason_e)
    assert out.action_logits.shape == (7, 4)
    assert out.reason_logits.shape == (7, 21)
    assert out.alpha_action.min() >= 0 and out.alpha_action.max() <= 1
    assert out.alpha_reason.min() >= 0 and out.alpha_reason.max() <= 1
    loss = out.action_logits.pow(2).mean() + out.reason_logits.pow(2).mean()
    loss.backward()
    assert sum(float(p.grad.abs().sum()) for p in router.parameters() if p.grad is not None) > 0
