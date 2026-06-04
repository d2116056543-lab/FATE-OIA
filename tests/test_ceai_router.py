import torch

from fate_oia.models.ceai_router import ParetoSafeRouter, guarded_action_metrics


def test_router_anchor_caps_and_guard():
    router = ParetoSafeRouter(action_dim=4, reason_dim=21, action_cap=0.04, reason_cap=0.12)
    base_a = torch.randn(2, 4)
    base_r = torch.randn(2, 21)
    act = base_a + torch.randn(2, 4) * 5
    reason = base_r + torch.randn(2, 21) * 5
    pair = torch.randn(2, 4, 21)
    q = torch.ones(2, 4, 21)
    out_off = router(base_a, base_r, act, reason, pair, q, readiness={"r2a_active": False})
    assert torch.allclose(out_off["final_action_logits"], base_a, atol=1e-6)
    out_on = router(base_a, base_r, act, reason, pair, q, readiness={"r2a_active": True})
    assert (out_on["final_action_logits"] - base_a).abs().max() <= 0.040001
    assert (out_on["final_reason_logits"] - base_r).abs().max() <= 0.180001
    guarded = guarded_action_metrics(base_score=0.7, final_score=0.68, tolerance=0.006)
    assert guarded["guarded_action_branch"] == "base"
