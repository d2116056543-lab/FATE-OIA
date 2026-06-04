import torch

from fate_oia.models.p3le_router import ParetoSafeRouter


def test_router_anchors_action_with_bounded_residual():
    router = ParetoSafeRouter(dim=32, action_dim=4, reason_dim=21, action_residual_cap=0.04)
    a_action = torch.randn(2, 4)
    out = router(
        a_action,
        torch.randn(2, 21),
        torch.randn(2, 4),
        torch.randn(2, 21),
        torch.randn(2, 4),
        torch.rand(2, 21),
        torch.randn(2, 32),
        router_scale=1.0,
    )
    delta = (out["final_action_logits"] - a_action).abs().max()
    assert float(delta) <= 0.041
    assert tuple(out["final_reason_logits"].shape) == (2, 21)
