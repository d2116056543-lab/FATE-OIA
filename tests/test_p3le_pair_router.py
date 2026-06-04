from pathlib import Path

import torch

from fate_oia.models.p3le_router import ParetoSafeRouter


def test_router_anchors_final_logits_to_base_paths():
    src = Path("fate_oia/models/p3le_router.py").read_text(encoding="utf-8")
    assert "base_action" in src
    assert "base_reason" in src
    assert "final_action = base_action" in src
    assert "final_reason = base_reason" in src

    router = ParetoSafeRouter(dim=16, action_dim=4, reason_dim=21)
    base_action = torch.randn(2, 4)
    base_reason = torch.randn(2, 21)
    out = router(
        base_action,
        base_reason,
        torch.randn(2, 4),
        torch.randn(2, 21),
        torch.randn(2, 4),
        torch.randn(2, 21),
        torch.randn(2, 4),
        torch.rand(2, 21),
        torch.randn(2, 16),
        0.0,
        0.0,
    )
    assert torch.allclose(out["final_action_logits"], base_action)
    assert torch.allclose(out["final_reason_logits"], base_reason)

