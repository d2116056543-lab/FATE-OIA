from __future__ import annotations

import torch

from fate_oia.models.acpr_oia_model import ACPROIAModel


def test_zero_gate_vista_model_matches_disabled_model_on_mock_dino():
    torch.manual_seed(7)
    base = ACPROIAModel(use_mock_dino=True, vista_enabled=False, threshold_enabled=False)
    torch.manual_seed(7)
    vista = ACPROIAModel(use_mock_dino=True, vista_enabled=True, threshold_enabled=False)
    vista.load_state_dict(base.state_dict(), strict=False)
    x = torch.randn(1, 3, 360, 640)
    out_base = base(x, epoch=0)
    out_vista = vista(x, epoch=0)
    assert torch.allclose(out_base["action_logits_base"], out_vista["action_logits_base"], atol=1e-6)
    assert torch.allclose(out_base["reason_logits_base"], out_vista["reason_logits_base"], atol=1e-6)
    assert torch.allclose(out_base["predicate_probs"], out_vista["predicate_probs"], atol=1e-6)

