from __future__ import annotations

import torch

from fate_oia.models.acpr_oia_model import ACPROIAModel


def test_vista_model_forward_outputs_required_keys():
    model = ACPROIAModel(use_mock_dino=True, vista_enabled=True, threshold_enabled=True)
    out = model(torch.randn(2, 3, 360, 640), epoch=4)
    assert out["action_logits_final_raw"].shape == (2, 4)
    assert out["reason_logits_final_raw"].shape == (2, 21)
    assert out["predicate_attention"].shape[-1] == 3600
    assert out["vista_gate_map"].shape == (2, 3600)
    assert "patch_tokens_by_layer_raw" in out
    assert "patch_tokens_by_layer_adapted" in out

