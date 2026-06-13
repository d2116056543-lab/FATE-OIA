from __future__ import annotations

import torch
from fate_oia.models.eagle_pu_model import EaglePUModel

def test_eagle_pu_forward_contract_with_mocked_dino():
    model = EaglePUModel(dim=32, dino_dim=32, action_dim=4, reason_dim=21, selected_layers=(3, 7, 11), freeze_dino=True, pretrained_weights="", use_mock_dino=True)
    out = model(torch.randn(2, 3, 360, 640), epoch=5)
    required = ["action_logits_final_raw", "reason_logits_final_raw", "action_logits_final_calibrated", "reason_logits_final_calibrated", "action_logits_direct", "reason_logits_direct", "action_visual_logits", "action_reason_logits", "prototype_reason_delta", "reason_graph_delta", "action_set_logits", "action_set_probs", "state_group_logits", "state_layer_weights", "label_attention", "edge_weights", "reason_to_set_logits"]
    for key in required:
        assert key in out, key
    assert out["action_logits_final_raw"].shape == (2, 4)
    assert out["reason_logits_final_raw"].shape == (2, 21)
    assert out["action_set_logits"].shape == (2, 16)
    assert out["label_attention"].shape == (2, 25, 3600)
    assert out["edge_weights"].shape == (2, 41, 41)
    assert out["reason_to_set_logits"].shape == (2, 21, 16)
    assert torch.allclose(out["action_logits_final_raw"], out["action_logits_direct"], atol=1e-6)
    assert out["prototype_reason_delta"].abs().max() <= 0.120001
    assert out["reason_graph_delta"].abs().max() <= 0.080001
