from __future__ import annotations

import torch

from fate_oia.models.sure_oia_model import SUREOIAFeatureModel


def test_sure_model_forward_keys_and_shapes() -> None:
    model = SUREOIAFeatureModel(dim=32, action_dim=4, reason_dim=21, relation_queries=6, max_edges_total=16)
    tokens = torch.randn(3, 20, 32)
    structured = [{"objects": [], "lanes": [], "drivable": {}} for _ in range(3)]
    out = model(tokens, structured=structured, image_meta=[{"patch_grid_h": 4, "patch_grid_w": 5}])
    assert out["action_final_logits"].shape == (3, 4)
    assert out["reason_final_logits"].shape == (3, 21)
    assert out["action_gt_scene_upper_logits"].shape == (3, 4)
    assert out["reason_gt_scene_upper_logits"].shape == (3, 21)
    assert out["relation_stats"]["selected_edges"] < out["relation_stats"]["candidate_edges"]
    assert out["memory_gate"].min().item() >= 0.0 and out["memory_gate"].max().item() <= 1.0
