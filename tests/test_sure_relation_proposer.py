from __future__ import annotations

import torch

from fate_oia.models.sure_relation_proposer import SURERelationProposer


def test_relation_proposer_outputs_fair_and_gt_tokens() -> None:
    torch.manual_seed(1)
    proposer = SURERelationProposer(dim=16, relation_queries=4)
    tokens = torch.randn(2, 12, 16)
    structured = [
        {"objects": [{"box2d": {"x1": 0, "y1": 0, "x2": 640, "y2": 360}}], "lanes": [], "drivable": {}},
        {"objects": [], "lanes": [], "drivable": {}},
    ]
    fair = proposer(tokens, structured=structured, image_meta=[{"patch_grid_h": 3, "patch_grid_w": 4}], use_gt_scene_upper=False)
    upper = proposer(tokens, structured=structured, image_meta=[{"patch_grid_h": 3, "patch_grid_w": 4}], use_gt_scene_upper=True)
    assert fair["relation_tokens"].shape == (2, 4, 16)
    assert upper["relation_tokens"].shape == (2, 4, 16)
    assert not torch.allclose(fair["relation_tokens"], upper["relation_tokens"])
    assert fair["stats"]["candidate_relations"] == 4
