import torch

from fate_oia.datasets.bdd100k_scene_state import scene_state_from_bdd100k_record
from fate_oia.models.ceai_scene_state import SceneStatePrototypeTransformer, masked_scene_state_bce


def test_scene_state_forward_and_masked_loss():
    model = SceneStatePrototypeTransformer(dim=16, scene_proto_count=3, implicit_proto_count=2, num_scene_states=5, num_heads=4)
    tokens = torch.randn(2, 11, 16)
    out = model(tokens)
    assert out["scene_state_tokens"].shape == (2, 3, 16)
    assert out["implicit_prototypes"].shape == (2, 2, 16)
    assert out["scene_state_logits"].shape == (2, 5)
    target = torch.ones(2, 5)
    mask = torch.tensor([[1, 1, 0, 0, 0], [0, 1, 1, 0, 1]], dtype=torch.float32)
    loss = masked_scene_state_bce(out["scene_state_logits"], target, mask)
    assert loss.requires_grad
    assert torch.isfinite(loss)


def test_bdd100k_scene_state_builder_masks_missing():
    rec = {
        "attributes": {"weather": "clear", "timeofday": "daytime"},
        "frames": [{"labels": [
            {"category": "traffic light", "box2d": {"x1": 1, "y1": 2, "x2": 10, "y2": 11}},
            {"category": "lane/single white", "poly2d": [{"vertices": [[0, 0], [10, 0], [10, 2]]}]},
            {"category": "car", "box2d": {"x1": 11, "y1": 2, "x2": 20, "y2": 11}},
        ]}],
        "drivable_map": "dummy.png",
    }
    state = scene_state_from_bdd100k_record(rec)
    assert len(state["scene_state_target"]) == len(state["scene_state_mask"])
    assert sum(state["scene_state_mask"]) > 0
    assert state["counts"]["traffic_control_count"] == 1
    assert state["counts"]["lane_poly_count"] >= 1
