from fate_oia.utils.acpr_action_semantic_maps import load_action_semantic_maps


def test_action_semantic_maps_shapes_and_names():
    maps = load_action_semantic_maps("configs/acpr_reason_predicate_grammar.yaml", "configs/acpr_scene_predicates.yaml")
    assert maps.action_reason_mask.shape == (4, 21)
    assert maps.action_predicate_mask.shape[0] == 4
    assert maps.forbidden_r2a_mask.shape == (4, 21)
    assert all(not n.lower().startswith("reason_") for n in maps.reason_names)
    assert maps.action_reason_mask[2, 12] > 0
    assert maps.action_reason_mask[3, 18] > 0
    assert maps.action_reason_mask[1, 6] > 0
