from fate_oia.datasets.aie_structured_evidence import AIEStructuredEvidenceBuilder


def test_traffic_light_presence_does_not_guess_color():
    builder = AIEStructuredEvidenceBuilder("configs/aie_scene_predicates.yaml", None)
    out = builder.build_from_records([{"labels": [{"category": "traffic light", "box2d": {"x1": 10, "y1": 10, "x2": 20, "y2": 20}}]}])
    ids = builder.name_to_id
    assert out["predicate_target"][0, ids["traffic_light_visible"]] == 1
    assert out["predicate_target_mask"][0, ids["traffic_light_green"]] == 0
    assert out["predicate_target_mask"][0, ids["traffic_light_red"]] == 0

