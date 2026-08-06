from fate_oia.datasets.aie_structured_evidence import AIEStructuredEvidenceBuilder


def test_generic_sign_does_not_become_stop_sign():
    builder = AIEStructuredEvidenceBuilder("configs/aie_scene_predicates.yaml", None)
    out = builder.build_from_records([{"labels": [{"category": "traffic sign", "box2d": {"x1": 10, "y1": 10, "x2": 20, "y2": 20}}]}])
    assert out["predicate_target_mask"][0, builder.name_to_id["stop_sign_present"]] == 0

