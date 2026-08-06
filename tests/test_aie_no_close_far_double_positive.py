from fate_oia.datasets.aie_structured_evidence import AIEStructuredEvidenceBuilder


def test_vehicle_geometry_never_marks_close_and_far_together():
    builder = AIEStructuredEvidenceBuilder("configs/aie_scene_predicates.yaml", None)
    record = {"labels": [{"category": "car", "box2d": {"x1": 500, "y1": 300, "x2": 700, "y2": 650}}]}
    out = builder.build_from_records([record]); ids = builder.name_to_id
    assert out["predicate_target"][0, [ids["front_vehicle_close"], ids["front_vehicle_far"]]].sum() <= 1

