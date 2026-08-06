from fate_oia.datasets.aie_structured_evidence import AIEStructuredEvidenceBuilder


def test_generic_lane_without_attributes_does_not_guess_turn_or_merge():
    builder = AIEStructuredEvidenceBuilder("configs/aie_scene_predicates.yaml", None)
    out = builder.build_from_records([{"labels": [{"category": "lane", "poly2d": []}]}])
    for name in ("left_turn_region", "right_turn_region", "merging_left_context", "merging_right_context"):
        assert out["predicate_target_mask"][0, builder.name_to_id[name]] == 0

