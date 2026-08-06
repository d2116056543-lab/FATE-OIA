from fate_oia.datasets.aie_structured_evidence import AIEStructuredEvidenceBuilder


def test_unknown_object_absence_is_not_automatic_negative():
    builder = AIEStructuredEvidenceBuilder("configs/aie_scene_predicates.yaml", None)
    out = builder.build_from_records([{}])
    assert int(out["predicate_target_mask"].sum()) == 0
    assert int(out["predicate_counter_mask"].sum()) == 0

