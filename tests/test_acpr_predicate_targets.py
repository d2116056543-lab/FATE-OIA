from fate_oia.models.acpr_predicate_targets import WeakPredicateTargetBuilder


def test_acpr_predicate_target_builder_missing_bdd100k():
    builder = WeakPredicateTargetBuilder("configs/acpr_scene_predicates.yaml", None)
    out = builder.build(["a.jpg", "b.jpg"])
    assert out["predicate_targets"].shape[0] == 2
    assert out["predicate_targets"].shape[1] >= 32
    assert out["source_counts"]["missing"] == 2
