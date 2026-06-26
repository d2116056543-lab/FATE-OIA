from __future__ import annotations

from fate_oia.models.acpr_predicate_targets import WeakPredicateTargetBuilder


def test_predicate_target_builder_constructs():
    builder = WeakPredicateTargetBuilder("configs/acpr_scene_predicates.yaml")
    assert len(builder.names) >= 32

