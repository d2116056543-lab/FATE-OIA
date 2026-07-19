from __future__ import annotations

from fate_oia.models.mosaic_native_semantics import load_icdor_ontology


_REQUIRED_V5_FIELDS = {
    "visual_role",
    "source_attributes",
    "presence_polarity",
    "observability_policy",
    "action_eligible",
    "reason_eligible",
    "geometry_loss_type",
}


def test_every_factor_declares_v5_visual_semantics() -> None:
    ontology = load_icdor_ontology("configs")
    for factor in ontology["factors"]:
        assert _REQUIRED_V5_FIELDS <= set(factor), factor["name"]
        assert factor["source_attributes"] == factor["attribute_constraints"]
        assert factor["action_eligible"] is bool(factor["target_candidates"]["actions"])
        if factor["source_kind"] == "image_only":
            assert factor["action_eligible"] is False
        if factor["type"] == "curve":
            assert factor["geometry_loss_type"] == "curve_distance"
        else:
            assert factor["geometry_loss_type"] == "object_region_dice"
