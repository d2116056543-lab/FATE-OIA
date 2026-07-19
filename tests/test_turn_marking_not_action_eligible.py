from __future__ import annotations

from fate_oia.models.mosaic_native_semantics import load_icdor_ontology


def test_turn_marking_not_action_eligible() -> None:
    ontology = load_icdor_ontology("configs")
    forbidden = {"left_turn_marking_visible", "right_turn_marking_visible"}
    routed = {
        edge["factor"]
        for directions in ontology["action_routes"].values()
        for edges in directions.values()
        for edge in edges
    }
    assert forbidden.isdisjoint(routed)
