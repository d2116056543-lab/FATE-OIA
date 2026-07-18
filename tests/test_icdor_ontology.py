from __future__ import annotations

from pathlib import Path

from fate_oia.models.mosaic_native_semantics import load_icdor_ontology


def test_icdor_ontology_has_complete_observable_factors_and_target_routes() -> None:
    config_root = Path("configs")
    ontology = load_icdor_ontology(config_root)

    assert tuple(ontology["action_names"]) == ("forward", "stop", "left", "right")
    assert len(ontology["reason_names"]) == 21
    assert len(ontology["factors"]) >= 20
    assert set(ontology["action_routes"]) == set(ontology["action_names"])
    assert set(ontology["reason_routes"]) == set(range(21))

    for factor in ontology["factors"]:
        assert factor["type"] in {"point", "object", "curve", "region"}
        assert factor["name"] in ontology["factor_index"]
        assert factor["mirror_of"] in ontology["factor_index"]
        assert factor["name"] not in {"no_left_lane", "left_turn_allowed"}
        assert "state" not in factor["name"]
        assert factor["role"] in {"observable", "reason_only", "latent_only"}
        assert factor["source_kind"] in {"grounded", "image_only", "proxy"}
        assert factor["visual_sources"] == factor["grounding_sources"]
        assert factor["negative_policy"] in {
            "reliable_if_source_complete",
            "reliable_if_attribute_complete",
            "unknown_without_depth",
            "unknown_without_direct_visual_evidence",
        }
        assert set(factor["target_candidates"]) == {"actions", "reasons"}
        assert set(factor["target_candidates"]["actions"]).issubset(ontology["action_index"])
        assert set(factor["target_candidates"]["reasons"]).issubset(set(range(21)))

    for action_name, directions in ontology["action_routes"].items():
        assert action_name in ontology["action_index"]
        assert set(directions) == {"support", "veto"}
        for direction, edges in directions.items():
            assert direction in {"support", "veto"}
            for edge in edges:
                assert edge["factor"] in ontology["factor_index"]
                assert edge["polarity"] in {"present", "absent"}
                assert set(edge) == {"factor", "polarity"}

    rules = ontology["certificate_rules"]
    assert rules["certified"]["min_confirmed_positive"] == 32
    assert rules["certified"]["min_reliable_negative"] == 32
    assert rules["certified"]["min_geometry_valid"] == 200
    assert rules["certified"]["min_full_minus_prior_lcb95"] == 0.02


def test_icdor_ontology_exposes_only_factor_routes_for_latent_reasons() -> None:
    ontology = load_icdor_ontology(Path("configs"))

    for reason_index, route in ontology["reason_routes"].items():
        assert reason_index in range(21)
        assert set(route) == {
            "group",
            "direct_factors",
            "latent_factors",
            "contradiction_factors",
            "escape_allowed",
            "absence_factors",
            "semantic_kind",
        }
        assert set(route["direct_factors"]).issubset(ontology["factor_index"])
        assert set(route["latent_factors"]).issubset(ontology["factor_index"])
        assert set(route["contradiction_factors"]).issubset(ontology["factor_index"])
        assert set(route["absence_factors"]).issubset(ontology["factor_index"])
        assert route["semantic_kind"] in {"observable_or_latent", "visual_plus_latent_proxy", "absence_observable"}
        assert all("state" not in factor for factor in route["latent_factors"])


def test_icdor_ontology_cross_checks_factor_target_candidates_against_runtime_routes() -> None:
    ontology = load_icdor_ontology(Path("configs"))
    factors = {factor["name"]: factor for factor in ontology["factors"]}
    for action_name, directions in ontology["action_routes"].items():
        for edges in directions.values():
            for edge in edges:
                assert action_name in factors[edge["factor"]]["target_candidates"]["actions"]
    for reason_index, route in ontology["reason_routes"].items():
        for field in ("direct_factors", "latent_factors", "contradiction_factors", "absence_factors"):
            for factor_name in route[field]:
                assert reason_index in factors[factor_name]["target_candidates"]["reasons"]
