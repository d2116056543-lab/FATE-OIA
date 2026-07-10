from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from fate_oia.models.mosaic_native_semantics import (
    _load_yaml,
    _normalize_reason_mapping,
    _require_document,
    load_mosaic_schema_bundle,
    validate_mosaic_schema_bundle,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "configs"
OFFICIAL_REASON_SOURCE_FILE = "configs/bdd_oia_reason_names_cvpr2020_table1.yaml"
OFFICIAL_REASON_SOURCE_SHA256 = "DCB3B28E7FCA784953CC01BC7406B4D74CAE288AADD95215029CA8294065AD24"


def _rename_factor(bundle: dict, old: str, new: str) -> None:
    for factor in bundle["factors"]:
        if factor["name"] == old:
            factor["name"] = new
        if factor["mirror_partner"] == old:
            factor["mirror_partner"] = new
        factor["contradicts"] = [new if item == old else item for item in factor["contradicts"]]
    for state in bundle["states"].values():
        for group in state["required_groups"]:
            group["any_of"] = [new if item == old else item for item in group["any_of"]]
        state["veto"] = [new if item == old else item for item in state["veto"]]
    for mapping in bundle["reason_observation"].values():
        for field in ("support_factors", "contradiction_factors", "visibility_factors"):
            mapping[field] = [new if item == old else item for item in mapping[field]]


def _make_directional_pair_identical(bundle: dict, left_reason: int, right_reason: int) -> None:
    right = bundle["reason_observation"][right_reason]
    left = bundle["reason_observation"][left_reason]
    for field in ("support_factors", "support_states", "visibility_factors", "contradiction_factors"):
        left[field] = list(right[field])
    for factor in bundle["factors"]:
        if left_reason in factor["reason_positive_anchors"] and factor["name"] not in (
            set(left["support_factors"]) | set(left["visibility_factors"])
        ):
            factor["reason_positive_anchors"].remove(left_reason)


def _remove_directional_evidence(bundle: dict, reason_id: int) -> None:
    mapping = bundle["reason_observation"][reason_id]
    mapping["support_factors"] = ["front_vehicle_visible", "pedestrian_front_visible"]
    mapping["support_states"] = []
    mapping["visibility_factors"] = ["front_vehicle_visible", "pedestrian_front_visible"]
    mapping["contradiction_factors"] = []
    for factor in bundle["factors"]:
        if reason_id in factor["reason_positive_anchors"] and factor["name"] not in mapping["support_factors"]:
            factor["reason_positive_anchors"].remove(reason_id)


def _replace_directional_evidence_with_lane_markings(bundle: dict) -> None:
    for reason_id, factor_name in ((14, "left_lane_marking_visible"), (20, "right_lane_marking_visible")):
        mapping = bundle["reason_observation"][reason_id]
        mapping["support_factors"] = ["front_vehicle_visible", factor_name]
        mapping["support_states"] = []
        mapping["visibility_factors"] = ["front_vehicle_visible", factor_name]
        mapping["contradiction_factors"] = []
        for factor in bundle["factors"]:
            if reason_id in factor["reason_positive_anchors"] and factor["name"] not in mapping["support_factors"]:
                factor["reason_positive_anchors"].remove(reason_id)


def _replace_reason_name(bundle: dict, reason_id: int, name: str) -> None:
    bundle["label_schema"]["reasons"][reason_id]["name"] = name
    bundle["reason_source_names"][reason_id] = name


def _move_directional_indicator_to_visibility_only(bundle: dict, reason_id: int, indicator: str) -> None:
    mapping = bundle["reason_observation"][reason_id]
    mapping["support_factors"] = [name for name in mapping["support_factors"] if name != indicator]
    if indicator not in mapping["visibility_factors"]:
        mapping["visibility_factors"].append(indicator)
    for factor in bundle["factors"]:
        if factor["name"] == indicator and reason_id in factor["reason_positive_anchors"]:
            factor["reason_positive_anchors"].remove(reason_id)


def _support_both_directional_indicators(bundle: dict, reason_id: int) -> None:
    mapping = bundle["reason_observation"][reason_id]
    mapping["support_factors"] = [
        "front_vehicle_visible",
        "front_vehicle_left_indicator_visible",
        "front_vehicle_right_indicator_visible",
    ]
    mapping["visibility_factors"] = list(mapping["support_factors"])
    mapping["contradiction_factors"] = []


def test_schema_has_provenance_4_actions_21_reasons_24_factors_8_states() -> None:
    bundle = load_mosaic_schema_bundle(CONFIG_ROOT)
    labels = bundle["label_schema"]
    factors = bundle["factors"]
    states = bundle["states"]
    reason_observation = bundle["reason_observation"]

    assert labels["action_dim"] == 4
    assert labels["reason_dim"] == 21
    assert [item["index"] for item in labels["actions"]] == list(range(4))
    assert [item["name"] for item in labels["actions"]] == ["forward", "stop", "left", "right"]
    assert [item["index"] for item in labels["reasons"]] == list(range(21))
    assert len({item["name"] for item in labels["reasons"]}) == 21
    assert [item["name"] for item in labels["reasons"]] == bundle["reason_source_names"]
    assert all(not item["name"].lower().startswith("reason_") for item in labels["reasons"])
    assert labels["reason_semantics"]["semantic_source"] == "official_cvpr2020_table1_category_order"
    assert labels["reason_semantics"]["raw_json_contains_names"] is False
    assert labels["reason_semantics"]["official_mapping_verified"] is True
    assert labels["reason_semantics"]["source_file"] == OFFICIAL_REASON_SOURCE_FILE
    assert labels["reason_semantics"]["source_sha256"] == OFFICIAL_REASON_SOURCE_SHA256
    assert labels["reason_semantics"]["training_role"] == "canonical_index_semantics_and_weak_factor_config"

    assert len(factors) == 24
    assert len({factor["name"] for factor in factors}) == 24
    factor_names = {factor["name"] for factor in factors}
    for factor in factors:
        assert factor["type"] in {"point", "object", "curve", "region"}
        assert factor["num_prototypes"] in {2, 3, 4}
        assert factor["region_prior"] in {
            "upper_front",
            "front_center",
            "left_corridor",
            "right_corridor",
            "center_corridor",
        }
        assert set(factor) >= {
            "name",
            "type",
            "entity",
            "attribute",
            "spatial",
            "num_prototypes",
            "region_prior",
            "reason_positive_anchors",
            "geometry_sources",
            "mirror_partner",
            "contradicts",
        }
        assert all(index in range(21) for index in factor["reason_positive_anchors"])
        assert set(factor["geometry_sources"]) <= {"box2d", "lane_polyline", "drivable_mask", "none"}
        assert factor["mirror_partner"] in factor_names
        assert set(factor["contradicts"]) <= factor_names

    by_name = {factor["name"]: factor for factor in factors}
    for factor in factors:
        assert by_name[factor["mirror_partner"]]["mirror_partner"] == factor["name"]
        for other in factor["contradicts"]:
            assert factor["name"] in by_name[other]["contradicts"]

    assert len(states) == 8
    assert set(states) == {
        "front_risk",
        "stop_obligation",
        "forward_feasible",
        "lane_follow_permitted",
        "left_affordance",
        "left_veto",
        "right_affordance",
        "right_veto",
    }

    assert set(reason_observation) == set(range(21))
    for reason_id, mapping in reason_observation.items():
        assert mapping["group"] in {"traffic_control", "obstacle", "lane", "other"}
        assert mapping["support_factors"] or mapping["support_states"]
        assert mapping["support_semantics"] in {"direct_observable", "weak_proxy"}
        assert mapping["visibility_factors"]
        assert set(mapping["support_factors"]) <= factor_names
        assert set(mapping["contradiction_factors"]) <= factor_names
        assert set(mapping["visibility_factors"]) <= factor_names
        assert set(mapping["support_states"]) <= set(states)
        assert 0.0 <= mapping["false_positive_max"] <= 0.05
        assert reason_id == int(reason_id)


def test_schema_validation_rejects_state_cycles_and_asymmetric_mirrors() -> None:
    bundle = load_mosaic_schema_bundle(CONFIG_ROOT)

    cyclic = deepcopy(bundle)
    cyclic["states"]["front_risk"]["required_groups"] = [{"any_of": ["stop_obligation"]}]
    with pytest.raises(ValueError, match="cycle"):
        validate_mosaic_schema_bundle(cyclic)

    asymmetric = deepcopy(bundle)
    asymmetric["factors"][0]["mirror_partner"] = asymmetric["factors"][1]["name"]
    with pytest.raises(ValueError, match="mirror"):
        validate_mosaic_schema_bundle(asymmetric)


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (lambda b: b["label_schema"]["reasons"][0].update(name="wrong semantic"), "official reason names"),
        (lambda b: b["states"].pop("front_risk"), "state names"),
        (lambda b: b["states"]["front_risk"].pop("veto"), "required_groups/veto"),
        (
            lambda b: b["states"]["front_risk"].update(
                required_groups=[{"any_of": ["front_vehicle_near"]}], veto=["front_vehicle_near"]
            ),
            "support and veto",
        ),
        (lambda b: b["factors"][0].update(entity=""), "entity"),
        (lambda b: b["factors"][0].update(geometry_sources=[]), "geometry source"),
        (lambda b: b["factors"][0].update(reason_positive_anchors="3"), "reason anchors"),
        (lambda b: b["factors"][0].update(contradicts=[b["factors"][0]["name"]]), "contradict itself"),
        (lambda b: b["reason_observation"][0].pop("contradiction_factors"), "missing fields"),
        (
            lambda b: b["reason_observation"][0].update(
                support_factors=["green_light_visible"], contradiction_factors=["green_light_visible"]
            ),
            "support and contradiction",
        ),
        (lambda b: b["label_schema"]["actions"][0].update(index="0"), "integer indices"),
        (lambda b: b["factors"][0].update(num_prototypes=3.5), "prototype count"),
        (lambda b: b["states"]["front_risk"].update(required_groups=[]), "required group"),
        (
            lambda b: b["states"]["front_risk"].update(required_groups=[{"any_of": {"front_vehicle_near": 1}}]),
            "any_of list",
        ),
        (lambda b: b["reason_observation"][0].update(contradiction_factors=""), "list fields"),
        (lambda b: b["factors"][0].update(spatial=""), "spatial"),
        (lambda b: b["factors"][0].update(geometry_sources=["none", "box2d"]), "none geometry source"),
        (lambda b: b["factors"][0].update(reason_positive_anchors=[True]), "reason anchors"),
        (lambda b: b["label_schema"].update(action_dim=4.0), "integer dimensions"),
        (lambda b: b["label_schema"]["actions"][0].update(name=True), "string names"),
        (lambda b: b["reason_observation"][0].update(false_positive_max="0.03"), "numeric false_positive_max"),
        (lambda b: b["factors"][0].update(extra_field=True), "unknown fields"),
        (lambda b: b["states"]["front_risk"].update(extra_field=True), "unknown fields"),
        (lambda b: b["reason_observation"][0].update(extra_field=True), "unknown fields"),
        (lambda b: b["label_schema"]["reason_semantics"].update(source_file="configs/copy.yaml"), "source_file"),
        (lambda b: b["reason_observation"][0].update(support_states=["forward_feasible"]), "direct observable"),
        (lambda b: b["factors"][0].update(attribute=True), "attribute"),
        (lambda b: b["factors"][0].update(geometry_sources=[["box2d"]]), "geometry source strings"),
        (lambda b: b["factors"].__setitem__(0, []), "factor entries"),
        (lambda b: b["label_schema"]["reason_semantics"].update(source_sha256=False), "source_sha256"),
        (lambda b: b.update(extra_document={}), "unknown top-level"),
    ],
)
def test_schema_validation_rejects_unsafe_or_incomplete_metadata(mutate, error: str) -> None:
    bundle = deepcopy(load_mosaic_schema_bundle(CONFIG_ROOT))
    mutate(bundle)
    with pytest.raises(ValueError, match=error):
        validate_mosaic_schema_bundle(bundle)


def test_state_and_reason_semantics_do_not_encode_unsafe_shortcuts() -> None:
    bundle = load_mosaic_schema_bundle(CONFIG_ROOT)
    states = bundle["states"]
    reasons = bundle["reason_observation"]
    factors = {factor["name"]: factor for factor in bundle["factors"]}

    assert "green_light_visible" not in states["stop_obligation"]["veto"]
    forward_support = {item for group in states["forward_feasible"]["required_groups"] for item in group["any_of"]}
    assert "traffic_light_visible" not in forward_support

    assert "left_drivable_visible" not in reasons[10]["contradiction_factors"]
    assert "right_drivable_visible" not in reasons[16]["contradiction_factors"]
    assert not reasons[12]["contradiction_factors"]
    assert not reasons[18]["contradiction_factors"]
    assert "left_turn_marking_visible" not in reasons[14]["support_factors"]
    assert "right_turn_marking_visible" not in reasons[20]["support_factors"]
    assert reasons[13]["support_semantics"] == "weak_proxy"
    assert reasons[19]["support_semantics"] == "weak_proxy"

    assert factors["traffic_light_visible"]["reason_positive_anchors"] == [3]
    assert factors["red_light_visible"]["reason_positive_anchors"] == []
    assert factors["green_light_visible"]["reason_positive_anchors"] == [0]
    assert "stop_sign_visible" not in factors
    assert factors["traffic_sign_visible"]["reason_positive_anchors"] == [4]
    assert factors["left_turn_marking_visible"]["reason_positive_anchors"] == [12]
    assert factors["right_turn_marking_visible"]["reason_positive_anchors"] == [18]
    assert factors["front_vehicle_left_indicator_visible"]["reason_positive_anchors"] == [14]
    assert factors["front_vehicle_right_indicator_visible"]["reason_positive_anchors"] == [20]
    assert reasons[4]["support_semantics"] == "direct_observable"
    assert reasons[14]["support_factors"] != reasons[20]["support_factors"]
    assert reasons[2]["support_factors"] == ["center_drivable_visible"]
    assert reasons[2]["support_states"] == []
    assert states["forward_feasible"]["veto"] == ["stop_obligation"]
    for mapping in reasons.values():
        if mapping["support_semantics"] == "direct_observable":
            assert mapping["support_states"] == []
    for reason_id, mapping in reasons.items():
        if mapping["support_semantics"] == "direct_observable":
            assert all(reason_id in factors[name]["reason_positive_anchors"] for name in mapping["support_factors"])


def test_reason_mapping_normalization_rejects_colliding_keys() -> None:
    with pytest.raises(ValueError, match="colliding reason keys"):
        _normalize_reason_mapping({0: {"group": "first"}, "00": {"group": "replacement"}})


def test_yaml_loader_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.yaml"
    path.write_text("reasons:\n  0: first\n  0: replacement\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate YAML key"):
        _load_yaml(path)


def test_validator_wraps_malformed_top_level_as_schema_error() -> None:
    malformed = {"label_schema": [], "factors": [], "states": [], "reason_observation": []}
    with pytest.raises(ValueError, match="schema bundle"):
        validate_mosaic_schema_bundle(malformed)


def test_document_wrapper_validation_rejects_missing_payload() -> None:
    with pytest.raises(ValueError, match="mosaic_observable_factors.yaml"):
        _require_document({"wrong": []}, "factors", list, "mosaic_observable_factors.yaml")


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (lambda b: b["label_schema"]["actions"][0].update(aliases=["go", "go"]), "duplicate aliases"),
        (lambda b: b["factors"][0].update(geometry_sources=["box2d", "box2d"]), "duplicate geometry"),
        (lambda b: b["factors"][0].update(reason_positive_anchors=[3, 3]), "duplicate reason anchors"),
        (lambda b: b["factors"][0].update(contradicts=[["green_light_visible"]]), "contradiction names"),
        (
            lambda b: b["states"]["front_risk"].update(
                required_groups=[{"any_of": ["front_vehicle_near", "front_vehicle_near"]}]
            ),
            "duplicate state references",
        ),
        (
            lambda b: b["reason_observation"][0].update(
                visibility_factors=["traffic_light_visible", "traffic_light_visible"]
            ),
            "duplicate reason references",
        ),
    ],
)
def test_schema_validation_rejects_duplicate_or_unhashable_references(mutate, error: str) -> None:
    bundle = deepcopy(load_mosaic_schema_bundle(CONFIG_ROOT))
    mutate(bundle)
    with pytest.raises(ValueError, match=error):
        validate_mosaic_schema_bundle(bundle)


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (
            lambda b: (
                b["factors"][1].update(reason_positive_anchors=[0]),
                b["factors"][2].update(reason_positive_anchors=[0]),
            ),
            "contradictory factors share reason anchors",
        ),
        (
            lambda b: _make_directional_pair_identical(b, 14, 20),
            "directional reason pair",
        ),
        (
            lambda b: b["reason_observation"][2].update(support_states=["forward_feasible"]),
            "duplicates factor support through state",
        ),
        (lambda b: _rename_factor(b, "traffic_light_visible", "front_risk"), "namespaces overlap"),
        (
            lambda b: next(f for f in b["factors"] if f["type"] == "curve").update(
                geometry_sources=["drivable_mask"]
            ),
            "geometry source incompatible",
        ),
        (
            lambda b: next(f for f in b["factors"] if f["name"] == "traffic_light_visible").update(
                reason_positive_anchors=[3, 8]
            ),
            "non-reciprocal reason anchor",
        ),
        (
            lambda b: b["label_schema"]["reason_semantics"].update(official_mapping_verified=False),
            "must remain verified",
        ),
        (lambda b: _remove_directional_evidence(b, 14), "required directional evidence"),
        (
            lambda b: b["reason_observation"][13].update(
                contradiction_factors=["left_corridor_occupied"]
            ),
            "duplicates contradiction through state veto",
        ),
        (
            lambda b: next(f for f in b["factors"] if f["name"] == "red_light_visible").update(
                reason_positive_anchors=[3]
            ),
            "non-reciprocal reason anchor",
        ),
        (
            lambda b: b["states"]["forward_feasible"].update(
                veto=["stop_obligation", "front_risk"]
            ),
            "duplicate signed factor paths",
        ),
        (
            lambda b: _replace_directional_evidence_with_lane_markings(b),
            "required directional evidence",
        ),
        (lambda b: _replace_reason_name(b, 0, "TBD reason zero"), "official reason names"),
        (lambda b: b["label_schema"]["actions"][0].update(name="TBD action"), "official action names"),
        (
            lambda b: _rename_factor(b, "traffic_light_visible", "traffic_light_visible "),
            "canonical factor names",
        ),
        (lambda b: b["factors"][0].update(type=[]), "invalid geometry type"),
        (lambda b: b["factors"][0].update(region_prior=[]), "invalid region prior"),
        (lambda b: b["reason_observation"][0].update(group=[]), "invalid group"),
        (lambda b: b["reason_observation"][0].update(support_semantics=[]), "invalid support semantics"),
        (
            lambda b: _move_directional_indicator_to_visibility_only(
                b, 14, "front_vehicle_left_indicator_visible"
            ),
            "required directional evidence",
        ),
        (lambda b: _support_both_directional_indicators(b, 14), "mutually contradictory support factors"),
        (lambda b: _rename_factor(b, "rider_front_visible", "factor_7"), "canonical factor schema"),
        (
            lambda b: next(
                f for f in b["factors"] if f["name"] == "front_vehicle_right_indicator_visible"
            ).update(attribute="left_indicator"),
            "canonical factor schema",
        ),
        (
            lambda b: next(f for f in b["factors"] if f["name"] == "left_obstacle_visible").update(
                region_prior="right_corridor"
            ),
            "canonical factor schema",
        ),
        (
            lambda b: b["states"].update(
                left_affordance=deepcopy(b["states"]["right_affordance"])
            ),
            "canonical decision state schema",
        ),
        (
            lambda b: b["reason_observation"][1].update(
                support_factors=["front_vehicle_visible", "green_light_visible"],
                visibility_factors=["green_light_visible"],
            ),
            "canonical reason observation schema",
        ),
    ],
)
def test_schema_validation_rejects_semantic_training_conflicts(mutate, error: str) -> None:
    bundle = deepcopy(load_mosaic_schema_bundle(CONFIG_ROOT))
    mutate(bundle)
    with pytest.raises(ValueError, match=error):
        validate_mosaic_schema_bundle(bundle)
