from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml


_REQUIRED_FACTOR_FIELDS = {
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
_FACTOR_TYPES = {"point", "object", "curve", "region"}
_REGION_PRIORS = {"upper_front", "front_center", "left_corridor", "right_corridor", "center_corridor"}
_GEOMETRY_SOURCES = {"box2d", "lane_polyline", "drivable_mask", "none"}
_GEOMETRY_SOURCE_BY_TYPE = {
    "point": {"box2d", "none"},
    "object": {"box2d", "none"},
    "curve": {"lane_polyline", "none"},
    "region": {"box2d", "drivable_mask", "none"},
}
_EXPECTED_STATES = {
    "front_risk",
    "stop_obligation",
    "forward_feasible",
    "lane_follow_permitted",
    "left_affordance",
    "left_veto",
    "right_affordance",
    "right_veto",
}
_REQUIRED_REASON_FIELDS = {
    "group",
    "support_semantics",
    "support_factors",
    "contradiction_factors",
    "support_states",
    "visibility_factors",
    "false_positive_max",
}
_LABEL_SCHEMA_FIELDS = {"action_dim", "reason_dim", "actions", "reasons", "reason_semantics"}
_ACTION_FIELDS = {"index", "name", "aliases", "group"}
_REASON_LABEL_FIELDS = {"index", "name"}
_SEMANTIC_FIELDS = {
    "semantic_source",
    "raw_json_contains_names",
    "official_mapping_verified",
    "training_role",
    "source_file",
    "source_sha256",
}
_EXPECTED_REASON_SOURCE_FILE = "configs/bdd_oia_reason_names_cvpr2020_table1.yaml"
_EXPECTED_REASON_SOURCE_SHA256 = "DCB3B28E7FCA784953CC01BC7406B4D74CAE288AADD95215029CA8294065AD24"
_EXPECTED_REASON_PAPER_URL = (
    "https://openaccess.thecvf.com/content_CVPR_2020/papers/"
    "Xu_Explainable_Object-Induced_Action_Decision_for_Autonomous_Vehicles_CVPR_2020_paper.pdf"
)
_EXPECTED_FACTOR_SCHEMA_SHA256 = "A0B95058D79DEC40005B914E0281E9CAB49BCAEBEA0871AEE8A3F48D0578422F"
_EXPECTED_STATE_SCHEMA_SHA256 = "CCE45554D990858AAE10FA2BC89314DF2B56046822D4CDD90E47E88C0F6078F9"
_EXPECTED_REASON_OBSERVATION_SHA256 = "820F7F4A6C7C02C69FC5E0F383FE83C93D93D1AE4C3CCF50BCE71B31D7D06A50"
_EXPECTED_ACTION_NAMES = ("forward", "stop", "left", "right")
_EXPECTED_REASON_NAMES = (
    "Traffic light is green",
    "Follow traffic",
    "Road is clear",
    "Traffic light",
    "Traffic sign",
    "Obstacle: car",
    "Obstacle: person",
    "Obstacle: rider",
    "Obstacle: others",
    "No lane on the left",
    "Obstacles on the left lane",
    "Solid line on the left",
    "On the left-turn lane",
    "Traffic light allows left",
    "Front car turning left",
    "No lane on the right",
    "Obstacles on the right lane",
    "Solid line on the right",
    "On the right-turn lane",
    "Traffic light allows right",
    "Front car turning right",
)
_DIRECTIONAL_REASON_PAIRS = ((9, 15), (10, 16), (11, 17), (12, 18), (13, 19), (14, 20))
_DIRECTIONAL_REQUIRED_EVIDENCE = {
    (9, 15): ({"left_solid_boundary_visible"}, {"right_solid_boundary_visible"}),
    (10, 16): ({"left_obstacle_visible"}, {"right_obstacle_visible"}),
    (11, 17): ({"left_solid_boundary_visible"}, {"right_solid_boundary_visible"}),
    (12, 18): ({"left_turn_marking_visible"}, {"right_turn_marking_visible"}),
    (13, 19): (
        {"traffic_light_visible", "left_affordance"},
        {"traffic_light_visible", "right_affordance"},
    ),
    (14, 20): (
        {"front_vehicle_visible", "front_vehicle_left_indicator_visible"},
        {"front_vehicle_visible", "front_vehicle_right_indicator_visible"},
    ),
}
_STATE_MIRRORS = {
    "left_affordance": "right_affordance",
    "right_affordance": "left_affordance",
    "left_veto": "right_veto",
    "right_veto": "left_veto",
}


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise ValueError(f"unhashable YAML key in {loader.name}") from exc
        if duplicate:
            raise ValueError(f"duplicate YAML key {key!r} in {loader.name}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)


def _load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.load(stream, Loader=_UniqueKeyLoader)


def _require_document(document: Any, key: str, value_type: type, document_name: str) -> Any:
    if not isinstance(document, dict) or set(document) != {key} or not isinstance(document[key], value_type):
        raise ValueError(f"{document_name} must contain exactly one {key!r} {value_type.__name__}")
    return document[key]


def _normalize_reason_mapping(raw_mapping: Any) -> dict[int, Any]:
    if not isinstance(raw_mapping, dict):
        raise ValueError("reason mapping must be a YAML mapping")
    normalized: dict[int, Any] = {}
    for raw_key, value in raw_mapping.items():
        if type(raw_key) is int:
            reason_id = raw_key
        elif isinstance(raw_key, str) and raw_key.isdecimal():
            reason_id = int(raw_key)
        else:
            raise ValueError(f"invalid reason key {raw_key!r}")
        if reason_id in normalized:
            raise ValueError(f"colliding reason keys normalize to {reason_id}")
        if isinstance(raw_key, str) and raw_key != str(reason_id):
            raise ValueError(f"invalid non-canonical reason key {raw_key!r}")
        normalized[reason_id] = value
    return normalized


def _require_unique_values(values: list[Any], duplicate_error: str, type_error: str | None = None) -> None:
    seen: set[Any] = set()
    for value in values:
        try:
            duplicate = value in seen
        except TypeError as exc:
            raise ValueError(type_error or duplicate_error) from exc
        if duplicate:
            raise ValueError(duplicate_error)
        try:
            seen.add(value)
        except TypeError as exc:
            raise ValueError(type_error or duplicate_error) from exc


def _validate_contiguous(items: list[dict[str, Any]], count: int, name: str) -> None:
    raw_indices = [item.get("index") for item in items]
    if any(type(index) is not int for index in raw_indices):
        raise ValueError(f"{name} must use exact integer indices")
    indices = list(raw_indices)
    if indices != list(range(count)):
        raise ValueError(f"{name} indices must be contiguous 0..{count - 1}, got {indices}")
    raw_names = [item.get("name") for item in items]
    if any(not isinstance(value, str) for value in raw_names):
        raise ValueError(f"{name} must use exact string names")
    names = [value.strip() for value in raw_names]
    if len(set(names)) != count or any(not value for value in names):
        raise ValueError(f"{name} names must be non-empty and unique strings")
    if any(value.lower().startswith(f"{name[:-1]}_") for value in names):
        raise ValueError(f"{name} contains placeholder names")


def _state_dependencies(states: dict[str, Any], factor_names: set[str]) -> dict[str, set[str]]:
    state_names = set(states)
    dependencies: dict[str, set[str]] = {name: set() for name in states}
    for state_name, spec in states.items():
        if not isinstance(spec, dict) or not {"required_groups", "veto"} <= set(spec):
            raise ValueError(f"state {state_name} required_groups/veto fields are mandatory")
        if set(spec) != {"required_groups", "veto"}:
            raise ValueError(f"state {state_name} has unknown fields")
        groups = spec["required_groups"]
        veto = spec["veto"]
        if not isinstance(groups, list) or not groups or not isinstance(veto, list):
            raise ValueError(f"state {state_name} required group and veto must be non-empty/list metadata")
        references: list[str] = []
        for group in groups:
            if not isinstance(group, dict) or set(group) != {"any_of"}:
                raise ValueError(f"state {state_name} has an invalid required group")
            alternatives = group["any_of"]
            if not isinstance(alternatives, list) or not alternatives or any(
                not isinstance(item, str) or not item for item in alternatives
            ):
                raise ValueError(f"state {state_name} any_of list must contain non-empty references")
            references.extend(alternatives)
        if any(not isinstance(item, str) or not item for item in veto):
            raise ValueError(f"state {state_name} veto list must contain non-empty references")
        references.extend(veto)
        support_reference_list = [item for group in groups for item in group["any_of"]]
        support_references = set(support_reference_list)
        overlap = support_references & set(veto)
        if overlap:
            raise ValueError(f"state {state_name} uses {sorted(overlap)} as both support and veto")
        _require_unique_values(support_reference_list, f"state {state_name} has duplicate state references")
        _require_unique_values(veto, f"state {state_name} has duplicate state references")
        for reference in references:
            if reference in state_names:
                dependencies[state_name].add(reference)
            elif reference not in factor_names:
                raise ValueError(f"state {state_name} references unknown factor/state {reference}")
    return dependencies


def _assert_acyclic(dependencies: dict[str, set[str]]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            raise ValueError(f"state dependency cycle detected at {node}")
        if node in visited:
            return
        visiting.add(node)
        for dependency in dependencies[node]:
            visit(dependency)
        visiting.remove(node)
        visited.add(node)

    for node in dependencies:
        visit(node)


def _add_path_counts(target: dict[str, int], source: dict[str, int]) -> None:
    for factor_name, count in source.items():
        target[factor_name] = target.get(factor_name, 0) + count


def _state_signed_factor_closure(
    state_name: str,
    states: dict[str, Any],
    factor_names: set[str],
    memo: dict[str, tuple[dict[str, int], dict[str, int]]],
) -> tuple[dict[str, int], dict[str, int]]:
    if state_name in memo:
        return memo[state_name]
    positive: dict[str, int] = {}
    negative: dict[str, int] = {}
    for group in states[state_name]["required_groups"]:
        for reference in group["any_of"]:
            if reference in factor_names:
                positive[reference] = positive.get(reference, 0) + 1
            else:
                child_positive, child_negative = _state_signed_factor_closure(reference, states, factor_names, memo)
                _add_path_counts(positive, child_positive)
                _add_path_counts(negative, child_negative)
    for reference in states[state_name]["veto"]:
        if reference in factor_names:
            negative[reference] = negative.get(reference, 0) + 1
        else:
            child_positive, child_negative = _state_signed_factor_closure(reference, states, factor_names, memo)
            _add_path_counts(negative, child_positive)
            _add_path_counts(positive, child_negative)
    duplicate_paths = {name for name, count in positive.items() if count > 1} | {
        name for name, count in negative.items() if count > 1
    }
    if duplicate_paths:
        raise ValueError(f"state {state_name} has duplicate signed factor paths: {sorted(duplicate_paths)}")
    ambiguous = set(positive) & set(negative)
    if ambiguous:
        raise ValueError(f"state {state_name} has ambiguous signed factor influence: {sorted(ambiguous)}")
    memo[state_name] = (positive, negative)
    return positive, negative


def _directional_support_set(mapping: dict[str, Any]) -> set[str]:
    return set(mapping["support_factors"]) | set(mapping["support_states"])


def _factor_schema_fingerprint(factors: list[dict[str, Any]]) -> str:
    normalized: list[dict[str, Any]] = []
    unordered_fields = {"reason_positive_anchors", "geometry_sources", "contradicts"}
    for factor in sorted(factors, key=lambda item: item["name"]):
        normalized.append(
            {
                key: sorted(value) if key in unordered_fields else value
                for key, value in factor.items()
            }
        )
    payload = json.dumps(normalized, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def _state_schema_fingerprint(states: dict[str, Any]) -> str:
    normalized = {
        state_name: {
            "required_groups": sorted(sorted(group["any_of"]) for group in spec["required_groups"]),
            "veto": sorted(spec["veto"]),
        }
        for state_name, spec in sorted(states.items())
    }
    payload = json.dumps(normalized, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def _reason_observation_fingerprint(reason_observation: dict[int, Any]) -> str:
    unordered_fields = {"support_factors", "contradiction_factors", "support_states", "visibility_factors"}
    normalized = [
        {
            "reason_id": reason_id,
            **{
                key: sorted(value) if key in unordered_fields else value
                for key, value in mapping.items()
            },
        }
        for reason_id, mapping in sorted(reason_observation.items())
    ]
    payload = json.dumps(normalized, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def validate_mosaic_schema_bundle(bundle: dict[str, Any]) -> None:
    expected_bundle_fields = {
        "label_schema",
        "reason_source_names",
        "factors",
        "states",
        "reason_observation",
    }
    if not isinstance(bundle, dict) or set(bundle) != expected_bundle_fields:
        if isinstance(bundle, dict) and set(bundle) - expected_bundle_fields:
            raise ValueError("schema bundle contains unknown top-level fields")
        raise ValueError("schema bundle must contain all MOSAIC documents")
    labels = bundle["label_schema"]
    factors = bundle["factors"]
    states = bundle["states"]
    reason_observation = bundle["reason_observation"]
    if not isinstance(labels, dict) or not isinstance(factors, list) or not isinstance(states, dict) or not isinstance(
        reason_observation, dict
    ):
        raise ValueError("schema bundle documents have invalid top-level types")
    if not isinstance(bundle["reason_source_names"], list) or len(bundle["reason_source_names"]) != 21:
        raise ValueError("schema bundle reason_source_names must be a 21-name list")

    if set(labels) != _LABEL_SCHEMA_FIELDS:
        raise ValueError("label schema has missing or unknown fields")
    if type(labels.get("action_dim")) is not int or type(labels.get("reason_dim")) is not int:
        raise ValueError("MOSAIC requires exact integer dimensions")
    if labels["action_dim"] != 4 or labels["reason_dim"] != 21:
        raise ValueError("MOSAIC requires action_dim=4 and reason_dim=21")
    if not isinstance(labels.get("actions"), list) or not isinstance(labels.get("reasons"), list):
        raise ValueError("label actions/reasons must be lists")
    for item in labels["actions"]:
        if not isinstance(item, dict) or set(item) != _ACTION_FIELDS:
            raise ValueError("action label has missing or unknown fields")
        if not isinstance(item["aliases"], list) or any(not isinstance(alias, str) or not alias for alias in item["aliases"]):
            raise ValueError("action aliases must be non-empty strings")
        _require_unique_values(item["aliases"], "action label has duplicate aliases")
        if not isinstance(item["group"], str) or not item["group"]:
            raise ValueError("action group must be a non-empty string")
    for item in labels["reasons"]:
        if not isinstance(item, dict) or set(item) != _REASON_LABEL_FIELDS:
            raise ValueError("reason label has missing or unknown fields")
    _validate_contiguous(labels.get("actions", []), 4, "actions")
    _validate_contiguous(labels.get("reasons", []), 21, "reasons")
    if tuple(item["name"] for item in labels["actions"]) != _EXPECTED_ACTION_NAMES:
        raise ValueError("action labels do not match official action names")
    if tuple(item["name"] for item in labels["reasons"]) != _EXPECTED_REASON_NAMES:
        raise ValueError("reason labels do not match official reason names")
    source_names = bundle.get("reason_source_names")
    label_reason_names = [str(item["name"]) for item in labels["reasons"]]
    if source_names != label_reason_names:
        raise ValueError("reason label names do not match the hashed source names")
    if tuple(source_names) != _EXPECTED_REASON_NAMES:
        raise ValueError("reason source does not match official reason names")
    semantics = labels.get("reason_semantics", {})
    if not isinstance(semantics, dict) or set(semantics) != _SEMANTIC_FIELDS:
        raise ValueError("reason semantics has missing or unknown fields")
    if semantics.get("raw_json_contains_names") is not False:
        raise ValueError("reason semantics must not claim names are stored in raw JSON")
    if semantics.get("official_mapping_verified") is not True:
        raise ValueError("official CVPR reason mapping must remain verified")
    if semantics.get("semantic_source") != "official_cvpr2020_table1_category_order":
        raise ValueError("reason semantic provenance is missing")
    if semantics.get("training_role") != "canonical_index_semantics_and_weak_factor_config":
        raise ValueError("official reason names must preserve the weak-factor training boundary")
    if semantics.get("source_file") != _EXPECTED_REASON_SOURCE_FILE:
        raise ValueError(f"reason source_file must be {_EXPECTED_REASON_SOURCE_FILE}")
    if not isinstance(semantics.get("source_sha256"), str) or semantics["source_sha256"] != _EXPECTED_REASON_SOURCE_SHA256:
        raise ValueError("reason source_sha256 must match the audited official mapping")

    if len(factors) != 24:
        raise ValueError(f"expected exactly 24 observable factors, got {len(factors)}")
    if any(not isinstance(factor, dict) for factor in factors):
        raise ValueError("factor entries must be mappings")
    factor_names = [factor.get("name") for factor in factors]
    if any(not isinstance(name, str) for name in factor_names):
        raise ValueError("factor names must be strings")
    if len(set(factor_names)) != 24 or any(not name for name in factor_names):
        raise ValueError("factor names must be non-empty and unique")
    if any(name != name.strip() for name in factor_names):
        raise ValueError("factor schema requires canonical factor names without surrounding whitespace")
    factor_name_set = set(factor_names)
    by_name = {factor["name"]: factor for factor in factors}
    for factor in factors:
        missing = _REQUIRED_FACTOR_FIELDS - set(factor)
        if missing:
            raise ValueError(f"factor {factor['name']} missing fields {sorted(missing)}")
        if set(factor) != _REQUIRED_FACTOR_FIELDS:
            raise ValueError(f"factor {factor['name']} has unknown fields")
        if not isinstance(factor["type"], str) or factor["type"] not in _FACTOR_TYPES:
            raise ValueError(f"factor {factor['name']} has invalid geometry type")
        if not isinstance(factor["entity"], str) or not factor["entity"].strip():
            raise ValueError(f"factor {factor['name']} has an empty entity")
        if factor["attribute"] is not None and (
            not isinstance(factor["attribute"], str) or not factor["attribute"].strip()
        ):
            raise ValueError(f"factor {factor['name']} has an invalid attribute")
        if type(factor["num_prototypes"]) is not int or factor["num_prototypes"] not in {2, 3, 4}:
            raise ValueError(f"factor {factor['name']} has invalid prototype count")
        if not isinstance(factor["spatial"], str) or factor["spatial"] not in _REGION_PRIORS:
            raise ValueError(f"factor {factor['name']} has invalid spatial region")
        if not isinstance(factor["region_prior"], str) or factor["region_prior"] not in _REGION_PRIORS:
            raise ValueError(f"factor {factor['name']} has invalid region prior")
        if not isinstance(factor["geometry_sources"], list) or not factor["geometry_sources"]:
            raise ValueError(f"factor {factor['name']} must have a non-empty geometry source list")
        if any(not isinstance(source, str) for source in factor["geometry_sources"]):
            raise ValueError(f"factor {factor['name']} geometry source strings are required")
        _require_unique_values(factor["geometry_sources"], f"factor {factor['name']} has duplicate geometry sources")
        if not set(factor["geometry_sources"]) <= _GEOMETRY_SOURCES:
            raise ValueError(f"factor {factor['name']} has invalid geometry source")
        if not set(factor["geometry_sources"]) <= _GEOMETRY_SOURCE_BY_TYPE[factor["type"]]:
            raise ValueError(f"factor {factor['name']} geometry source incompatible with factor type")
        if "none" in factor["geometry_sources"] and len(factor["geometry_sources"]) != 1:
            raise ValueError(f"factor {factor['name']} mixes none geometry source with annotations")
        if not isinstance(factor["reason_positive_anchors"], list) or any(
            type(index) is not int or index not in range(21) for index in factor["reason_positive_anchors"]
        ):
            raise ValueError(f"factor {factor['name']} has invalid reason anchors")
        _require_unique_values(
            factor["reason_positive_anchors"], f"factor {factor['name']} has duplicate reason anchors"
        )
        mirror = factor["mirror_partner"]
        if not isinstance(mirror, str) or not isinstance(factor["contradicts"], list):
            raise ValueError(f"factor {factor['name']} mirror/contradicts must use string/list metadata")
        if any(not isinstance(other, str) or not other for other in factor["contradicts"]):
            raise ValueError(f"factor {factor['name']} contradiction names must be non-empty strings")
        _require_unique_values(factor["contradicts"], f"factor {factor['name']} has duplicate contradiction names")
        if mirror not in factor_name_set or by_name[mirror]["mirror_partner"] != factor["name"]:
            raise ValueError(f"factor {factor['name']} has an asymmetric mirror link")
        for other in factor["contradicts"]:
            if other == factor["name"]:
                raise ValueError(f"factor {factor['name']} cannot contradict itself")
            if other not in factor_name_set or factor["name"] not in by_name[other]["contradicts"]:
                raise ValueError(f"factor {factor['name']} has an invalid contradiction link")
            shared_anchors = set(factor["reason_positive_anchors"]) & set(by_name[other]["reason_positive_anchors"])
            if shared_anchors:
                raise ValueError(
                    f"contradictory factors share reason anchors {sorted(shared_anchors)}: {factor['name']} and {other}"
                )
    if set(states) != _EXPECTED_STATES:
        raise ValueError(f"decision state names must be exactly {sorted(_EXPECTED_STATES)}")
    state_names = set(states)
    if factor_name_set & state_names:
        raise ValueError(f"factor and state namespaces overlap: {sorted(factor_name_set & state_names)}")
    state_dependencies = _state_dependencies(states, factor_name_set)
    _assert_acyclic(state_dependencies)
    state_factor_effects = {
        state_name: _state_signed_factor_closure(state_name, states, factor_name_set, {}) for state_name in states
    }

    if set(reason_observation) != set(range(21)):
        raise ValueError("reason observation mapping must cover indices 0..20")
    for reason_id, mapping in reason_observation.items():
        if not isinstance(mapping, dict) or not _REQUIRED_REASON_FIELDS <= set(mapping):
            raise ValueError(f"reason {reason_id} has missing fields")
        if set(mapping) != _REQUIRED_REASON_FIELDS:
            raise ValueError(f"reason {reason_id} has unknown fields")
        if not isinstance(mapping.get("group"), str) or mapping["group"] not in {
            "traffic_control",
            "obstacle",
            "lane",
            "other",
        }:
            raise ValueError(f"reason {reason_id} has an invalid group")
        if not isinstance(mapping["support_semantics"], str) or mapping["support_semantics"] not in {
            "direct_observable",
            "weak_proxy",
        }:
            raise ValueError(f"reason {reason_id} has invalid support semantics")
        list_fields = ("support_factors", "support_states", "visibility_factors", "contradiction_factors")
        if any(
            not isinstance(mapping[field], list)
            or any(not isinstance(item, str) or not item for item in mapping[field])
            for field in list_fields
        ):
            raise ValueError(f"reason {reason_id} list fields must contain non-empty string references")
        for field in list_fields:
            _require_unique_values(mapping[field], f"reason {reason_id} has duplicate reason references in {field}")
        support_factors = set(mapping["support_factors"])
        support_states = set(mapping["support_states"])
        visibility_factors = set(mapping["visibility_factors"])
        contradiction_factors = set(mapping["contradiction_factors"])
        if not support_factors and not support_states:
            raise ValueError(f"reason {reason_id} has no support")
        if not visibility_factors:
            raise ValueError(f"reason {reason_id} has no visibility factor")
        if not support_factors <= factor_name_set or not visibility_factors <= factor_name_set:
            raise ValueError(f"reason {reason_id} references an unknown support/visibility factor")
        if not contradiction_factors <= factor_name_set or not support_states <= state_names:
            raise ValueError(f"reason {reason_id} references an unknown contradiction/state")
        overlap = support_factors & contradiction_factors
        if overlap:
            raise ValueError(f"reason {reason_id} uses {sorted(overlap)} as both support and contradiction")
        if mapping["support_semantics"] == "direct_observable" and support_states:
            raise ValueError(f"reason {reason_id} direct observable support cannot use decision states")
        if mapping["support_semantics"] == "direct_observable" and any(
            reason_id not in by_name[factor_name]["reason_positive_anchors"] for factor_name in support_factors
        ):
            raise ValueError(f"reason {reason_id} direct observable support uses a non-specific factor")
        state_positive_factors: set[str] = set()
        state_negative_factors: set[str] = set()
        for state_name in support_states:
            state_positive_paths, state_negative_paths = state_factor_effects[state_name]
            state_positive = set(state_positive_paths)
            state_negative = set(state_negative_paths)
            duplicate_positive = state_positive_factors & state_positive
            duplicate_negative = state_negative_factors & state_negative
            if duplicate_positive or duplicate_negative:
                raise ValueError(
                    f"reason {reason_id} duplicates factor support through states: "
                    f"{sorted(duplicate_positive | duplicate_negative)}"
                )
            conflicting = (state_positive_factors & state_negative) | (state_negative_factors & state_positive)
            if conflicting:
                raise ValueError(f"reason {reason_id} has conflicting state factor influence: {sorted(conflicting)}")
            state_positive_factors.update(state_positive)
            state_negative_factors.update(state_negative)
        duplicated_support = support_factors & state_positive_factors
        if duplicated_support:
            raise ValueError(
                f"reason {reason_id} duplicates factor support through state: {sorted(duplicated_support)}"
            )
        duplicated_contradiction = contradiction_factors & state_negative_factors
        if duplicated_contradiction:
            raise ValueError(
                f"reason {reason_id} duplicates contradiction through state veto: {sorted(duplicated_contradiction)}"
            )
        conflicting_direct = (support_factors & state_negative_factors) | (
            contradiction_factors & state_positive_factors
        )
        if conflicting_direct:
            raise ValueError(f"reason {reason_id} has conflicting direct/state evidence: {sorted(conflicting_direct)}")
        positive_evidence = support_factors | state_positive_factors
        contradictory_positive_pairs = {
            tuple(sorted((factor_name, other)))
            for factor_name in positive_evidence
            for other in by_name[factor_name]["contradicts"]
            if other in positive_evidence
        }
        if contradictory_positive_pairs:
            raise ValueError(
                f"reason {reason_id} has mutually contradictory support factors: "
                f"{sorted(contradictory_positive_pairs)}"
            )
        raw_false_positive_max = mapping["false_positive_max"]
        if type(raw_false_positive_max) not in {int, float} or not math.isfinite(raw_false_positive_max):
            raise ValueError(f"reason {reason_id} requires numeric false_positive_max")
        false_positive_max = float(raw_false_positive_max)
        if not 0.0 <= false_positive_max <= 0.05:
            raise ValueError(f"reason {reason_id} has an invalid false-positive bound")

    for factor in factors:
        for reason_id in factor["reason_positive_anchors"]:
            mapping = reason_observation[reason_id]
            if factor["name"] not in set(mapping["support_factors"]):
                raise ValueError(
                    f"factor {factor['name']} has a non-reciprocal reason anchor {reason_id}"
                )

    for left_reason, right_reason in _DIRECTIONAL_REASON_PAIRS:
        left_refs = _directional_support_set(reason_observation[left_reason])
        right_refs = _directional_support_set(reason_observation[right_reason])
        required_left, required_right = _DIRECTIONAL_REQUIRED_EVIDENCE[(left_reason, right_reason)]
        if not required_left <= left_refs or not required_right <= right_refs:
            raise ValueError(f"directional reason pair {left_reason}/{right_reason} lacks required directional evidence")
        if left_refs == right_refs:
            raise ValueError(f"directional reason pair {left_reason}/{right_reason} has identical evidence")
        mirrored = any(
            (
                reference in by_name
                and by_name[reference]["mirror_partner"] != reference
                and by_name[reference]["mirror_partner"] in right_refs
            )
            or _STATE_MIRRORS.get(reference) in right_refs
            for reference in left_refs
        )
        if not mirrored:
            raise ValueError(f"directional reason pair {left_reason}/{right_reason} lacks mirrored evidence")

    if _factor_schema_fingerprint(factors) != _EXPECTED_FACTOR_SCHEMA_SHA256:
        raise ValueError("observable factors do not match the canonical factor schema")
    if _state_schema_fingerprint(states) != _EXPECTED_STATE_SCHEMA_SHA256:
        raise ValueError("decision states do not match the canonical decision state schema")
    if _reason_observation_fingerprint(reason_observation) != _EXPECTED_REASON_OBSERVATION_SHA256:
        raise ValueError("reason observations do not match the canonical reason observation schema")


def load_mosaic_schema_bundle(config_root: str | Path) -> dict[str, Any]:
    config_root = Path(config_root)
    labels = _load_yaml(config_root / "mosaic_label_schema.yaml")
    factors_doc = _load_yaml(config_root / "mosaic_observable_factors.yaml")
    states_doc = _load_yaml(config_root / "mosaic_decision_states.yaml")
    reasons_doc = _load_yaml(config_root / "mosaic_reason_observation.yaml")
    if not isinstance(labels, dict) or not isinstance(labels.get("reason_semantics"), dict):
        raise ValueError("mosaic_label_schema.yaml has an invalid reason_semantics section")
    if labels["reason_semantics"].get("source_file") != _EXPECTED_REASON_SOURCE_FILE:
        raise ValueError(f"reason source_file must be {_EXPECTED_REASON_SOURCE_FILE}")
    if labels["reason_semantics"].get("source_sha256") != _EXPECTED_REASON_SOURCE_SHA256:
        raise ValueError("reason source_sha256 must match the audited external mapping")
    factors = _require_document(factors_doc, "factors", list, "mosaic_observable_factors.yaml")
    states = _require_document(states_doc, "states", dict, "mosaic_decision_states.yaml")
    reasons = _require_document(reasons_doc, "reasons", dict, "mosaic_reason_observation.yaml")
    source_path = config_root.parent / _EXPECTED_REASON_SOURCE_FILE
    source_bytes = source_path.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest().upper()
    if source_hash != labels["reason_semantics"]["source_sha256"]:
        raise ValueError("reason semantic source hash does not match mosaic_label_schema.yaml")
    source_doc = yaml.load(source_bytes.decode("utf-8"), Loader=_UniqueKeyLoader)
    if (
        not isinstance(source_doc, dict)
        or set(source_doc) != {"provenance", "names"}
        or not isinstance(source_doc.get("provenance"), dict)
        or not isinstance(source_doc.get("names"), dict)
    ):
        raise ValueError("reason semantic source must contain provenance and names mappings")
    provenance = source_doc["provenance"]
    if (
        provenance.get("paper_url") != _EXPECTED_REASON_PAPER_URL
        or type(provenance.get("table")) is not int
        or provenance["table"] != 1
        or provenance.get("category_source") != "official_paper_table"
        or provenance.get("local_raw_json_contains_names") is not False
    ):
        raise ValueError("reason semantic source has invalid official CVPR provenance")
    source_name_mapping = _normalize_reason_mapping(source_doc["names"])
    if set(source_name_mapping) != set(range(21)) or any(
        not isinstance(source_name_mapping[index], str) or not source_name_mapping[index]
        for index in range(21)
    ):
        raise ValueError("reason semantic source names must cover 0..20 with strings")
    source_names = [source_name_mapping[index] for index in range(21)]
    bundle = {
        "label_schema": labels,
        "reason_source_names": source_names,
        "factors": factors,
        "states": states,
        "reason_observation": _normalize_reason_mapping(reasons),
    }
    validate_mosaic_schema_bundle(bundle)
    return bundle


# IC-DOR deliberately uses an ontology that is separate from the legacy
# decision-state bundle.  The legacy bundle contains state-to-action semantics;
# this loader accepts only observable factors and target-owned candidate routes.
_ICDOR_REQUIRED_FACTOR_FIELDS = {
    "name",
    "type",
    "num_prototypes",
    "weak_regions",
    "mirror_of",
    "contradicts",
    "positive_reason_anchors",
    "grounding_sources",
}
_ICDOR_OPTIONAL_FACTOR_FIELDS = {
    "role", "visual_sources", "attribute_constraints", "negative_policy", "source_kind", "observable",
}
_ICDOR_REASON_ROUTE_FIELDS = {
    "group",
    "direct_factors",
    "latent_factors",
    "contradiction_factors",
    "escape_allowed",
}
_ICDOR_OPTIONAL_REASON_ROUTE_FIELDS = {"absence_factors", "semantic_kind"}
_ICDOR_ALLOWED_GROUNDING_SOURCES = {
    "box2d", "lane_polyline", "drivable_mask", "bdd100k_attributes", "corridor_overlap", "image_only"
}
_ICDOR_FORBIDDEN_FACTOR_NAMES = {
    "no_left_lane",
    "no_right_lane",
    "left_turn_allowed",
    "right_turn_allowed",
    "forward_feasible",
    "stop_obligation",
    "lane_follow_permitted",
}


def _icdor_name_list(value: Any, *, field: str, factor_names: set[str]) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(name, str) or name not in factor_names for name in value):
        raise ValueError(f"IC-DOR {field} must be a list of known factor names")
    if len(set(value)) != len(value):
        raise ValueError(f"IC-DOR {field} must not contain duplicate factors")
    return list(value)


def _icdor_numeric(mapping: dict[str, Any], key: str, expected: float) -> None:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) != expected:
        raise ValueError(f"IC-DOR certificate rule {key} must equal {expected}")


def load_icdor_ontology(config_root: str | Path) -> dict[str, Any]:
    """Load the IC-DOR observable-factor and target-route ontology.

    This parser intentionally does not call ``load_mosaic_schema_bundle``:
    that legacy helper validates decision states, while IC-DOR must prevent
    decision-state semantics from becoming an input to its action route.
    """

    config_root = Path(config_root)
    labels = _load_yaml(config_root / "mosaic_label_schema.yaml")
    factors_doc = _load_yaml(config_root / "mosaic_icdor_factor_candidates.yaml")
    action_routes_doc = _load_yaml(config_root / "mosaic_icdor_action_routes.yaml")
    reason_routes_doc = _load_yaml(config_root / "mosaic_icdor_reason_routes.yaml")
    certificate_doc = _load_yaml(config_root / "mosaic_icdor_certificate_rules.yaml")

    if not isinstance(labels, dict) or not _LABEL_SCHEMA_FIELDS <= set(labels):
        raise ValueError("mosaic_label_schema.yaml is missing the IC-DOR label schema fields")
    if labels["action_dim"] != 4 or labels["reason_dim"] != 21:
        raise ValueError("IC-DOR requires exactly four actions and 21 reasons")
    actions = labels["actions"]
    reasons = labels["reasons"]
    if not isinstance(actions, list) or not isinstance(reasons, list):
        raise ValueError("IC-DOR label actions/reasons must be lists")
    _validate_contiguous(actions, 4, "actions")
    _validate_contiguous(reasons, 21, "reasons")
    action_names = tuple(item["name"].strip() for item in actions)
    reason_names = tuple(item["name"].strip() for item in reasons)
    if action_names != _EXPECTED_ACTION_NAMES or reason_names != _EXPECTED_REASON_NAMES:
        raise ValueError("IC-DOR requires the audited BDD-OIA action/reason order")

    factors = _require_document(factors_doc, "factors", list, "mosaic_icdor_factor_candidates.yaml")
    if len(factors) < 20:
        raise ValueError("IC-DOR requires a non-trivial observable factor inventory")
    factor_names: list[str] = []
    for factor in factors:
        if not isinstance(factor, dict) or not _ICDOR_REQUIRED_FACTOR_FIELDS <= set(factor):
            raise ValueError("IC-DOR factor candidates must use the complete observable-factor schema")
        if set(factor) - (_ICDOR_REQUIRED_FACTOR_FIELDS | _ICDOR_OPTIONAL_FACTOR_FIELDS):
            raise ValueError("IC-DOR factor candidates contain unknown fields")
        factor.setdefault("role", "observable")
        factor.setdefault("visual_sources", list(factor["grounding_sources"]))
        factor.setdefault("attribute_constraints", {})
        factor.setdefault("negative_policy", "unknown_if_source_incomplete")
        factor.setdefault("source_kind", "grounded" if "image_only" not in factor["grounding_sources"] else "image_only")
        factor.setdefault("observable", factor["role"] == "observable")
        name = factor["name"]
        if not isinstance(name, str) or not name or name in _ICDOR_FORBIDDEN_FACTOR_NAMES or "state" in name:
            raise ValueError(f"IC-DOR factor {name!r} is not an observable atomic factor")
        if factor["type"] not in _FACTOR_TYPES:
            raise ValueError(f"IC-DOR factor {name} has invalid type")
        if type(factor["num_prototypes"]) is not int or factor["num_prototypes"] < 2:
            raise ValueError(f"IC-DOR factor {name} needs at least two independent prototypes")
        if not isinstance(factor["weak_regions"], list) or not factor["weak_regions"]:
            raise ValueError(f"IC-DOR factor {name} requires at least one weak region")
        if any(region not in _REGION_PRIORS for region in factor["weak_regions"]):
            raise ValueError(f"IC-DOR factor {name} has an invalid weak region")
        if not isinstance(factor["mirror_of"], str) or not factor["mirror_of"]:
            raise ValueError(f"IC-DOR factor {name} requires mirror_of")
        if not isinstance(factor["contradicts"], list) or not all(isinstance(item, str) for item in factor["contradicts"]):
            raise ValueError(f"IC-DOR factor {name} has invalid contradictions")
        if not isinstance(factor["positive_reason_anchors"], list) or any(
            type(reason_id) is not int or reason_id not in range(21)
            for reason_id in factor["positive_reason_anchors"]
        ):
            raise ValueError(f"IC-DOR factor {name} has invalid positive reason anchors")
        if not isinstance(factor["grounding_sources"], list) or not factor["grounding_sources"]:
            raise ValueError(f"IC-DOR factor {name} requires declared grounding sources")
        if not set(factor["grounding_sources"]) <= _ICDOR_ALLOWED_GROUNDING_SOURCES:
            raise ValueError(f"IC-DOR factor {name} has an invalid grounding source")
        factor_names.append(name)
    _require_unique_values(factor_names, "IC-DOR factor names must be unique")
    factor_name_set = set(factor_names)
    by_factor = {factor["name"]: factor for factor in factors}
    for factor in factors:
        mirror = factor["mirror_of"]
        if mirror not in factor_name_set or by_factor[mirror]["mirror_of"] != factor["name"]:
            raise ValueError(f"IC-DOR factor {factor['name']} has a non-reciprocal mirror")
        for contradicted in factor["contradicts"]:
            if contradicted not in factor_name_set or factor["name"] not in by_factor[contradicted]["contradicts"]:
                raise ValueError(f"IC-DOR factor {factor['name']} has a non-reciprocal contradiction")

    action_routes = _require_document(action_routes_doc, "action_routes", dict, "mosaic_icdor_action_routes.yaml")
    if set(action_routes) != set(action_names):
        raise ValueError("IC-DOR action routes must cover exactly the four official action names")
    normalized_action_routes: dict[str, dict[str, list[dict[str, str]]]] = {}
    for action_name, directions in action_routes.items():
        if not isinstance(directions, dict) or set(directions) != {"support", "veto"}:
            raise ValueError(f"IC-DOR action route {action_name} must separate support and veto")
        normalized_action_routes[action_name] = {}
        for direction, edges in directions.items():
            if not isinstance(edges, list):
                raise ValueError(f"IC-DOR {action_name}/{direction} edges must be a list")
            normalized_edges: list[dict[str, str]] = []
            for edge in edges:
                if not isinstance(edge, dict) or set(edge) != {"factor", "polarity"}:
                    raise ValueError("IC-DOR routes are candidate semantics, not fixed weights")
                factor_name = edge["factor"]
                polarity = edge["polarity"]
                if factor_name not in factor_name_set or polarity not in {"present", "absent"}:
                    raise ValueError("IC-DOR action route uses an unknown factor or polarity")
                normalized_edges.append({"factor": factor_name, "polarity": polarity})
            normalized_action_routes[action_name][direction] = normalized_edges

    reason_routes_raw = _require_document(reason_routes_doc, "reason_routes", dict, "mosaic_icdor_reason_routes.yaml")
    reason_routes = _normalize_reason_mapping(reason_routes_raw)
    if set(reason_routes) != set(range(21)):
        raise ValueError("IC-DOR reason routes must cover exactly the 21 official reason labels")
    for reason_id, route in reason_routes.items():
        if not isinstance(route, dict) or not _ICDOR_REASON_ROUTE_FIELDS <= set(route):
            raise ValueError(f"IC-DOR reason route {reason_id} has an invalid schema")
        if set(route) - (_ICDOR_REASON_ROUTE_FIELDS | _ICDOR_OPTIONAL_REASON_ROUTE_FIELDS):
            raise ValueError(f"IC-DOR reason route {reason_id} has unknown fields")
        route.setdefault("absence_factors", [])
        route.setdefault("semantic_kind", "observable_or_latent")
        if not isinstance(route["group"], str) or not route["group"]:
            raise ValueError(f"IC-DOR reason route {reason_id} requires a semantic group")
        for field in ("direct_factors", "latent_factors", "contradiction_factors"):
            route[field] = _icdor_name_list(route[field], field=field, factor_names=factor_name_set)
        route["absence_factors"] = _icdor_name_list(route["absence_factors"], field="absence_factors", factor_names=factor_name_set)
        if type(route["escape_allowed"]) is not bool:
            raise ValueError(f"IC-DOR reason route {reason_id} escape_allowed must be boolean")
        if any("state" in name for name in route["latent_factors"]):
            raise ValueError("IC-DOR latent routes must not reference decision states")

    certificate_rules = _require_document(
        certificate_doc, "certificate_rules", dict, "mosaic_icdor_certificate_rules.yaml"
    )
    if set(certificate_rules) != {"version", "certified", "reason_only", "abstained"}:
        raise ValueError("IC-DOR certificate rules have an invalid top-level schema")
    if certificate_rules["version"] != "icdor_v3":
        raise ValueError("IC-DOR certificate version must be icdor_v3")
    certified = certificate_rules["certified"]
    if not isinstance(certified, dict):
        raise ValueError("IC-DOR certified rules must be a mapping")
    expected_certified = {
        "min_confirmed_positive": 32.0,
        "min_reliable_negative": 32.0,
        "min_geometry_valid": 200.0,
        "min_full_minus_prior_lcb95": 0.02,
        "min_content_fraction": 0.70,
        "min_query_shuffle_drop_lcb95": 0.01,
        "min_image_shuffle_drop_lcb95": 0.01,
        "min_grounding_minus_random_lcb95": 0.02,
        "min_effective_prototype_count": 1.5,
        "max_dominant_prototype_rate": 0.85,
    }
    if set(certified) != set(expected_certified) | {"require_nonzero_presence_visibility_variance"}:
        raise ValueError("IC-DOR certified rules must use the complete plan thresholds")
    for key, expected in expected_certified.items():
        _icdor_numeric(certified, key, expected)
    if certified["require_nonzero_presence_visibility_variance"] is not True:
        raise ValueError("IC-DOR certificate must require non-degenerate presence/visibility")
    for tier_name, expected_fields in {
        "reason_only": {"require_stable_content_shuffle", "allow_action_route"},
        "abstained": {"allow_explicit_routes", "allow_escape_token"},
    }.items():
        tier = certificate_rules[tier_name]
        if not isinstance(tier, dict) or set(tier) != expected_fields:
            raise ValueError(f"IC-DOR {tier_name} certificate rules are incomplete")
        if any(type(value) is not bool for value in tier.values()):
            raise ValueError(f"IC-DOR {tier_name} certificate rules must be boolean")
    if certificate_rules["reason_only"]["allow_action_route"]:
        raise ValueError("reason_only factors must never enter the action route")
    if certificate_rules["abstained"]["allow_explicit_routes"]:
        raise ValueError("abstained factors must not enter explicit routes")

    return {
        "action_names": action_names,
        "reason_names": reason_names,
        "action_index": {name: index for index, name in enumerate(action_names)},
        "reason_index": {name: index for index, name in enumerate(reason_names)},
        "factors": factors,
        "factor_index": {name: index for index, name in enumerate(factor_names)},
        "action_routes": normalized_action_routes,
        "reason_routes": reason_routes,
        "certificate_rules": certificate_rules,
    }
