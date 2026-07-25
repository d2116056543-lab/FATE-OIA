"""Behavioral contract tests for the P0 RAEL semantic reason ontology."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml
from yaml.constructor import ConstructorError


EXPECTED_NAMES = [
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
]
ROLES = {
    "forward_support",
    "forward_veto",
    "stop_support",
    "stop_veto",
    "left_support",
    "left_veto",
    "right_support",
    "right_veto",
    "neutral_context",
}
REQUIRED_FIELDS = {
    "id",
    "name",
    "entity",
    "state",
    "sector",
    "role",
    "mirror_partner",
    "explicit_evidence_families",
    "pu_eligible",
}
PLACEHOLDER = re.compile(r"^(reason[_ -]?\d+|unknown|tbd|placeholder)$", re.IGNORECASE)


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects silent duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key ({key!r})",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_schema() -> dict[str, Any]:
    path = _root() / "configs/rael_reason_semantics.yaml"
    assert path.is_file(), f"P0 semantic reason schema is missing: {path.as_posix()}"
    payload = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    assert isinstance(payload, dict), "reason schema must be a YAML mapping"
    return payload


def test_reason_schema_is_complete_named_and_soft_prior_only() -> None:
    schema = _load_schema()
    assert schema["schema_version"] == 1
    assert schema["role_application"] == "soft_prior_only"
    assert schema["hard_action_masks"] is False
    rows = schema["reasons"]
    assert isinstance(rows, list) and len(rows) == 21
    assert {row["id"] for row in rows} == set(range(21))
    assert [row["name"] for row in rows] == EXPECTED_NAMES
    assert {row["role"] for row in rows} == ROLES
    for row in rows:
        assert REQUIRED_FIELDS <= set(row), f"reason {row.get('id')} is missing a required field"
        assert isinstance(row["name"], str) and row["name"].strip()
        assert not PLACEHOLDER.fullmatch(row["name"].strip())
        assert isinstance(row["entity"], str) and row["entity"]
        assert isinstance(row["state"], str) and row["state"]
        assert isinstance(row["sector"], str) and row["sector"]
        assert row["role"] in ROLES
        assert isinstance(row["explicit_evidence_families"], list)
        assert row["explicit_evidence_families"]
        assert len(row["explicit_evidence_families"]) == len(set(row["explicit_evidence_families"]))
        assert isinstance(row["pu_eligible"], bool)
        assert "soft_action_prior" not in row


def test_reason_mirrors_are_symmetric_and_never_hard_code_action_compatibility() -> None:
    rows = _load_schema()["reasons"]
    by_id = {row["id"]: row for row in rows}
    for row in rows:
        partner = row["mirror_partner"]
        if partner is None:
            continue
        assert isinstance(partner, int) and partner in by_id
        assert partner != row["id"]
        assert by_id[partner]["mirror_partner"] == row["id"]
    expected_mirrors = {
        9: (15, "left_corridor", "right_corridor", "left_veto", "right_veto"),
        10: (16, "left_corridor", "right_corridor", "left_veto", "right_veto"),
        11: (17, "left_corridor", "right_corridor", "left_veto", "right_veto"),
        12: (18, "left_corridor", "right_corridor", "left_support", "right_support"),
        13: (19, "upper_control", "upper_control", "left_support", "right_support"),
        14: (20, "front_center", "front_center", "left_veto", "right_veto"),
    }
    for left_id, (right_id, left_sector, right_sector, left_role, right_role) in expected_mirrors.items():
        assert by_id[left_id]["mirror_partner"] == right_id
        assert by_id[left_id]["sector"] == left_sector
        assert by_id[right_id]["sector"] == right_sector
        assert by_id[left_id]["role"] == left_role
        assert by_id[right_id]["role"] == right_role
    serialized = yaml.safe_dump({"reasons": rows}, sort_keys=True).lower()
    assert "compatible_actions" not in serialized
    assert "hard_negative_reasons" not in serialized
    assert "soft_action_prior" not in serialized


def test_reason_schema_loader_rejects_duplicate_keys(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate_reason_schema.yaml"
    duplicate.write_text("reason: first\nreason: second\n", encoding="utf-8")
    with pytest.raises(ConstructorError, match="duplicate key"):
        yaml.load(duplicate.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
