from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REASON_KEYS = {"id", "name", "entity", "state", "sector", "decision_role", "allowed_evidence_families", "explicit_certifiable", "mirror_partner"}
FIELD_KEYS = {"name", "family", "sector", "part_type", "num_parts", "supervision_sources", "geometry_required", "minimum_coverage"}


def _load(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"PRECISE schema must be a mapping: {path}")
    return loaded


def load_reason_semantics(path: str | Path) -> list[dict[str, Any]]:
    rows = _load(path).get("reasons")
    if not isinstance(rows, list) or len(rows) != 21:
        raise ValueError("PRECISE reason semantics must contain exactly 21 rows")
    result = [dict(row) for row in rows]
    if [row.get("id") for row in result] != list(range(21)):
        raise ValueError("PRECISE reason ids must be contiguous 0..20")
    for row in result:
        if not REASON_KEYS.issubset(row) or not isinstance(row["name"], str) or not row["name"]:
            raise ValueError("PRECISE reason semantics is missing a required field")
        if row["name"].lower().startswith("reason_"):
            raise ValueError("PRECISE reason semantics cannot use placeholder names")
        if not isinstance(row["allowed_evidence_families"], list):
            raise ValueError("allowed_evidence_families must be a list")
    partner = {row["id"]: row["mirror_partner"] for row in result}
    for left, right in ((9, 15), (10, 16), (11, 17), (12, 18), (13, 19), (14, 20)):
        if partner[left] != right or partner[right] != left:
            raise ValueError("PRECISE reason mirror pairs are inconsistent")
    return result


def load_action_semantics(path: str | Path) -> list[dict[str, Any]]:
    rows = _load(path).get("actions")
    expected = [(0, "forward"), (1, "stop"), (2, "left"), (3, "right")]
    if not isinstance(rows, list) or [(row.get("id"), row.get("name")) for row in rows] != expected:
        raise ValueError("PRECISE action semantics must be forward/stop/left/right")
    if rows[2].get("query_base") != "side_shared" or rows[3].get("query_base") != "side_shared":
        raise ValueError("PRECISE side actions must share a query base")
    return [dict(row) for row in rows]


def load_evidence_fields(path: str | Path) -> list[dict[str, Any]]:
    payload = _load(path)
    fields = payload.get("explicit_fields")
    expected = ["traffic_light", "traffic_sign", "actor_left", "actor_center", "actor_right", "drivable_left", "drivable_center", "drivable_right", "boundary_left", "boundary_right"]
    if not isinstance(fields, list) or [row.get("name") for row in fields] != expected:
        raise ValueError("PRECISE evidence fields do not match the fixed initial schema")
    result = [dict(row) for row in fields]
    for row in result:
        if not FIELD_KEYS.issubset(row):
            raise ValueError("PRECISE evidence field is missing required metadata")
        if row["part_type"] == "ordered_curve" and row["num_parts"] != 8:
            raise ValueError("PRECISE boundary curves require 8 ordered parts")
        if row["family"] in {"traffic_control", "actor"} and row["num_parts"] != 4:
            raise ValueError("PRECISE point fields require 4 parts")
        if row["family"] == "drivable" and row["num_parts"] != 8:
            raise ValueError("PRECISE drivable fields require 8 parts")
    latent = payload.get("latent_slots", {})
    if latent.get("count") != 6 or latent.get("certifiable") is not False or latent.get("name") is not None:
        raise ValueError("PRECISE latent slots must be unnamed, non-certifiable six-slot evidence")
    return result


def field_schema_hash_payload(fields: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: row[key] for key in sorted(row)} for row in fields]
