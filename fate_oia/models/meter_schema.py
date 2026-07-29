from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml


class METERFactorSchema:
    """Validated runtime source for TESA factor ownership and grounding."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        data = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        rows = list(data.get("factors", []))
        if [int(row.get("id", -1)) for row in rows] != list(range(21)):
            raise ValueError("METER factor schema must contain ordered IDs 0..20")
        required = {"name", "state_set", "groundability", "action_owned", "mirror_partner"}
        if any(not required.issubset(row) for row in rows):
            raise ValueError("METER factor schema is missing a required field")
        self.rows: tuple[dict[str, Any], ...] = tuple(rows)
        self.state_cardinalities = tuple(len(row["state_set"]) for row in rows)
        self.action_ownership = tuple(float(row["action_owned"]) for row in rows)
        self.groundable_mask = tuple(
            0.0 if str(row["groundability"]).lower() in {"none", "latent", "unavailable"} else 1.0
            for row in rows
        )
        self.mirror_pairs = tuple(
            (int(row["id"]), int(row["mirror_partner"]))
            for row in rows
            if row["mirror_partner"] is not None and int(row["id"]) < int(row["mirror_partner"])
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.path.read_bytes()).hexdigest()


def default_meter_factor_schema() -> METERFactorSchema:
    path = Path(__file__).resolve().parents[2] / "configs" / "meter_factor_schema.yaml"
    return METERFactorSchema(path)
