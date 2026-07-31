from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml


class METERFactorSchema:
    """Validated runtime source for HECA measurement identity and ownership."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        data = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        rows = list(data.get("factors", []))
        if [int(row.get("id", -1)) for row in rows] != list(range(21)):
            raise ValueError("METER factor schema must contain ordered IDs 0..20")
        required = {
            "name",
            "factor_group",
            "text_prompt",
            "state_set",
            "state_prompts",
            "groundability",
            "action_owned",
            "mirror_partner",
        }
        if any(not required.issubset(row) for row in rows):
            raise ValueError("METER factor schema is missing a required field")
        action_names = tuple(
            str(name)
            for name in data.get(
                "action_names", ("forward", "stop", "left", "right")
            )
        )
        if action_names != ("forward", "stop", "left", "right"):
            raise ValueError("METER action_names must be forward, stop, left, right")
        if any("compatible_actions" in row for row in rows):
            raise ValueError("HECA forbids hard compatible_actions masks")
        if any(len(row["state_set"]) != len(row["state_prompts"]) for row in rows):
            raise ValueError("Every HECA state requires one ontology prompt")
        self.rows: tuple[dict[str, Any], ...] = tuple(rows)
        self.action_names = action_names
        self.state_cardinalities = tuple(len(row["state_set"]) for row in rows)
        self.action_ownership = tuple(float(row["action_owned"]) for row in rows)
        if self.action_ownership[14] != 0.0 or self.action_ownership[20] != 0.0:
            raise ValueError("Latent factors 14/20 must be action-owned zero")
        if self.action_ownership[1] != 0.5:
            raise ValueError("Factor 1 must use partial action ownership 0.5")
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
