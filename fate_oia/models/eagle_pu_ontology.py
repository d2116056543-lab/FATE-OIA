from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_eagle_pu_ontology(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "actions" not in data or "reasons" not in data or "groups" not in data:
        raise ValueError("Ontology must contain actions, reasons, and groups")
    if sorted(int(k) for k in data["actions"].keys()) != list(range(4)):
        raise ValueError("actions must be indexed 0..3")
    if sorted(int(k) for k in data["reasons"].keys()) != list(range(21)):
        raise ValueError("reasons must be indexed 0..20")
    for idx, rec in data["reasons"].items():
        if str(rec.get("name", "")).lower().startswith("reason_"):
            raise ValueError(f"placeholder reason name forbidden at {idx}")
        for key in ["group", "positive_states", "negative_states", "compatible_actions", "hard_negatives", "spatial_prior"]:
            if key not in rec:
                raise ValueError(f"reason {idx} missing {key}")
    return data
