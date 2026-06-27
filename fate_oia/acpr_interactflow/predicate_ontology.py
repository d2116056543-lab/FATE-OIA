from __future__ import annotations

from pathlib import Path

import yaml


class InteractPredicateOntology:
    def __init__(self, predicate_config: str | Path, fallback_oia_config: str | Path = "configs/acpr_scene_predicates.yaml") -> None:
        self.path = Path(predicate_config)
        data = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        oia_path = Path(data.get("oia_predicate_config") or fallback_oia_config)
        oia_data = yaml.safe_load(oia_path.read_text(encoding="utf-8")) if oia_path.exists() else {"predicates": []}
        oia_predicates = list((oia_data or {}).get("predicates", []))[:32]
        psi_predicates = list(data.get("psi_predicates", []))
        if len(oia_predicates) < 32:
            raise ValueError("InteractFlow requires 32 OIA predicates for transfer")
        if len(psi_predicates) != 16:
            raise ValueError("InteractFlow requires exactly 16 PSI predicates")
        self.predicates = oia_predicates + psi_predicates
        self.names = [str(p["name"]) for p in self.predicates]
        if len(self.names) != 48:
            raise ValueError(f"Expected 48 predicates, got {len(self.names)}")

    def groups(self) -> list[str]:
        return [str(p.get("group", p.get("region", "global"))) for p in self.predicates]

