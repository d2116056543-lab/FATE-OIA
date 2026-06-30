from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .acpr_ntmcal_text_atoms import NativePredicateSpec, build_atom_vocab


REQUIRED_NAMES = {
    "traffic_light_red", "traffic_light_green", "traffic_light_visible", "stop_sign_present",
    "front_vehicle_close", "pedestrian_front", "cyclist_front", "obstacle_front", "road_clear",
    "lane_left_available", "lane_right_available", "left_solid_boundary", "right_solid_boundary",
    "lane_absent_left", "lane_absent_right", "open_left_gap", "open_right_gap",
    "drivable_center", "drivable_left", "drivable_right",
}


class NativePredicateBank:
    def __init__(self, specs: list[NativePredicateSpec]) -> None:
        self.specs = sorted(specs, key=lambda s: s.id)
        self.names = [s.name for s in self.specs]
        self.name_to_id = {s.name: s.id for s in self.specs}
        self.atom_vocab = build_atom_vocab(self.specs)
        self.audit()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "NativePredicateBank":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        specs = []
        for row in data.get("predicates", []):
            specs.append(NativePredicateSpec(
                id=int(row["id"]), name=str(row["name"]), entity=str(row["entity"]),
                attribute=str(row["attribute"]), spatial=str(row["spatial"]), polarity=str(row["polarity"]),
                action_scope=str(row["action_scope"]), region=str(row["region"]),
                support_actions=list(row.get("support_actions", [])),
                contra_predicates=list(row.get("contra_predicates", [])),
                mirror_of=row.get("mirror_of"),
            ))
        return cls(specs)

    def audit(self) -> dict[str, Any]:
        ids = [s.id for s in self.specs]
        if len(self.specs) < 40:
            raise ValueError("NTMCal predicate bank must contain at least 40 predicates")
        if ids != list(range(len(ids))):
            raise ValueError("predicate ids must be contiguous from 0")
        if len(set(self.names)) != len(self.names):
            raise ValueError("predicate names must be unique")
        bad = {"predicate_0", "reason_0", "unknown", "placeholder", "tmp", "dummy"} & set(self.names)
        if bad:
            raise ValueError(f"placeholder predicate names forbidden: {sorted(bad)}")
        missing = sorted(REQUIRED_NAMES - set(self.names))
        if missing:
            raise ValueError(f"missing required predicates: {missing}")
        for s in self.specs:
            for c in s.contra_predicates:
                if c not in self.name_to_id:
                    raise ValueError(f"contra predicate {c} missing for {s.name}")
            if s.mirror_of and s.mirror_of not in self.name_to_id:
                raise ValueError(f"mirror predicate {s.mirror_of} missing for {s.name}")
        return {"predicate_count": len(self.specs), "names": self.names, "required_missing": []}
