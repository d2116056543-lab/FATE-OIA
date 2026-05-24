from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def bdd_oia_base_stem(file_name: str) -> str:
    stem = Path(str(file_name).replace("\\", "/")).stem
    return re.sub(r"_[0-9]+$", "", stem)


def load_bdd100k_objects(label_json: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(label_json).read_text(encoding="utf-8", errors="ignore"))
    objects: list[dict[str, Any]] = []
    for frame in data.get("frames", []) or []:
        objects.extend(frame.get("objects", []) or [])
        objects.extend(frame.get("labels", []) or [])
    return objects


@dataclass(frozen=True)
class GroundingPaths:
    label_json: str | None
    drivable_map: str | None
    semantic_seg: str | None


class BDD100KGroundingIndex:
    def __init__(self, bdd100k_root: str | Path = "raw_data/BDD100K") -> None:
        self.root = Path(bdd100k_root)
        self.label_map = {p.stem: p for p in (self.root / "bdd100k_labels").rglob("*.json")} if (self.root / "bdd100k_labels").exists() else {}
        self.drivable_map = {}
        for p in (self.root / "bdd100k_drivable_maps").rglob("*.png") if (self.root / "bdd100k_drivable_maps").exists() else []:
            stem = p.stem
            for suffix in ["_drivable_color", "_drivable_id", "_color", "_id"]:
                if stem.endswith(suffix):
                    stem = stem[: -len(suffix)]
                    break
            self.drivable_map.setdefault(stem, p)
        self.seg_map = {}
        for p in (self.root / "bdd100k_seg").rglob("*.png") if (self.root / "bdd100k_seg").exists() else []:
            stem = p.stem
            for suffix in ["_train_color", "_train_id", "_val_color", "_val_id", "_color", "_id"]:
                if stem.endswith(suffix):
                    stem = stem[: -len(suffix)]
                    break
            self.seg_map.setdefault(stem, p)

    def lookup(self, bdd_oia_file_name: str) -> GroundingPaths:
        base = bdd_oia_base_stem(bdd_oia_file_name)
        return GroundingPaths(
            label_json=str(self.label_map[base]) if base in self.label_map else None,
            drivable_map=str(self.drivable_map[base]) if base in self.drivable_map else None,
            semantic_seg=str(self.seg_map[base]) if base in self.seg_map else None,
        )

    def audit_file_names(self, file_names: list[str]) -> dict[str, float | int]:
        n = len(file_names)
        label = sum(1 for x in file_names if self.lookup(x).label_json)
        drive = sum(1 for x in file_names if self.lookup(x).drivable_map)
        seg = sum(1 for x in file_names if self.lookup(x).semantic_seg)
        return {
            "total": n,
            "label_json": label,
            "label_json_rate": label / n if n else 0.0,
            "drivable_map": drive,
            "drivable_map_rate": drive / n if n else 0.0,
            "semantic_seg": seg,
            "semantic_seg_rate": seg / n if n else 0.0,
        }
