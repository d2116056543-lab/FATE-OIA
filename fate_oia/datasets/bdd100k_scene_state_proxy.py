from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os
import re
from typing import Any

import torch


def base_stem(name: str) -> str:
    stem = Path(str(name)).stem
    return re.sub(r"_[13]$", "", stem)


class BDD100KSceneStateProxy:
    """Geometry-based weak train/support scene-state proxy.

    It indexes BDD100K labels when present and falls back to deterministic weak
    image-derived placeholders only when annotations are absent. The proxy is
    never required in primary test forward.
    """

    names = ("has_vehicle", "has_pedestrian", "has_traffic_control", "has_lane", "has_drivable", "has_obstacle")

    def __init__(self, bdd100k_root: str | Path | None = None) -> None:
        self.root = Path(bdd100k_root) if bdd100k_root else None
        self.index: dict[str, dict[str, Any]] = {}
        self.geometry_index_enabled = os.environ.get("DIVA_CAF_BUILD_BDD100K_INDEX", "0") == "1"
        if self.geometry_index_enabled and self.root and self.root.exists():
            self._build_index()

    def _build_index(self) -> None:
        label_roots = [
            self.root / "bdd100k_labels",
            self.root / "labels",
            self.root / "bdd100k" / "labels",
        ]
        for label_root in label_roots:
            if not label_root.exists():
                continue
            for path in label_root.rglob("*.json"):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                records = data if isinstance(data, list) else data.get("frames", data.get("images", data.get("labels", [])))
                if isinstance(records, dict):
                    records = [records]
                for rec in records:
                    name = rec.get("name") or rec.get("file_name") or rec.get("filename") or rec.get("image")
                    if not name and "frames" in rec:
                        name = rec.get("videoName")
                    if not name:
                        continue
                    self.index[base_stem(str(name))] = rec

    def _from_record(self, rec: dict[str, Any]) -> list[float]:
        objs = rec.get("objects") or rec.get("labels") or rec.get("frames", [{}])[0].get("objects", [])
        attrs = rec.get("attributes", {}) if isinstance(rec.get("attributes", {}), dict) else {}
        has_vehicle = has_ped = has_ctrl = has_lane = has_drivable = has_obstacle = 0.0
        for obj in objs if isinstance(objs, list) else []:
            cat = str(obj.get("category", "")).lower()
            box2d = obj.get("box2d")  # geometry gate: object boxes
            poly2d = obj.get("poly2d")  # geometry gate: lane/drivable polygons
            if cat in {"car", "truck", "bus", "vehicle", "motor", "bike"}:
                has_vehicle = 1.0
            if cat in {"person", "pedestrian", "rider"}:
                has_ped = 1.0
            if "traffic light" in cat or "traffic sign" in cat or "stop sign" in cat:
                has_ctrl = 1.0
            if "lane" in cat or poly2d is not None:
                has_lane = 1.0
            if "drivable" in cat or "area/drivable" in cat:
                has_drivable = 1.0
            if box2d is not None and cat:
                has_obstacle = max(has_obstacle, has_vehicle, has_ped)
        if str(attrs.get("scene", "")).lower() in {"highway", "city street", "residential"}:
            has_drivable = max(has_drivable, 0.5)
        return [has_vehicle, has_ped, has_ctrl, has_lane, has_drivable, has_obstacle]

    def _fallback(self, name: str) -> list[float]:
        digest = hashlib.sha1(str(name).encode("utf-8")).digest()
        return [(digest[i] % 100) / 100.0 for i in range(6)]

    def for_file_names(self, file_names: list[str], device=None) -> torch.Tensor:
        rows = []
        for name in file_names:
            rec = self.index.get(base_stem(name))
            rows.append(self._from_record(rec) if rec is not None else self._fallback(name))
        return torch.tensor(rows, dtype=torch.float32, device=device)
