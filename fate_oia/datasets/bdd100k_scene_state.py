from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch


SCENE_STATE_NAMES = [
    "traffic_control_present",
    "vehicle_present",
    "front_vehicle_or_obstacle_density",
    "vulnerable_user_present",
    "lane_poly_present",
    "lane_left_proxy",
    "lane_right_proxy",
    "drivable_present",
    "drivable_direct_proxy",
    "weather_or_time_attr_present",
    "global_context_present",
]


def match_bdd_oia_filename_to_bdd100k_base_stem(file_name: str) -> str:
    stem = Path(str(file_name).replace("\\", "/")).stem
    return re.sub(r"_[0-9]+$", "", stem)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_bdd100k_scene_state_index(root: str | Path, split: str | None = None) -> dict[str, dict[str, Any]]:
    """Return an intentionally lazy index descriptor.

    Full recursive BDD100K indexing can take many minutes on the remote disk.
    CEAI training only needs the current batch stems, so BDD100KSceneStateIndex
    resolves candidate paths on demand and caches hits.
    """
    return {"_root": str(root), "_split": split or "any"}

def _labels_from_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    for frame in record.get("frames", []) or []:
        labels.extend(frame.get("labels", []) or [])
        labels.extend(frame.get("objects", []) or [])
    return labels


def scene_state_from_bdd100k_record(record: dict[str, Any] | None) -> dict[str, Any]:
    target = [0.0 for _ in SCENE_STATE_NAMES]
    mask = [0.0 for _ in SCENE_STATE_NAMES]
    counts = {
        "traffic_control_count": 0,
        "vehicle_count": 0,
        "vulnerable_count": 0,
        "lane_poly_count": 0,
        "drivable_count": 0,
        "left_region_proxy_count": 0,
        "right_region_proxy_count": 0,
        "object_count": 0,
    }
    if not record:
        return {"scene_state_target": target, "scene_state_mask": mask, "counts": counts, "missing": True}
    labels = _labels_from_record(record)
    attrs = record.get("attributes", {}) or {}
    counts["object_count"] = len(labels)
    for label in labels:
        cat = str(label.get("category", "")).lower()
        if cat in {"traffic light", "traffic sign", "stop sign"}:
            counts["traffic_control_count"] += 1
        if cat in {"car", "truck", "bus", "motor", "vehicle", "trailer"}:
            counts["vehicle_count"] += 1
        if cat in {"person", "pedestrian", "rider", "bike", "bicycle"}:
            counts["vulnerable_count"] += 1
        if "lane" in cat or "crosswalk" in cat or "road curb" in cat:
            if label.get("poly2d"):
                counts["lane_poly_count"] += 1
            if "left" in cat or "yellow" in cat:
                counts["left_region_proxy_count"] += 1
            if "right" in cat or "white" in cat:
                counts["right_region_proxy_count"] += 1
        if "drivable" in cat or "area" in cat:
            counts["drivable_count"] += 1
    if record.get("_drivable_map") or record.get("drivable_map"):
        counts["drivable_count"] += 1
    target[0] = 1.0 if counts["traffic_control_count"] > 0 else 0.0
    target[1] = 1.0 if counts["vehicle_count"] > 0 else 0.0
    target[2] = min(1.0, counts["vehicle_count"] / 3.0)
    target[3] = 1.0 if counts["vulnerable_count"] > 0 else 0.0
    target[4] = 1.0 if counts["lane_poly_count"] > 0 else 0.0
    target[5] = 1.0 if counts["left_region_proxy_count"] > 0 else 0.0
    target[6] = 1.0 if counts["right_region_proxy_count"] > 0 else 0.0
    target[7] = 1.0 if counts["drivable_count"] > 0 else 0.0
    target[8] = 1.0 if counts["drivable_count"] > 0 and counts["vehicle_count"] < 3 else 0.0
    target[9] = 1.0 if bool(attrs) else 0.0
    target[10] = 1.0 if labels or attrs else 0.0
    for i in range(len(mask)):
        mask[i] = 1.0
    return {"scene_state_target": target, "scene_state_mask": mask, "counts": counts, "missing": False}


class BDD100KSceneStateIndex:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.records = build_bdd100k_scene_state_index(root)
        self.cache: dict[str, dict[str, Any]] = {}

    def _candidate_label_paths(self, base: str) -> list[Path]:
        label_root = self.root / "bdd100k_labels" / "bdd100k" / "labels" / "100k"
        return [label_root / split / f"{base}.json" for split in ("train", "val", "test")]

    def _candidate_drivable_paths(self, base: str) -> list[Path]:
        drive_root = self.root / "bdd100k_drivable_maps" / "bdd100k" / "drivable_maps"
        names = [
            f"{base}_drivable_color.png",
            f"{base}_drivable_id.png",
            f"{base}_color.png",
            f"{base}_id.png",
        ]
        candidates: list[Path] = []
        for split in ("train", "val", "test"):
            for sub in ("color_labels", "labels"):
                for name in names:
                    candidates.append(drive_root / sub / split / name)
        return candidates

    def lookup(self, file_name: str) -> dict[str, Any]:
        base = match_bdd_oia_filename_to_bdd100k_base_stem(file_name)
        if base in self.cache:
            return self.cache[base]
        rec: dict[str, Any] = {}
        for path in self._candidate_label_paths(base):
            if path.exists():
                try:
                    rec = _load_json(path)
                    rec["_label_json"] = str(path)
                except Exception:
                    rec = {}
                break
        for path in self._candidate_drivable_paths(base):
            if path.exists():
                rec["_drivable_map"] = str(path)
                break
        self.cache[base] = rec
        return rec

    def batch_targets(self, file_names: list[str], device: torch.device | None = None) -> dict[str, torch.Tensor | list[dict[str, Any]]]:
        rows = [scene_state_from_bdd100k_record(self.lookup(x)) for x in file_names]
        target = torch.tensor([r["scene_state_target"] for r in rows], dtype=torch.float32, device=device)
        mask = torch.tensor([r["scene_state_mask"] for r in rows], dtype=torch.float32, device=device)
        return {"target": target, "mask": mask, "records": rows}
