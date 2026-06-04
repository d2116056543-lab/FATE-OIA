from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch


SCENE_STATE_NAMES = [
    "traffic_control_present",
    "front_vehicle_count",
    "front_obstacle_count",
    "vulnerable_front_count",
    "left_object_count",
    "right_object_count",
    "lane_left_proxy",
    "lane_right_proxy",
    "lane_center_proxy",
    "direct_drivable_proxy",
    "global_context_present",
]


def match_bdd_oia_filename_to_bdd100k_base_stem(file_name: str) -> str:
    stem = Path(str(file_name).replace("\\", "/")).stem
    return re.sub(r"_[0-9]+$", "", stem)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_bdd100k_scene_state_index(root: str | Path, split: str | None = None) -> dict[str, dict[str, Any]]:
    return {"_root": str(root), "_split": split or "any", "_lazy": True}


def _labels_from_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    for frame in record.get("frames", []) or []:
        labels.extend(frame.get("labels", []) or [])
        labels.extend(frame.get("objects", []) or [])
    labels.extend(record.get("labels", []) or [])
    return labels


def _box_center(label: dict[str, Any]) -> tuple[float, float] | None:
    box = label.get("box2d") or {}
    keys = ("x1", "y1", "x2", "y2")
    if not all(k in box for k in keys):
        return None
    return (float(box["x1"]) + float(box["x2"])) / 2.0, (float(box["y1"]) + float(box["y2"])) / 2.0


def _poly_mean_x(label: dict[str, Any]) -> float | None:
    polys = label.get("poly2d") or []
    xs: list[float] = []
    for poly in polys:
        vertices = poly.get("vertices") if isinstance(poly, dict) else None
        if vertices is None and isinstance(poly, list):
            vertices = poly
        for point in vertices or []:
            if isinstance(point, dict) and "x" in point:
                xs.append(float(point["x"]))
            elif isinstance(point, (list, tuple)) and len(point) >= 2:
                xs.append(float(point[0]))
    return sum(xs) / len(xs) if xs else None


def scene_state_from_bdd100k_record(record: dict[str, Any] | None) -> dict[str, Any]:
    target = [0.0 for _ in SCENE_STATE_NAMES]
    mask = [0.0 for _ in SCENE_STATE_NAMES]
    counts = {
        "traffic_control_count": 0,
        "front_vehicle_count": 0,
        "front_obstacle_count": 0,
        "vulnerable_front_count": 0,
        "left_object_count": 0,
        "right_object_count": 0,
        "lane_left_count": 0,
        "lane_right_count": 0,
        "lane_center_count": 0,
        "lane_poly_count": 0,
        "drivable_count": 0,
        "object_count": 0,
    }
    proxy_quality = {
        "object_geometry": "box2d_center",
        "lane_geometry": "poly2d_mean_x",
        "direct_drivable_proxy": "unavailable",
    }
    if not record:
        return {"scene_state_target": target, "scene_state_mask": mask, "counts": counts, "proxy_quality": proxy_quality, "missing": True}
    width = float(record.get("width") or record.get("image_width") or 1280)
    height = float(record.get("height") or record.get("image_height") or 720)
    labels = _labels_from_record(record)
    counts["object_count"] = len(labels)
    for label in labels:
        cat = str(label.get("category", "")).lower()
        center = _box_center(label)
        if cat in {"traffic light", "traffic sign", "stop sign"}:
            counts["traffic_control_count"] += 1
        if center is not None:
            cx, cy = center
            if cx < 0.45 * width:
                counts["left_object_count"] += 1
            if cx > 0.55 * width:
                counts["right_object_count"] += 1
            is_front = 0.30 * width <= cx <= 0.70 * width and cy >= 0.35 * height
        else:
            is_front = False
        if cat in {"car", "truck", "bus", "motor", "vehicle", "trailer"} and is_front:
            counts["front_vehicle_count"] += 1
        if cat in {"person", "pedestrian", "rider", "bike", "bicycle", "motorcycle"} and is_front:
            counts["vulnerable_front_count"] += 1
        if cat not in {"traffic light", "traffic sign", "stop sign"} and is_front:
            counts["front_obstacle_count"] += 1
        if "lane" in cat or "crosswalk" in cat or "road curb" in cat:
            mx = _poly_mean_x(label)
            if mx is not None:
                if mx < 0.45 * width:
                    counts["lane_left_count"] += 1
                    counts["lane_poly_count"] += 1
                elif mx > 0.55 * width:
                    counts["lane_right_count"] += 1
                    counts["lane_poly_count"] += 1
                else:
                    counts["lane_center_count"] += 1
                    counts["lane_poly_count"] += 1
    if record.get("_drivable_map") or record.get("drivable_map"):
        counts["drivable_count"] += 1
        proxy_quality["direct_drivable_proxy"] = "weak_drivable_map_presence"
    target[0] = 1.0 if counts["traffic_control_count"] > 0 else 0.0
    target[1] = min(1.0, counts["front_vehicle_count"] / 3.0)
    target[2] = min(1.0, counts["front_obstacle_count"] / 3.0)
    target[3] = min(1.0, counts["vulnerable_front_count"] / 2.0)
    target[4] = min(1.0, counts["left_object_count"] / 5.0)
    target[5] = min(1.0, counts["right_object_count"] / 5.0)
    target[6] = 1.0 if counts["lane_left_count"] > 0 else 0.0
    target[7] = 1.0 if counts["lane_right_count"] > 0 else 0.0
    target[8] = 1.0 if counts["lane_center_count"] > 0 else 0.0
    target[9] = 1.0 if counts["drivable_count"] > 0 else 0.0
    target[10] = 1.0 if labels or record.get("attributes") else 0.0
    for i in range(len(mask)):
        mask[i] = 1.0
    return {"scene_state_target": target, "scene_state_mask": mask, "counts": counts, "proxy_quality": proxy_quality, "missing": False}


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
        names = [f"{base}_drivable_color.png", f"{base}_drivable_id.png", f"{base}_color.png", f"{base}_id.png"]
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
