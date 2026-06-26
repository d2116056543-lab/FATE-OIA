from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import yaml

from fate_oia.datasets.bdd100k_grounding import BDD100KGroundingIndex


def _norm_xy(x: float, y: float, image_size: tuple[int, int] = (1280, 720)) -> tuple[float, float]:
    if x > 1.5 or y > 1.5:
        return x / max(float(image_size[0]), 1.0), y / max(float(image_size[1]), 1.0)
    return x, y


def _region_for_xy(x: float, y: float) -> str:
    x, y = _norm_xy(x, y)
    if y < 0.45:
        return "upper_traffic_region"
    if x < 0.42 and y > 0.35:
        return "left_corridor"
    if x > 0.58 and y > 0.35:
        return "right_corridor"
    if y > 0.60:
        return "bottom_drivable_region"
    return "front_center"


class WeakPredicateTargetBuilder:
    """Build weak scene-predicate targets from BDD100K geometry, not strings only."""

    def __init__(self, scene_config: str | Path, bdd100k_root: str | Path | None = None) -> None:
        data = yaml.safe_load(Path(scene_config).read_text(encoding="utf-8")) or {}
        self.predicates = list(data.get("predicates", []))
        if len(self.predicates) < 32:
            raise ValueError("ACPR requires at least 32 scene predicates")
        self.names = [str(p["name"]) for p in self.predicates]
        self.name_to_id = {n: i for i, n in enumerate(self.names)}
        self.index = BDD100KGroundingIndex(bdd100k_root) if bdd100k_root else None

    def _iter_items(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for frame in data.get("frames", []):
            if isinstance(frame, dict):
                items.extend([x for x in frame.get("labels", []) if isinstance(x, dict)])
        items.extend([x for x in data.get("labels", []) if isinstance(x, dict)])
        return items

    def _record_targets(self, items: list[dict[str, Any]], drivable_available: bool = False) -> tuple[set[str], dict[str, int], dict[str, float]]:
        names: set[str] = set()
        counts = {"object_box": 0, "lane_poly": 0, "drivable": int(drivable_available), "weak_unknown": 0}
        reliability: dict[str, float] = {}
        for item in items:
            cat = str(item.get("category", "")).lower()
            box = item.get("box2d") or item.get("box")
            poly = item.get("poly2d") or item.get("polyline")
            region = "front_center"
            box_bottom_norm = 0.0
            if isinstance(box, dict):
                x1 = float(box.get("x1", box.get("left", 0.0)))
                x2 = float(box.get("x2", box.get("right", x1)))
                y1 = float(box.get("y1", box.get("top", 0.0)))
                y2 = float(box.get("y2", box.get("bottom", y1)))
                _, box_bottom_norm = _norm_xy((x1 + x2) * 0.5, y2)
                region = _region_for_xy((x1 + x2) * 0.5, (y1 + y2) * 0.5)
                counts["object_box"] += 1
            elif poly:
                if isinstance(poly, list) and poly and isinstance(poly[0], dict):
                    pts = poly[0].get("vertices", poly[0].get("points", []))
                elif isinstance(poly, list) and len(poly) == 1 and isinstance(poly[0], list):
                    pts = poly[0]
                else:
                    pts = poly
                xs, ys = [], []
                for p in pts or []:
                    if isinstance(p, dict):
                        xs.append(float(p.get("x", 0.0))); ys.append(float(p.get("y", 0.0)))
                    elif isinstance(p, (list, tuple)) and len(p) >= 2:
                        xs.append(float(p[0])); ys.append(float(p[1]))
                if xs and ys:
                    region = _region_for_xy(sum(xs) / len(xs), sum(ys) / len(ys))
                    counts["lane_poly"] += 1
            if cat in {"traffic light"}:
                names.add("traffic_light_visible")
                if region == "upper_traffic_region":
                    reliability["traffic_light_visible"] = 0.80
            if cat in {"traffic sign"}:
                names.add("traffic_sign_visible")
            if cat in {"car", "truck", "bus"}:
                if region == "left_corridor":
                    names.add("vehicle_left")
                elif region == "right_corridor":
                    names.add("vehicle_right")
                else:
                    names.add("front_vehicle_close" if box_bottom_norm >= 0.65 else "front_vehicle_far")
                    names.add("road_crowded")
            if cat in {"person", "pedestrian"}:
                names.update(["pedestrian_front", "road_crowded"])
            if cat in {"rider", "bike", "bicycle", "motorcycle"}:
                names.add("cyclist_front")
            if "lane" in cat or poly:
                if region == "left_corridor":
                    names.update(["lane_left_available", "left_lane_boundary", "left_solid_boundary"])
                elif region == "right_corridor":
                    names.update(["lane_right_available", "right_lane_boundary", "right_solid_boundary"])
            if cat in {"crosswalk"}:
                names.add("crosswalk_region")
            if cat in {"intersection"}:
                names.add("intersection_region")
            if cat in {"obstacle", "train"}:
                names.add("obstacle_front")
        if drivable_available:
            names.update(["drivable_center", "drivable_left", "drivable_right"])
        names.add("global_scene_context")
        for n in names:
            reliability.setdefault(n, 0.70 if n not in {"road_clear", "global_scene_context"} else 0.45)
        return names, counts, reliability

    def build_from_records(self, records: list[dict[str, Any]], device: torch.device | None = None) -> dict[str, torch.Tensor | dict[str, int]]:
        b = len(records)
        m = len(self.predicates)
        target = torch.zeros(b, m, dtype=torch.float32, device=device)
        mask = torch.zeros(b, m, dtype=torch.float32, device=device)
        rel = torch.zeros(b, m, dtype=torch.float32, device=device)
        source = torch.full((b, m), -1, dtype=torch.long, device=device)
        coverage = {"object_box_count": 0, "lane_poly_count": 0, "drivable_count": 0, "weak_unknown_count": 0}
        for i, rec in enumerate(records):
            items = self._iter_items(rec)
            drivable_available = bool(rec.get("drivable_available", False))
            names, counts, reliability = self._record_targets(items, drivable_available)
            if not items and not drivable_available:
                counts["weak_unknown"] += 1
            for k, v in counts.items():
                if k + "_count" in coverage:
                    coverage[k + "_count"] += int(v)
                elif k in coverage:
                    coverage[k] += int(v)
            for name in names:
                if name in self.name_to_id:
                    j = self.name_to_id[name]
                    target[i, j] = 1.0
                    mask[i, j] = 1.0
                    rel[i, j] = float(reliability.get(name, 0.5))
                    source[i, j] = 0 if name.startswith(("front", "vehicle", "pedestrian", "traffic")) else 1
            if not names:
                coverage["weak_unknown_count"] += 1
        source_counts = {"label_json": b - coverage["weak_unknown_count"], "heuristic": coverage["object_box_count"] + coverage["lane_poly_count"] + coverage["drivable_count"], "missing": coverage["weak_unknown_count"]}
        return {"predicate_targets": target, "predicate_mask": mask, "predicate_source": source, "predicate_reliability": rel, "predicate_coverage": coverage, "source_counts": source_counts}

    def _record_for_file(self, file_name: str) -> dict[str, Any]:
        if self.index is None:
            return {}
        paths = self.index.lookup(file_name)
        rec: dict[str, Any] = {"drivable_available": bool(paths.drivable_map)}
        if paths.label_json:
            try:
                rec.update(json.loads(Path(paths.label_json).read_text(encoding="utf-8", errors="ignore")))
            except Exception:
                pass
        return rec

    def build(self, file_names: list[str], device: torch.device | None = None) -> dict[str, torch.Tensor | dict[str, int]]:
        return self.build_from_records([self._record_for_file(fn) for fn in file_names], device=device)
