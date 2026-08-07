from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor
from PIL import Image

from .bdd100k_grounding import BDD100KGroundingIndex


def _labels(record: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [x for x in record.get("labels", []) if isinstance(x, dict)]
    for frame in record.get("frames", []) or []:
        if isinstance(frame, dict):
            rows.extend(x for x in frame.get("labels", []) if isinstance(x, dict))
            rows.extend(x for x in frame.get("objects", []) if isinstance(x, dict))
    return rows


def _box(item: dict[str, Any]) -> tuple[float, float, float, float] | None:
    value = item.get("box2d") or item.get("box")
    if not isinstance(value, dict):
        return None
    x1, y1 = float(value.get("x1", 0)) / 1280.0, float(value.get("y1", 0)) / 720.0
    x2, y2 = float(value.get("x2", x1)) / 1280.0, float(value.get("y2", y1)) / 720.0
    return tuple(max(0.0, min(1.0, x)) for x in (x1, y1, x2, y2))


def _box_mask(box: tuple[float, float, float, float], grid=(45, 80)) -> Tensor:
    h, w = grid
    x1, y1, x2, y2 = box
    xa, xb = int(x1 * w), max(int(x2 * w) + 1, int(x1 * w) + 1)
    ya, yb = int(y1 * h), max(int(y2 * h) + 1, int(y1 * h) + 1)
    result = torch.zeros(h, w)
    result[max(0, ya):min(h, yb), max(0, xa):min(w, xb)] = 1.0
    return result.flatten()


def _drivable(path: str | None, grid=(45, 80)) -> Tensor | None:
    if not path:
        return None
    try:
        image = Image.open(path).convert("L").resize((grid[1], grid[0]), Image.Resampling.NEAREST)
        return (torch.tensor(list(image.getdata()), dtype=torch.float32).reshape(grid) > 0).float()
    except OSError:
        return None


class AIECertStructuredEvidenceBuilder:
    """External-only observations; absence remains unknown unless its source is complete."""
    OBJECT_PREDICATES = {"traffic_light_red", "traffic_light_green", "stop_sign_present", "front_vehicle_close",
        "front_vehicle_far", "pedestrian_front", "cyclist_front", "obstacle_front", "road_clear", "road_crowded",
        "parked_vehicle_left", "parked_vehicle_right", "vehicle_left", "vehicle_right", "traffic_light_visible",
        "traffic_sign_visible", "crosswalk_region", "intersection_region", "low_front_visibility"}
    LANE_PREDICATES = {"lane_left_available", "lane_right_available", "left_turn_region", "right_turn_region",
        "merging_left_context", "merging_right_context", "ego_lane_centered"}
    DRIVABLE_PREDICATES = {"drivable_center", "drivable_left", "drivable_right", "open_left_gap", "open_right_gap"}

    def __init__(self, scene_config: str | Path, counter_config: str | Path,
                 bdd100k_root: str | Path | None, grid_hw=(45, 80)):
        scene = yaml.safe_load(Path(scene_config).read_text(encoding="utf-8")) or {}
        self.names = [str(row["name"]) for row in scene.get("predicates", [])]
        if len(self.names) != 32:
            raise ValueError("AIE-CERT requires 32 predicates")
        self.name_to_id = {name: i for i, name in enumerate(self.names)}
        counter = yaml.safe_load(Path(counter_config).read_text(encoding="utf-8")) or {}
        self.reason_rules = counter.get("reasons", {})
        self.index = BDD100KGroundingIndex(bdd100k_root) if bdd100k_root else None
        self.grid_hw = grid_hw

    def _set(self, state, row, name, mask=None, reliability=1.0):
        if name not in self.name_to_id:
            return
        col = self.name_to_id[name]
        state["predicate_target"][row, col] = 1.0
        state["predicate_positive_mask"][row, col] = 1.0
        state["predicate_reliability"][row, col] = reliability
        if mask is not None:
            state["predicate_map_target"][row, col] = torch.maximum(state["predicate_map_target"][row, col], mask)
            state["predicate_map_mask"][row, col] = 1.0

    def build_from_records(self, records: list[dict[str, Any]], device=None) -> dict[str, Any]:
        b, p, n = len(records), 32, self.grid_hw[0] * self.grid_hw[1]
        state = {"predicate_target": torch.zeros(b, p), "predicate_positive_mask": torch.zeros(b, p),
            "predicate_counter_mask": torch.zeros(b, p), "predicate_map_target": torch.zeros(b, p, n),
            "predicate_map_mask": torch.zeros(b, p), "predicate_reliability": torch.zeros(b, p),
            "predicate_source_complete": torch.zeros(b, p)}
        source_counts = {"object": 0, "lane": 0, "drivable": 0}
        for row, record in enumerate(records):
            object_complete = bool(record.get("object_source_complete"))
            lane_complete = bool(record.get("lane_source_complete"))
            drivable_complete = bool(record.get("drivable_source_complete"))
            source_counts["object"] += int(object_complete)
            source_counts["lane"] += int(lane_complete)
            source_counts["drivable"] += int(drivable_complete)
            for name in self.OBJECT_PREDICATES:
                if name in self.name_to_id: state["predicate_source_complete"][row, self.name_to_id[name]] = object_complete
            for name in self.LANE_PREDICATES:
                if name in self.name_to_id: state["predicate_source_complete"][row, self.name_to_id[name]] = lane_complete
            for name in self.DRIVABLE_PREDICATES:
                if name in self.name_to_id: state["predicate_source_complete"][row, self.name_to_id[name]] = drivable_complete
            front_hazard, vehicle_count = False, 0
            for item in _labels(record):
                category = str(item.get("category", "")).lower()
                attrs = {str(k).lower(): str(v).lower() for k, v in (item.get("attributes") or {}).items()}
                box = _box(item)
                mask = _box_mask(box, self.grid_hw) if box else None
                cx = (box[0] + box[2]) / 2 if box else 0.5
                bottom = box[3] if box else 0.0
                area = (box[2] - box[0]) * (box[3] - box[1]) if box else 0.0
                if category == "traffic light":
                    self._set(state, row, "traffic_light_visible", mask, 0.9)
                    color = attrs.get("trafficlightcolor", attrs.get("color", ""))
                    if color in {"red", "green"}: self._set(state, row, f"traffic_light_{color}", mask, 1.0)
                elif category == "traffic sign":
                    self._set(state, row, "traffic_sign_visible", mask, 0.9)
                    if "stop" in attrs.get("type", ""): self._set(state, row, "stop_sign_present", mask, 1.0)
                elif category in {"car", "truck", "bus"} and box:
                    vehicle_count += 1
                    if cx < 0.42: self._set(state, row, "vehicle_left", mask, 0.85)
                    elif cx > 0.58: self._set(state, row, "vehicle_right", mask, 0.85)
                    else:
                        front_hazard = True
                        if bottom >= 0.68 and area >= 0.018: self._set(state, row, "front_vehicle_close", mask, 0.9)
                        elif bottom <= 0.58 and area <= 0.012: self._set(state, row, "front_vehicle_far", mask, 0.8)
                elif category in {"pedestrian", "person"} and 0.35 <= cx <= 0.65:
                    front_hazard = True; self._set(state, row, "pedestrian_front", mask, 0.9)
                elif category in {"rider", "bike", "bicycle", "motorcycle"} and 0.30 <= cx <= 0.70:
                    front_hazard = True; self._set(state, row, "cyclist_front", mask, 0.85)
                elif category == "crosswalk": self._set(state, row, "crosswalk_region", mask, 0.9)
                elif category == "intersection": self._set(state, row, "intersection_region", mask, 0.9)
                elif "lane" in category:
                    side = attrs.get("direction", attrs.get("side", ""))
                    lane_type = attrs.get("type", attrs.get("lane_type", ""))
                    available = attrs.get("available", attrs.get("drivable", "")) in {"true", "yes", "1"}
                    if side in {"left", "right"} and available:
                        self._set(state, row, f"lane_{side}_available", mask, 0.75)
                    if side in {"left", "right"} and "turn" in lane_type:
                        self._set(state, row, f"{side}_turn_region", mask, 0.9)
                    if side in {"left", "right"} and "merge" in lane_type:
                        self._set(state, row, f"merging_{side}_context", mask, 0.9)
            if object_complete and not front_hazard: self._set(state, row, "road_clear", None, 0.7)
            if object_complete and vehicle_count >= 6: self._set(state, row, "road_crowded", None, 0.75)
            drive = _drivable(record.get("drivable_map_path"), self.grid_hw)
            if drive is not None:
                h, w = self.grid_hw; x = torch.linspace(0, 1, w)[None].expand(h, -1)
                for name, region in {"drivable_center": (x >= .35) & (x <= .65), "drivable_left": x < .45,
                                     "drivable_right": x > .55, "open_left_gap": x < .45,
                                     "open_right_gap": x > .55}.items():
                    localized = drive * region
                    if float(localized.mean()) > .05: self._set(state, row, name, localized.flatten(), .85)
        state["predicate_counter_mask"] = state["predicate_source_complete"] * (1.0 - state["predicate_target"])
        reason_positive = torch.zeros(b, 21)
        reason_counter = torch.zeros(b, 21)
        reason_reliability = torch.zeros(b, 21)
        reason_observable = torch.zeros(b, 21)
        for reason in range(21):
            rule = self.reason_rules.get(reason, self.reason_rules.get(str(reason), {}))
            pos_ids = [self.name_to_id[x] for x in rule.get("positive_predicates", []) if x in self.name_to_id]
            neg_ids = [self.name_to_id[x] for x in rule.get("contradictory_predicates", []) if x in self.name_to_id]
            if pos_ids: reason_positive[:, reason] = state["predicate_target"][:, pos_ids].amax(-1)
            if neg_ids:
                observed = state["predicate_target"][:, neg_ids] * state["predicate_source_complete"][:, neg_ids]
                reason_counter[:, reason] = observed.amax(-1)
                reason_reliability[:, reason] = (observed * state["predicate_reliability"][:, neg_ids]).amax(-1)
                reason_observable[:, reason] = state["predicate_source_complete"][:, neg_ids].amax(-1)
        tensors = {**state, "reason_positive_support": reason_positive, "reason_verified_counter": reason_counter,
                   "reason_counter_reliability": reason_reliability, "reason_observable_mask": reason_observable}
        tensors = {k: v.to(device) for k, v in tensors.items()}
        coverage = {"predicate_positive": int(state["predicate_target"].sum()),
                    "predicate_counter": int(state["predicate_counter_mask"].sum()),
                    "reason_counter": int(reason_counter.sum())}
        return {**tensors, "source_counts": source_counts, "coverage": coverage,
                "per_predicate_coverage": state["predicate_positive_mask"].mean(0).tolist(),
                "per_reason_counter_coverage": reason_counter.mean(0).tolist()}

    def build(self, file_names: list[str], device=None) -> dict[str, Any]:
        records = []
        for name in file_names:
            record: dict[str, Any] = {}
            if self.index is not None:
                paths = self.index.lookup(name)
                record.update(object_source_complete=bool(paths.label_json), lane_source_complete=bool(paths.label_json),
                              drivable_source_complete=bool(paths.drivable_map), drivable_map_path=paths.drivable_map)
                for source in (paths.label_json,):
                    if source:
                        try:
                            data = json.loads(Path(source).read_text(encoding="utf-8", errors="ignore"))
                            if isinstance(data, dict):
                                record.setdefault("frames", []).extend(data.get("frames", []))
                                record.setdefault("labels", []).extend(data.get("labels", []))
                        except (OSError, json.JSONDecodeError):
                            pass
            records.append(record)
        return self.build_from_records(records, device=device)
