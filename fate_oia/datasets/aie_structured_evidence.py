from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch import Tensor
from PIL import Image

from .bdd100k_grounding import BDD100KGroundingIndex


def _attributes(item: dict[str, Any]) -> dict[str, Any]:
    values = item.get("attributes") or item.get("attribute") or {}
    return {str(k).lower(): str(v).lower() for k, v in values.items()} if isinstance(values, dict) else {}


def _items(record: dict[str, Any]) -> list[dict[str, Any]]:
    result = [x for x in record.get("labels", []) if isinstance(x, dict)]
    for frame in record.get("frames", []) or []:
        if isinstance(frame, dict):
            result.extend(x for x in frame.get("labels", []) if isinstance(x, dict))
            result.extend(x for x in frame.get("objects", []) if isinstance(x, dict))
    return result


def _box(item: dict[str, Any]) -> tuple[float, float, float, float] | None:
    value = item.get("box2d") or item.get("box")
    if not isinstance(value, dict):
        return None
    x1 = float(value.get("x1", value.get("left", 0))) / 1280.0
    y1 = float(value.get("y1", value.get("top", 0))) / 720.0
    x2 = float(value.get("x2", value.get("right", x1))) / 1280.0
    y2 = float(value.get("y2", value.get("bottom", y1))) / 720.0
    return tuple(max(0.0, min(1.0, x)) for x in (x1, y1, x2, y2))


def _box_map(box: tuple[float, float, float, float], grid_hw: tuple[int, int]) -> Tensor:
    h, w = grid_hw
    x1, y1, x2, y2 = box
    ix1, ix2 = max(0, int(x1 * w)), min(w, max(int(x2 * w) + 1, int(x1 * w) + 1))
    iy1, iy2 = max(0, int(y1 * h)), min(h, max(int(y2 * h) + 1, int(y1 * h) + 1))
    mask = torch.zeros(1, 1, h, w)
    mask[:, :, iy1:iy2, ix1:ix2] = 1
    kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3) / 16
    return F.conv2d(mask, kernel, padding=1).flatten()


def _polyline_map(item: dict[str, Any], grid_hw: tuple[int, int]) -> Tensor | None:
    raw = item.get("poly2d") or item.get("polyline")
    if not raw:
        return None
    if isinstance(raw, list) and raw and isinstance(raw[0], dict) and "vertices" in raw[0]:
        raw = raw[0]["vertices"]
    points = []
    for point in raw if isinstance(raw, list) else []:
        if isinstance(point, dict): points.append((float(point.get("x", 0)), float(point.get("y", 0))))
        elif isinstance(point, (list, tuple)) and len(point) >= 2: points.append((float(point[0]), float(point[1])))
    if len(points) < 2:
        return None
    h, w = grid_hw; mask = torch.zeros(h, w)
    for (x1, y1), (x2, y2) in zip(points[:-1], points[1:]):
        x1, x2 = x1 / 1280.0 if x1 > 1.5 else x1, x2 / 1280.0 if x2 > 1.5 else x2
        y1, y2 = y1 / 720.0 if y1 > 1.5 else y1, y2 / 720.0 if y2 > 1.5 else y2
        steps = max(2, int(max(abs(x2-x1)*w, abs(y2-y1)*h)) + 1)
        for t in torch.linspace(0, 1, steps):
            x = int(round(float((x1 + t * (x2-x1)) * (w-1)))); y = int(round(float((y1 + t * (y2-y1)) * (h-1))))
            mask[max(0,y-1):min(h,y+2), max(0,x-1):min(w,x+2)] = 1
    return mask.flatten()


def _drivable_map(path: str | Path, grid_hw: tuple[int, int]) -> Tensor | None:
    try:
        image = Image.open(path).convert("L").resize((grid_hw[1], grid_hw[0]), Image.Resampling.NEAREST)
        values = torch.tensor(list(image.getdata()), dtype=torch.float32).reshape(grid_hw)
        return (values > 0).float().flatten()
    except OSError:
        return None


class AIEStructuredEvidenceBuilder:
    """Conservative train-only BDD100K observations; unknowns remain masked."""

    def __init__(self, scene_config: str | Path, bdd100k_root: str | Path | None, grid_hw: tuple[int, int] = (45, 80)) -> None:
        config = yaml.safe_load(Path(scene_config).read_text(encoding="utf-8")) or {}
        self.predicates = list(config.get("predicates", []))
        if len(self.predicates) != 32:
            raise ValueError("AIE requires exactly 32 conservative predicates")
        self.names = [str(row["name"]) for row in self.predicates]
        self.name_to_id = {name: index for index, name in enumerate(self.names)}
        self.grid_hw = grid_hw
        self.index = BDD100KGroundingIndex(bdd100k_root) if bdd100k_root else None

    def _set_positive(self, state: dict[str, Tensor], row: int, name: str, evidence_map: Tensor | None, reliability: float) -> None:
        if name not in self.name_to_id:
            return
        col = self.name_to_id[name]
        state["predicate_target"][row, col] = 1
        state["predicate_target_mask"][row, col] = 1
        state["predicate_reliability"][row, col] = float(reliability)
        if evidence_map is not None:
            state["predicate_map_target"][row, col] = torch.maximum(state["predicate_map_target"][row, col], evidence_map)
            state["predicate_map_mask"][row, col] = 1

    def _set_counter(self, state: dict[str, Tensor], row: int, name: str) -> None:
        if name in self.name_to_id:
            col = self.name_to_id[name]
            state["predicate_counter_target"][row, col] = 1
            state["predicate_counter_mask"][row, col] = 1

    def build_from_records(self, records: list[dict[str, Any]], device: torch.device | None = None) -> dict[str, Any]:
        batch, predicates = len(records), len(self.names)
        patches = self.grid_hw[0] * self.grid_hw[1]
        state = {
            "predicate_target": torch.zeros(batch, predicates),
            "predicate_target_mask": torch.zeros(batch, predicates),
            "predicate_counter_target": torch.zeros(batch, predicates),
            "predicate_counter_mask": torch.zeros(batch, predicates),
            "predicate_map_target": torch.zeros(batch, predicates, patches),
            "predicate_map_mask": torch.zeros(batch, predicates),
            "predicate_reliability": torch.zeros(batch, predicates),
            "predicate_source_complete": torch.zeros(batch, predicates),
        }
        source_counts = {"object_json": 0, "drivable_map": 0, "missing": 0}
        coverage = {"positive": 0, "counter": 0, "mapped": 0, "unknown": 0}
        vehicle_categories = {"car", "truck", "bus"}
        for row, record in enumerate(records):
            items = _items(record)
            object_complete = bool(record.get("object_source_complete", bool(record.get("frames") or "labels" in record)))
            drive_map = _drivable_map(record["drivable_map_path"], self.grid_hw) if record.get("drivable_map_path") else None
            source_counts["object_json" if object_complete else "missing"] += 1
            source_counts["drivable_map"] += int(drive_map is not None)
            front_hazard = False
            for item in items:
                category = str(item.get("category", "")).lower()
                attrs = _attributes(item)
                box = _box(item)
                map_value = _box_map(box, self.grid_hw) if box is not None else _polyline_map(item, self.grid_hw)
                if box is not None:
                    x1, y1, x2, y2 = box
                    cx, bottom, area = (x1 + x2) / 2, y2, max(0, x2 - x1) * max(0, y2 - y1)
                else:
                    cx, bottom, area = 0.5, 0.0, 0.0
                if category == "traffic light":
                    self._set_positive(state, row, "traffic_light_visible", map_value, 0.9)
                    color = attrs.get("trafficlightcolor", attrs.get("color", ""))
                    if color in {"red", "green"}:
                        self._set_positive(state, row, f"traffic_light_{color}", map_value, 1.0)
                        self._set_counter(state, row, "traffic_light_green" if color == "red" else "traffic_light_red")
                if category == "traffic sign":
                    self._set_positive(state, row, "traffic_sign_visible", map_value, 0.9)
                    sign_type = attrs.get("type", attrs.get("sign", ""))
                    if "stop" in sign_type:
                        self._set_positive(state, row, "stop_sign_present", map_value, 1.0)
                if category in vehicle_categories and box is not None:
                    if cx < 0.42:
                        self._set_positive(state, row, "vehicle_left", map_value, 0.85)
                        if attrs.get("parked") in {"true", "yes", "1"}:
                            self._set_positive(state, row, "parked_vehicle_left", map_value, 1.0)
                    elif cx > 0.58:
                        self._set_positive(state, row, "vehicle_right", map_value, 0.85)
                        if attrs.get("parked") in {"true", "yes", "1"}:
                            self._set_positive(state, row, "parked_vehicle_right", map_value, 1.0)
                    else:
                        front_hazard = True
                        if bottom >= 0.68 and area >= 0.018:
                            self._set_positive(state, row, "front_vehicle_close", map_value, 0.9)
                            self._set_counter(state, row, "front_vehicle_far")
                        elif bottom <= 0.58 and area <= 0.012:
                            self._set_positive(state, row, "front_vehicle_far", map_value, 0.8)
                            self._set_counter(state, row, "front_vehicle_close")
                if category in {"pedestrian", "person"} and 0.35 <= cx <= 0.65:
                    front_hazard = True; self._set_positive(state, row, "pedestrian_front", map_value, 0.9)
                if category in {"rider", "bike", "bicycle", "motorcycle"} and 0.30 <= cx <= 0.70:
                    front_hazard = True; self._set_positive(state, row, "cyclist_front", map_value, 0.85)
                if category in {"obstacle", "train"}:
                    front_hazard = True; self._set_positive(state, row, "obstacle_front", map_value, 0.85)
                if category == "crosswalk": self._set_positive(state, row, "crosswalk_region", map_value, 0.9)
                if category == "intersection": self._set_positive(state, row, "intersection_region", map_value, 0.9)
                if "lane" in category:
                    direction = attrs.get("direction", attrs.get("side", ""))
                    lane_type = attrs.get("type", attrs.get("lane_type", ""))
                    available = attrs.get("available", attrs.get("drivable", "")) in {"true", "yes", "1"}
                    if direction in {"left", "right"} and available:
                        self._set_positive(state, row, f"lane_{direction}_available", map_value, 0.75)
                    if "turn" in lane_type and direction in {"left", "right"}:
                        self._set_positive(state, row, f"{direction}_turn_region", map_value, 0.9)
                    if "merge" in lane_type and direction in {"left", "right"}:
                        self._set_positive(state, row, f"merging_{direction}_context", map_value, 0.9)
            if object_complete and not front_hazard:
                self._set_positive(state, row, "road_clear", None, 0.7)
            if drive_map is not None:
                h, w = self.grid_hw; grid = drive_map.reshape(h, w); x = torch.linspace(0, 1, w)[None]
                regions = {
                    "drivable_center": ((x >= 0.35) & (x <= 0.65)).float().expand(h, -1),
                    "drivable_left": (x < 0.45).float().expand(h, -1),
                    "drivable_right": (x > 0.55).float().expand(h, -1),
                    "open_left_gap": (x < 0.45).float().expand(h, -1),
                    "open_right_gap": (x > 0.55).float().expand(h, -1),
                }
                for name, region in regions.items():
                    localized = (grid * region).flatten()
                    if float(localized.mean()) > 0.05:
                        self._set_positive(state, row, name, localized, 0.85)
            state["predicate_source_complete"][row] = float(object_complete)
            for name in ("drivable_center", "drivable_left", "drivable_right", "open_left_gap", "open_right_gap"):
                state["predicate_source_complete"][row, self.name_to_id[name]] = float(drive_map is not None)
        for key in list(state):
            state[key] = state[key].to(device=device)
        coverage["positive"] = int(state["predicate_target"].sum().item())
        coverage["counter"] = int(state["predicate_counter_mask"].sum().item())
        coverage["mapped"] = int(state["predicate_map_mask"].sum().item())
        coverage["unknown"] = int((state["predicate_target_mask"] == 0).sum().item())
        return {**state, "source_counts": source_counts, "coverage": coverage}

    def _record_for_file(self, file_name: str) -> dict[str, Any]:
        if self.index is None:
            return {}
        paths = self.index.lookup(file_name)
        record: dict[str, Any] = {"object_source_complete": bool(paths.label_json), "drivable_map_path": paths.drivable_map}
        if paths.label_json:
            try:
                record.update(json.loads(Path(paths.label_json).read_text(encoding="utf-8", errors="ignore")))
            except (OSError, json.JSONDecodeError):
                record["object_source_complete"] = False
        return record

    def build(self, file_names: list[str], device: torch.device | None = None) -> dict[str, Any]:
        return self.build_from_records([self._record_for_file(name) for name in file_names], device=device)

