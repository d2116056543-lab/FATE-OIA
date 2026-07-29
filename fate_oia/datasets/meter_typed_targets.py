from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image, ImageDraw
from torch import Tensor

from fate_oia.grounding.mask_builder import drivable_map_to_mask


def _box(obj: dict[str, Any]) -> tuple[float, float, float, float] | None:
    value = obj.get("box2d") or obj.get("box")
    if not isinstance(value, dict):
        return None
    keys = ("x1", "y1", "x2", "y2")
    if not all(key in value for key in keys):
        return None
    return tuple(float(value[key]) for key in keys)


def _category(obj: dict[str, Any]) -> str:
    return str(obj.get("category", obj.get("name", ""))).lower()


def _attribute(obj: dict[str, Any], *names: str) -> str:
    attributes = obj.get("attributes", {})
    if not isinstance(attributes, dict):
        attributes = {}
    for name in names:
        value = attributes.get(name, obj.get(name))
        if value not in (None, "", "unknown", "none"):
            return str(value).lower()
    return ""


def _box_mask(
    boxes: list[tuple[float, float, float, float]],
    *,
    grid_hw: tuple[int, int],
    image_size: tuple[int, int],
) -> Tensor:
    height, width = grid_hw
    image_width, image_height = image_size
    mask = torch.zeros(height, width, dtype=torch.float32)
    for x1, y1, x2, y2 in boxes:
        gx1 = max(0, min(width - 1, int(x1 / max(image_width, 1) * width)))
        gx2 = max(gx1 + 1, min(width, int((x2 + 1) / max(image_width, 1) * width)))
        gy1 = max(0, min(height - 1, int(y1 / max(image_height, 1) * height)))
        gy2 = max(gy1 + 1, min(height, int((y2 + 1) / max(image_height, 1) * height)))
        mask[gy1:gy2, gx1:gx2] = 1.0
    return mask


def _points(value: Any) -> list[tuple[float, float]]:
    if isinstance(value, dict):
        value = value.get("vertices") or value.get("points") or value.get("verts")
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list):
        return []
    points: list[tuple[float, float]] = []
    for item in value:
        if isinstance(item, dict) and "x" in item and "y" in item:
            points.append((float(item["x"]), float(item["y"])))
        elif isinstance(item, dict):
            points.extend(_points(item.get("vertices") or item.get("points") or item.get("verts")))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            points.append((float(item[0]), float(item[1])))
    return points


def _polyline_mask(
    polylines: list[Any],
    *,
    image_size: tuple[int, int],
    grid_hw: tuple[int, int],
) -> Tensor:
    image_width, image_height = image_size
    out_height, out_width = grid_hw
    image = Image.new("L", (out_width, out_height), 0)
    draw = ImageDraw.Draw(image)
    for raw in polylines:
        if isinstance(raw, dict):
            raw = raw.get("poly2d") or raw.get("polyline") or raw.get("vertices") or raw.get("points")
        points = _points(raw)
        if len(points) < 2:
            continue
        scaled = [
            (
                max(0, min(out_width - 1, int(round(x / max(image_width, 1) * (out_width - 1))))),
                max(0, min(out_height - 1, int(round(y / max(image_height, 1) * (out_height - 1))))),
            )
            for x, y in points
        ]
        draw.line(scaled, fill=1, width=max(1, round(max(out_height, out_width) / 160)))
    return torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).clone().view(out_height, out_width).float()


def _normalise_mask(mask: Tensor) -> Tensor:
    total = mask.sum()
    return mask / total.clamp_min(1.0) if bool(total > 0) else mask


class METERTypedTargetBuilder:
    """Conservative typed weak targets; unknown never becomes a negative."""

    def __init__(
        self,
        schema_path: str | Path,
        *,
        grid_hw: tuple[int, int] = (45, 80),
        image_size: tuple[int, int] = (1280, 720),
    ) -> None:
        raw = yaml.safe_load(Path(schema_path).read_text(encoding="utf-8")) or {}
        self.factors = list(raw["factors"])
        if [int(row["id"]) for row in self.factors] != list(range(21)):
            raise ValueError("TESA schema must contain ordered factor IDs 0..20")
        self.grid_hw = tuple(int(value) for value in grid_hw)
        self.image_size = tuple(int(value) for value in image_size)

    def _geometry_masks(self, record: dict[str, Any]) -> dict[str, Tensor]:
        image_size_raw = record.get("image_size", self.image_size)
        image_size = (int(image_size_raw[0]), int(image_size_raw[1]))
        object_polylines = [
            obj
            for obj in record.get("objects", [])
            if isinstance(obj, dict)
            and obj.get("poly2d")
            and any(
                name in _category(obj)
                for name in ("lane", "road marking", "drivable")
            )
        ]
        polylines = (
            list(record.get("lanes", []))
            + list(record.get("polylines", []))
            + object_polylines
        )
        lane = _polyline_mask(polylines, image_size=image_size, grid_hw=self.grid_hw)
        drivable = torch.zeros(self.grid_hw, dtype=torch.float32)
        if record.get("drivable_map_path"):
            try:
                drivable = drivable_map_to_mask(str(record["drivable_map_path"]), self.grid_hw)
            except (OSError, ValueError):
                drivable = torch.zeros(self.grid_hw, dtype=torch.float32)
        columns = torch.arange(self.grid_hw[1]).view(1, -1).expand(*self.grid_hw)
        rows = torch.arange(self.grid_hw[0]).view(-1, 1).expand(*self.grid_hw)
        left = columns < self.grid_hw[1] / 2.0
        lower = rows >= self.grid_hw[0] * 0.35
        center = (columns >= self.grid_hw[1] * 0.25) & (
            columns < self.grid_hw[1] * 0.70
        )
        return {
            "lane_left": lane * left,
            "lane_right": lane * (~left),
            "drivable": drivable,
            "drivable_center": drivable * center * lower,
            "drivable_left": drivable * left * lower,
            "drivable_right": drivable * (~left) * lower,
        }

    def build(self, record: dict[str, Any]) -> dict[str, Tensor]:
        factor_count = len(self.factors)
        anchor = torch.zeros(factor_count, *self.grid_hw)
        anchor_valid = torch.zeros(factor_count, dtype=torch.bool)
        state_target = torch.full((factor_count,), -1, dtype=torch.long)
        state_valid = torch.zeros(factor_count, dtype=torch.bool)
        present_valid = torch.zeros(factor_count, dtype=torch.bool)
        absent_valid = torch.zeros(factor_count, dtype=torch.bool)
        source_complete = torch.zeros(factor_count, dtype=torch.bool)
        observability = torch.zeros(factor_count)
        observability_valid = torch.zeros(factor_count, dtype=torch.bool)
        source_weight = torch.zeros(factor_count)

        objects = list(record.get("objects", []))
        detection_complete = bool(record.get("source_complete", False))
        image_size_raw = record.get("image_size", self.image_size)
        image_size = (int(image_size_raw[0]), int(image_size_raw[1]))
        groups: dict[str, list[tuple[float, float, float, float]]] = {}
        for obj in objects:
            box = _box(obj)
            if box is not None:
                groups.setdefault(_category(obj), []).append(box)

        category_rules = {
            0: ("traffic light",), 3: ("traffic light",), 4: ("traffic sign",),
            5: ("car", "bus", "truck", "vehicle"),
            6: ("person", "pedestrian"), 7: ("rider", "cyclist", "bike", "motor"),
            8: ("obstacle", "other"),
        }
        for factor_id, categories in category_rules.items():
            boxes = [box for category, values in groups.items() if any(name in category for name in categories) for box in values]
            mask = _box_mask(boxes, grid_hw=self.grid_hw, image_size=image_size)
            if bool(mask.any()):
                anchor[factor_id] = mask
                anchor_valid[factor_id] = True
                observability[factor_id] = 1.0
                observability_valid[factor_id] = True
                source_weight[factor_id] = 1.0
                if factor_id != 0:
                    state_target[factor_id] = 0
                    state_valid[factor_id] = True
                    present_valid[factor_id] = True
            elif detection_complete and factor_id != 0:
                state_target[factor_id] = 1
                state_valid[factor_id] = True
                absent_valid[factor_id] = True
                observability[factor_id] = 1.0
                observability_valid[factor_id] = True
                source_weight[factor_id] = 0.8
            source_complete[factor_id] = detection_complete

        lights = [obj for obj in objects if "traffic light" in _category(obj)]
        colors = {_attribute(obj, "trafficLightColor", "traffic_light_color", "color") for obj in lights}
        colors.discard("")
        if lights:
            if "green" in colors:
                state_target[0], state_valid[0] = 0, True
            elif colors & {"red", "yellow", "redyellow", "red_yellow"}:
                state_target[0], state_valid[0] = 1, True
            if state_valid[0]:
                present_valid[0] = True
            source_weight[0] = 1.0 if state_valid[0] else 0.5
            source_complete[0] = state_valid[0]

        vehicle_boxes = [box for category, values in groups.items() if any(name in category for name in ("car", "bus", "truck", "vehicle")) for box in values]
        if vehicle_boxes:
            mask = _box_mask(vehicle_boxes, grid_hw=self.grid_hw, image_size=image_size)
            anchor[1], anchor_valid[1] = mask, bool(mask.any())
            observability[1], observability_valid[1], source_weight[1] = 1.0, bool(mask.any()), 0.5

        geometry = self._geometry_masks(record)
        center_drivable = geometry["drivable_center"]
        if bool(center_drivable.any()):
            anchor[2], anchor_valid[2] = center_drivable, True
            observability[2], observability_valid[2], source_weight[2] = 1.0, True, 0.8
            source_complete[2] = detection_complete
            if detection_complete:
                occupied_boxes = [
                    box
                    for category, values in groups.items()
                    if any(
                        name in category
                        for name in (
                            "car",
                            "bus",
                            "truck",
                            "vehicle",
                            "person",
                            "pedestrian",
                            "rider",
                            "cyclist",
                            "bike",
                            "motor",
                            "obstacle",
                        )
                    )
                    for box in values
                ]
                occupied_mask = _box_mask(
                    occupied_boxes, grid_hw=self.grid_hw, image_size=image_size
                ) * center_drivable
                if bool(occupied_mask.any()):
                    state_target[2], state_valid[2] = 1, True
                else:
                    state_target[2], state_valid[2] = 0, True
        for factor_id, side in ((9, "lane_left"), (11, "lane_left"), (12, "lane_left"), (15, "lane_right"), (17, "lane_right"), (18, "lane_right")):
            if bool(geometry[side].any()):
                anchor[factor_id], anchor_valid[factor_id] = geometry[side], True
                observability[factor_id], observability_valid[factor_id], source_weight[factor_id] = 1.0, True, 0.8
        for obj in objects:
            if not obj.get("poly2d") or not any(
                name in _category(obj) for name in ("lane", "road marking")
            ):
                continue
            points = _points(obj.get("poly2d"))
            if not points:
                continue
            is_left = sum(point[0] for point in points) / len(points) < image_size[0] / 2.0
            solid_factor = 11 if is_left else 17
            turn_factor = 12 if is_left else 18
            style = _attribute(obj, "laneStyle", "lane_style", "style")
            if style:
                if "solid" in style and "dashed" not in style:
                    state_target[solid_factor] = 0
                    state_valid[solid_factor] = True
                elif any(name in style for name in ("dashed", "broken", "double dashed")):
                    state_target[solid_factor] = 1
                    state_valid[solid_factor] = True
                if state_valid[solid_factor]:
                    source_weight[solid_factor] = 1.0
            direction = _attribute(
                obj, "laneDirection", "lane_direction", "direction", "turn"
            )
            expected = "left" if is_left else "right"
            if direction:
                state_target[turn_factor] = (
                    0 if expected in direction and "straight" not in direction else 1
                )
                state_valid[turn_factor] = True
                source_weight[turn_factor] = 1.0
        for factor_id, side, boxes in (
            (10, "drivable_left", [box for box in vehicle_boxes if (box[0] + box[2]) * 0.5 < image_size[0] / 2.0]),
            (16, "drivable_right", [box for box in vehicle_boxes if (box[0] + box[2]) * 0.5 >= image_size[0] / 2.0]),
        ):
            corridor = geometry[side]
            if not bool(corridor.any()):
                continue
            source_complete[factor_id] = detection_complete
            mask = _box_mask(boxes, grid_hw=self.grid_hw, image_size=image_size) * corridor
            if bool(mask.any()):
                anchor[factor_id], anchor_valid[factor_id] = mask, True
                observability[factor_id], observability_valid[factor_id], source_weight[factor_id] = 1.0, True, 1.0
                state_target[factor_id], state_valid[factor_id] = 0, True
            elif detection_complete:
                anchor[factor_id], anchor_valid[factor_id] = corridor, True
                observability[factor_id], observability_valid[factor_id], source_weight[factor_id] = 1.0, True, 0.8
                state_target[factor_id], state_valid[factor_id] = 1, True

        # Null denotes missing localized evidence, not a negative signed state.
        # A reliable anchor is non-null even when its state is red, occupied,
        # unavailable, or otherwise contradictory to the factor name.
        present_valid |= anchor_valid
        absent_valid &= ~anchor_valid

        flat = anchor.flatten(1)
        flat = torch.where(anchor_valid.unsqueeze(-1), flat / flat.sum(-1, keepdim=True).clamp_min(1.0), flat)
        return {
            "factor_anchor_map": flat.view(factor_count, *self.grid_hw),
            "factor_anchor_valid": anchor_valid,
            "factor_state_target": state_target,
            "factor_state_valid": state_valid,
            "factor_present_valid": present_valid,
            "factor_absent_valid": absent_valid,
            "factor_source_complete": source_complete,
            "factor_observability": observability,
            "factor_observability_valid": observability_valid,
            "factor_source_weight": source_weight,
        }
