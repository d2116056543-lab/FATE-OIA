from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor


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


def _box_mask(
    boxes: list[tuple[float, float, float, float]],
    *,
    grid_hw: tuple[int, int] = (45, 80),
    image_hw: tuple[int, int] = (720, 1280),
) -> Tensor:
    height, width = grid_hw
    mask = torch.zeros(height, width, dtype=torch.float32)
    for x1, y1, x2, y2 in boxes:
        gx1 = max(0, min(width - 1, int(x1 / image_hw[1] * width)))
        gx2 = max(gx1 + 1, min(width, int((x2 + 1) / image_hw[1] * width)))
        gy1 = max(0, min(height - 1, int(y1 / image_hw[0] * height)))
        gy2 = max(gy1 + 1, min(height, int((y2 + 1) / image_hw[0] * height)))
        mask[gy1:gy2, gx1:gx2] = 1.0
    return mask


class METERTypedTargetBuilder:
    """Conservative typed weak targets; unknown never becomes a negative."""

    def __init__(self, schema_path: str | Path) -> None:
        raw = yaml.safe_load(Path(schema_path).read_text(encoding="utf-8"))
        self.factors = list(raw["factors"])
        if [int(row["id"]) for row in self.factors] != list(range(21)):
            raise ValueError("TESA schema must contain ordered factor IDs 0..20")

    def build(self, record: dict[str, Any]) -> dict[str, Tensor]:
        factor_count = len(self.factors)
        anchor = torch.zeros(factor_count, 45, 80)
        anchor_valid = torch.zeros(factor_count, dtype=torch.bool)
        state_target = torch.full((factor_count,), -1, dtype=torch.long)
        state_valid = torch.zeros(factor_count, dtype=torch.bool)
        observability = torch.zeros(factor_count)
        observability_valid = torch.zeros(factor_count, dtype=torch.bool)
        source_weight = torch.zeros(factor_count)

        objects = list(record.get("objects", []))
        groups: dict[str, list[tuple[float, float, float, float]]] = {}
        for obj in objects:
            box = _box(obj)
            if box is not None:
                groups.setdefault(_category(obj), []).append(box)

        category_rules = {
            3: ("traffic light",),
            4: ("traffic sign",),
            5: ("car", "bus", "truck", "vehicle"),
            6: ("person", "pedestrian"),
            7: ("rider", "cyclist", "bike", "motor"),
            8: ("obstacle", "other"),
        }
        for factor_id, categories in category_rules.items():
            boxes = [
                box
                for category, values in groups.items()
                if any(name in category for name in categories)
                for box in values
            ]
            mask = _box_mask(boxes)
            if bool(mask.any()):
                anchor[factor_id] = mask
                anchor_valid[factor_id] = True
                state_target[factor_id] = 0  # present
                state_valid[factor_id] = True
                observability[factor_id] = 1.0
                observability_valid[factor_id] = True
                source_weight[factor_id] = 1.0
            elif bool(record.get("source_complete", False)):
                # Absence is valid only for complete object sources.
                state_target[factor_id] = 1
                state_valid[factor_id] = True
                observability[factor_id] = 1.0
                observability_valid[factor_id] = True
                source_weight[factor_id] = 0.8

        # Corridor/boundary factors receive conservative region anchors only
        # when lane/drivable metadata exists; semantic state remains unknown
        # unless an explicit style/availability annotation is present.
        has_lane = bool(record.get("lanes") or record.get("polylines"))
        has_drivable = bool(record.get("drivable_map_path"))
        if has_lane or has_drivable:
            rows = torch.arange(45).view(-1, 1).expand(45, 80)
            cols = torch.arange(80).view(1, -1).expand(45, 80)
            regions = {
                "left": ((cols < 40) & (rows >= 16)).float(),
                "right": ((cols >= 40) & (rows >= 16)).float(),
                "center": ((cols >= 24) & (cols < 56) & (rows >= 16)).float(),
            }
            for factor_id in (9, 10, 11, 12):
                anchor[factor_id] = regions["left"]
                anchor_valid[factor_id] = True
                observability[factor_id] = 1.0
                observability_valid[factor_id] = True
                source_weight[factor_id] = 0.5
            for factor_id in (15, 16, 17, 18):
                anchor[factor_id] = regions["right"]
                anchor_valid[factor_id] = True
                observability[factor_id] = 1.0
                observability_valid[factor_id] = True
                source_weight[factor_id] = 0.5
            anchor[2] = regions["center"]
            anchor_valid[2] = True
            observability[2] = 1.0
            observability_valid[2] = True
            source_weight[2] = 0.5

        # Normalize valid anchors; invalid rows remain zero and are masked.
        flat = anchor.flatten(1)
        flat = torch.where(
            anchor_valid.unsqueeze(-1),
            flat / flat.sum(-1, keepdim=True).clamp_min(1.0),
            flat,
        )
        return {
            "factor_anchor_map": flat.view(factor_count, 45, 80),
            "factor_anchor_valid": anchor_valid,
            "factor_state_target": state_target,
            "factor_state_valid": state_valid,
            "factor_observability": observability,
            "factor_observability_valid": observability_valid,
            "factor_source_weight": source_weight,
        }
