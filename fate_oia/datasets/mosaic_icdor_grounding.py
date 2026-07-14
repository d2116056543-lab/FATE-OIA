from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F


class ICDORGroundingObservationBuilder:
    """Train-only BDD100K observations with explicit unknown/weak-negative states.

    It never receives reason labels and never turns an unavailable source into a
    hard negative. Geometry is supervision/audit only, never a test forward input.
    """

    def __init__(self, factors: Sequence[dict[str, Any]], *, grid_hw: tuple[int, int] = (45, 80)) -> None:
        if not factors or any(not {"name", "type", "grounding_sources"} <= set(item) for item in factors):
            raise ValueError("IC-DOR grounding factors require name/type/grounding_sources")
        self.factors = tuple(factors)
        self.grid_hw = grid_hw

    @staticmethod
    def _box_mask(box: dict[str, Any], image_hw: tuple[int, int], grid_hw: tuple[int, int], device: torch.device) -> torch.Tensor | None:
        try:
            x1, y1, x2, y2 = (float(box[key]) for key in ("x1", "y1", "x2", "y2"))
        except (KeyError, TypeError, ValueError):
            return None
        height, width = image_hw
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            return None
        rows, cols = grid_hw
        left, right = int(x1 / width * cols), max(int((x2 + width - 1) / width * cols), 1)
        top, bottom = int(y1 / height * rows), max(int((y2 + height - 1) / height * rows), 1)
        mask = torch.zeros(grid_hw, device=device)
        mask[max(top, 0):min(bottom, rows), max(left, 0):min(right, cols)] = 1.0
        return mask if bool(mask.any()) else None

    @staticmethod
    def _matches(name: str, category: str) -> bool:
        value = category.lower().replace("_", " ")
        if "traffic_light" in name:
            return "traffic light" in value
        if "traffic_sign" in name:
            return "traffic sign" in value or value.strip() == "sign"
        if "pedestrian" in name:
            return "pedestrian" in value or "person" in value
        if "rider" in name:
            return any(token in value for token in ("rider", "bicycle", "cyclist"))
        if "vehicle" in name or "occupied" in name or "obstacle" in name:
            return any(token in value for token in ("car", "truck", "bus", "train", "motorcycle", "pedestrian", "person", "rider", "bicycle"))
        return False

    @staticmethod
    def _region_mask(region: str | None, grid_hw: tuple[int, int], device: torch.device) -> torch.Tensor:
        """Return the declared ego region; unknown names deliberately fall back to all patches."""
        rows, cols = grid_hw
        y = (torch.arange(rows, device=device, dtype=torch.float32) + 0.5) / rows
        x = (torch.arange(cols, device=device, dtype=torch.float32) + 0.5) / cols
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        if region == "upper_front":
            keep = (yy <= 0.45) & (xx >= 0.25) & (xx <= 0.75)
        elif region == "front_center":
            keep = (yy >= 0.35) & (xx >= 0.25) & (xx <= 0.75)
        elif region == "left_corridor":
            keep = (yy >= 0.35) & (xx <= 0.45)
        elif region == "right_corridor":
            keep = (yy >= 0.35) & (xx >= 0.55)
        elif region == "center_corridor":
            keep = (yy >= 0.35) & (xx >= 0.30) & (xx <= 0.70)
        else:
            keep = torch.ones(grid_hw, dtype=torch.bool, device=device)
        return keep.float()

    def _restrict_to_declared_region(self, mask: torch.Tensor, factor: dict[str, Any], device: torch.device) -> torch.Tensor:
        regions = factor.get("weak_regions", [])
        region = str(regions[0]) if isinstance(regions, (list, tuple)) and regions else None
        return mask * self._region_mask(region, self.grid_hw, device)

    def _drivable_mask(self, value: Any, device: torch.device) -> torch.Tensor | None:
        if isinstance(value, (str, bytes)):
            try:
                with Image.open(value) as image:
                    mask = torch.from_numpy(np.asarray(image.convert("L"), dtype=np.float32).copy()).to(device)
            except (FileNotFoundError, OSError, ValueError):
                return None
        elif isinstance(value, torch.Tensor):
            mask = value.float().to(device)
        else:
            return None
        if mask.ndim != 2:
            return None
        return F.interpolate(mask[None, None], size=self.grid_hw, mode="nearest")[0, 0].gt(0).float()

    def __call__(self, records: Sequence[dict[str, Any] | None], *, device: torch.device, split: str) -> dict[str, torch.Tensor]:
        if split != "train":
            raise ValueError("IC-DOR grounding observation is train-only")
        batch, factor_count = len(records), len(self.factors)
        shape = (batch, factor_count)
        presence_target = torch.zeros(shape, device=device)
        presence_known_mask = torch.zeros(shape, device=device)
        visibility_target = torch.zeros(shape, device=device)
        visibility_known_mask = torch.zeros(shape, device=device)
        weak_negative_mask = torch.zeros(shape, device=device)
        geometry_known_mask = torch.zeros(shape, device=device)
        geometry_masks = torch.zeros(batch, factor_count, *self.grid_hw, device=device)
        source_available = torch.zeros(shape, dtype=torch.bool, device=device)
        for row, record in enumerate(records):
            if record is None:
                continue
            raw_hw = record.get("image_size", (720, 1280))
            if not isinstance(raw_hw, (list, tuple)) or len(raw_hw) != 2:
                continue
            image_hw = (int(raw_hw[0]), int(raw_hw[1]))
            objects_available = "objects" in record
            lanes_available = "lanes" in record
            drivable_available = "drivable_mask" in record or "drivable_map" in record
            objects = record.get("objects", []) if objects_available else []
            lanes = record.get("lanes", []) if lanes_available else []
            drivable = self._drivable_mask(record.get("drivable_mask", record.get("drivable_map")), device)
            for column, factor in enumerate(self.factors):
                name, kind = str(factor["name"]), str(factor["type"])
                sources = set(factor["grounding_sources"])
                positive_mask: torch.Tensor | None = None
                available = False
                if "box2d" in sources and objects_available:
                    available = True
                    for item in objects:
                        if isinstance(item, dict) and self._matches(name, str(item.get("category", ""))):
                            candidate = self._box_mask(item.get("box2d", {}), image_hw, self.grid_hw, device)
                            if candidate is not None:
                                candidate = self._restrict_to_declared_region(candidate, factor, device)
                                if bool(candidate.any()):
                                    positive_mask = candidate if positive_mask is None else torch.maximum(positive_mask, candidate)
                elif "lane_polyline" in sources and lanes_available:
                    available = True
                    for lane in lanes:
                        if not isinstance(lane, dict) or "lane" not in str(lane.get("category", "")).lower():
                            continue
                        vertices = []
                        for poly in lane.get("poly2d", []) or []:
                            if isinstance(poly, dict):
                                vertices.extend(poly.get("vertices", []) or [])
                        if len(vertices) >= 2:
                            mask = torch.zeros(self.grid_hw, device=device)
                            for vertex in vertices:
                                if isinstance(vertex, (tuple, list)) and len(vertex) >= 2:
                                    x, y = float(vertex[0]), float(vertex[1])
                                    mask[min(max(int(y / image_hw[0] * self.grid_hw[0]), 0), self.grid_hw[0]-1), min(max(int(x / image_hw[1] * self.grid_hw[1]), 0), self.grid_hw[1]-1)] = 1.0
                            if bool(mask.any()):
                                candidate = F.max_pool2d(mask[None, None], 3, 1, 1)[0, 0]
                                candidate = self._restrict_to_declared_region(candidate, factor, device)
                                if bool(candidate.any()):
                                    positive_mask = candidate if positive_mask is None else torch.maximum(positive_mask, candidate)
                elif "drivable_mask" in sources and drivable_available and drivable is not None:
                    available = True
                    candidate = self._restrict_to_declared_region(drivable, factor, device)
                    positive_mask = candidate if bool(candidate.any()) else None
                source_available[row, column] = available
                if positive_mask is not None and bool(positive_mask.any()):
                    presence_target[row, column] = 1.0
                    presence_known_mask[row, column] = 1.0
                    visibility_target[row, column] = 1.0
                    visibility_known_mask[row, column] = 1.0
                    geometry_known_mask[row, column] = 1.0
                    geometry_masks[row, column] = positive_mask
                elif available:
                    # Complete source but no matching item: weak negative only.
                    weak_negative_mask[row, column] = 1.0
        return {"presence_target": presence_target, "presence_known_mask": presence_known_mask, "visibility_target": visibility_target, "visibility_known_mask": visibility_known_mask, "weak_negative_mask": weak_negative_mask, "geometry_known_mask": geometry_known_mask, "geometry_masks": geometry_masks, "source_available": source_available}
