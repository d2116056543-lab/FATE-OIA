from __future__ import annotations

import json
import math
from functools import lru_cache
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from .bdd100k_grounding import GroundingPaths, load_bdd100k_objects


@lru_cache(maxsize=20000)
def _cached_bdd100k_objects(path: str) -> tuple[dict[str, Any], ...]:
    return tuple(load_bdd100k_objects(path))


@lru_cache(maxsize=20000)
def _cached_drivable_patch_mask(path: str, grid_hw: tuple[int, int]) -> torch.Tensor:
    from PIL import Image
    import numpy as np

    array = np.asarray(Image.open(path))
    if array.ndim == 3:
        array = array.max(axis=-1)
    mask = torch.from_numpy((array > 0).astype("float32"))
    return F.interpolate(mask[None, None], size=grid_hw, mode="nearest")[0, 0].to(torch.uint8)


class MOSAICGroundingObservationBuilder:
    SOURCE_UNKNOWN = 0
    SOURCE_REASON_ANCHOR = 1
    SOURCE_BOX2D = 2
    SOURCE_LANE_POLYLINE = 3
    SOURCE_DRIVABLE_MASK = 4
    _RELIABILITY = {
        SOURCE_UNKNOWN: 0.0,
        SOURCE_REASON_ANCHOR: 0.35,
        SOURCE_BOX2D: 0.80,
        SOURCE_LANE_POLYLINE: 0.75,
        SOURCE_DRIVABLE_MASK: 0.90,
    }
    # An absent instance in an available BDD100K source is useful but weaker
    # than a positive annotation. Keeping these below 0.5 prevents incomplete
    # weak labels from becoming hard negatives.
    _NEGATIVE_RELIABILITY = {
        SOURCE_BOX2D: 0.30,
        SOURCE_LANE_POLYLINE: 0.25,
        SOURCE_DRIVABLE_MASK: 0.35,
    }
    _UNRELIABLE_REASON_ATTRIBUTES = {"near", "left_indicator", "right_indicator"}

    def __init__(self, factors: Sequence[dict[str, Any]], *, grid_hw: tuple[int, int] = (45, 80)) -> None:
        factors = tuple(factors)
        required = {
            "name",
            "type",
            "entity",
            "attribute",
            "spatial",
            "reason_positive_anchors",
            "geometry_sources",
        }
        if not factors or any(not isinstance(factor, dict) or not required <= set(factor) for factor in factors):
            raise ValueError("grounding builder requires complete factor definitions")
        if grid_hw != (45, 80):
            raise ValueError("grounding builder requires the 45x80 patch grid")
        self.factors = factors
        self.grid_hw = grid_hw

    @staticmethod
    def _record_value(record: Any, key: str, default: Any = None) -> Any:
        if record is None:
            return default
        if isinstance(record, dict):
            return record.get(key, default)
        return getattr(record, key, default)

    @staticmethod
    def _flatten_tokens(value: Any) -> set[str]:
        tokens: set[str] = set()
        if isinstance(value, dict):
            for key, item in value.items():
                tokens.update(str(key).lower().replace("-", "_").split())
                tokens.update(MOSAICGroundingObservationBuilder._flatten_tokens(item))
        elif isinstance(value, (list, tuple)):
            for item in value:
                tokens.update(MOSAICGroundingObservationBuilder._flatten_tokens(item))
        elif value is not None:
            normalized = str(value).lower().replace("-", "_").replace("/", "_")
            tokens.update(normalized.replace("_", " ").split())
            tokens.add(normalized)
        return tokens

    @staticmethod
    def _box(object_record: dict[str, Any]) -> tuple[float, float, float, float] | None:
        box = object_record.get("box2d")
        if not isinstance(box, dict):
            return None
        try:
            values = tuple(float(box[key]) for key in ("x1", "y1", "x2", "y2"))
        except (KeyError, TypeError, ValueError):
            return None
        x1, y1, x2, y2 = values
        if not all(math.isfinite(value) for value in values) or x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    @staticmethod
    def _spatial_accepts(spatial: str, x_norm: float, y_norm: float) -> bool:
        if spatial == "upper_front":
            return 0.20 <= x_norm <= 0.80 and y_norm <= 0.60
        if spatial == "front_center":
            return 0.25 <= x_norm <= 0.75 and y_norm >= 0.20
        if spatial == "left_corridor":
            return x_norm <= 0.55 and y_norm >= 0.25
        if spatial == "right_corridor":
            return x_norm >= 0.45 and y_norm >= 0.25
        if spatial == "center_corridor":
            return 0.28 <= x_norm <= 0.72 and y_norm >= 0.30
        return False

    @staticmethod
    def _occupancy_spatial_accepts(spatial: str, x_norm: float, y_norm: float) -> bool:
        # Occupancy is a mutually exclusive lane-level state, not a broad
        # visual prior. Prevent one center object from vetoing all directions.
        if y_norm < 0.35:
            return False
        if spatial == "left_corridor":
            return x_norm < 0.35
        if spatial == "center_corridor":
            return 0.35 <= x_norm <= 0.65
        if spatial == "right_corridor":
            return x_norm > 0.65
        return False

    @staticmethod
    def _entity_matches(entity: str, category: str) -> bool:
        category = category.lower().replace("_", " ")
        vehicle_categories = {"car", "truck", "bus", "train", "vehicle", "motorcycle"}
        if entity == "vehicle":
            return any(token in category for token in vehicle_categories)
        if entity == "pedestrian":
            return "pedestrian" in category or "person" in category
        if entity == "rider":
            return "rider" in category or "bicycle" in category or "cyclist" in category
        if entity == "obstacle":
            return any(token in category for token in vehicle_categories | {"pedestrian", "person", "rider", "bicycle"})
        if entity == "traffic_light":
            return "traffic light" in category
        if entity == "traffic_sign":
            return "traffic sign" in category or "sign" == category.strip()
        return False

    @staticmethod
    def _attribute_matches(attribute: Any, object_record: dict[str, Any]) -> bool:
        if attribute is None:
            return True
        tokens = MOSAICGroundingObservationBuilder._flatten_tokens(
            {"category": object_record.get("category"), "attributes": object_record.get("attributes")}
        )
        required = str(attribute).lower()
        aliases = {
            "left_indicator": {"left_indicator", "left indicator", "indicator_left"},
            "right_indicator": {"right_indicator", "right indicator", "indicator_right"},
            "left_turn": {"left_turn", "left turn"},
            "right_turn": {"right_turn", "right turn"},
            "near": {"near", "close"},
            "solid": {"solid"},
            "red": {"red"},
            "green": {"green"},
        }
        return bool(tokens & aliases.get(required, {required}))

    @classmethod
    def _reason_anchor_is_reliable(cls, factor: dict[str, Any]) -> bool:
        attribute = factor.get("attribute")
        return attribute is None or str(attribute) not in cls._UNRELIABLE_REASON_ATTRIBUTES

    def _box_mask(
        self,
        box: tuple[float, float, float, float],
        image_hw: tuple[int, int],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        image_height, image_width = image_hw
        grid_height, grid_width = self.grid_hw
        x1, y1, x2, y2 = box
        left = max(0, min(grid_width - 1, int(math.floor(x1 / image_width * grid_width))))
        right = max(left + 1, min(grid_width, int(math.ceil(x2 / image_width * grid_width))))
        top = max(0, min(grid_height - 1, int(math.floor(y1 / image_height * grid_height))))
        bottom = max(top + 1, min(grid_height, int(math.ceil(y2 / image_height * grid_height))))
        mask = torch.zeros(self.grid_hw, device=device)
        mask[top:bottom, left:right] = 1.0
        return mask

    def _polyline_mask(
        self,
        vertices: list[Any],
        image_hw: tuple[int, int],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        image_height, image_width = image_hw
        grid_height, grid_width = self.grid_hw
        mask = torch.zeros(self.grid_hw, device=device)
        valid_vertices: list[tuple[float, float]] = []
        for vertex in vertices:
            if isinstance(vertex, (list, tuple)) and len(vertex) >= 2:
                x, y = float(vertex[0]), float(vertex[1])
                if math.isfinite(x) and math.isfinite(y):
                    valid_vertices.append((x, y))
        for start, end in zip(valid_vertices, valid_vertices[1:]):
            distance = max(abs(end[0] - start[0]) / image_width * grid_width, abs(end[1] - start[1]) / image_height * grid_height)
            steps = max(2, int(math.ceil(distance * 2.0)))
            for step in range(steps + 1):
                fraction = step / steps
                x = start[0] + fraction * (end[0] - start[0])
                y = start[1] + fraction * (end[1] - start[1])
                column = max(0, min(grid_width - 1, int(x / image_width * grid_width)))
                row = max(0, min(grid_height - 1, int(y / image_height * grid_height)))
                mask[row, column] = 1.0
        return F.max_pool2d(mask[None, None], kernel_size=3, stride=1, padding=1)[0, 0]

    def _load_drivable_mask(self, value: Any) -> torch.Tensor | None:
        if isinstance(value, torch.Tensor):
            return value.float()
        if isinstance(value, (str, Path)) and Path(value).exists():
            return _cached_drivable_patch_mask(str(Path(value).resolve()), self.grid_hw).float()
        return None

    @staticmethod
    def _objects_and_lanes(record: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        objects = list(MOSAICGroundingObservationBuilder._record_value(record, "objects", []) or [])
        lanes = list(MOSAICGroundingObservationBuilder._record_value(record, "lanes", []) or [])
        label_json = MOSAICGroundingObservationBuilder._record_value(record, "label_json")
        if label_json:
            loaded = _cached_bdd100k_objects(str(Path(label_json).resolve()))
            objects.extend(item for item in loaded if isinstance(item, dict) and "box2d" in item)
            lanes.extend(item for item in loaded if isinstance(item, dict) and "poly2d" in item)
        return objects, lanes

    def _match_box_factor(
        self,
        factor: dict[str, Any],
        objects: list[dict[str, Any]],
        image_hw: tuple[int, int],
        device: torch.device,
    ) -> torch.Tensor | None:
        combined: torch.Tensor | None = None
        for object_record in objects:
            box = self._box(object_record)
            if box is None:
                continue
            center_x = (box[0] + box[2]) / (2.0 * image_hw[1])
            center_y = (box[1] + box[3]) / (2.0 * image_hw[0])
            is_occupancy = factor["entity"] == "occupancy"
            spatial_matches = (
                self._occupancy_spatial_accepts(str(factor["spatial"]), center_x, center_y)
                if is_occupancy
                else self._spatial_accepts(str(factor["spatial"]), center_x, center_y)
            )
            if not spatial_matches:
                continue
            if is_occupancy:
                matched = self._entity_matches("obstacle", str(object_record.get("category", "")))
            else:
                matched = self._entity_matches(str(factor["entity"]), str(object_record.get("category", "")))
            if not matched or not self._attribute_matches(factor["attribute"], object_record):
                continue
            mask = self._box_mask(box, image_hw, device=device)
            combined = mask if combined is None else torch.maximum(combined, mask)
        return combined

    def _match_lane_factor(
        self,
        factor: dict[str, Any],
        lanes: list[dict[str, Any]],
        image_hw: tuple[int, int],
        device: torch.device,
    ) -> torch.Tensor | None:
        combined: torch.Tensor | None = None
        for lane in lanes:
            category = str(lane.get("category", "")).lower()
            if "lane" not in category:
                continue
            if not self._attribute_matches(factor["attribute"], lane):
                continue
            poly_entries = lane.get("poly2d", [])
            if isinstance(poly_entries, dict):
                poly_entries = [poly_entries]
            for poly in poly_entries or []:
                vertices = poly.get("vertices", []) if isinstance(poly, dict) else []
                if not vertices:
                    continue
                mean_x = sum(float(vertex[0]) for vertex in vertices) / len(vertices) / image_hw[1]
                mean_y = sum(float(vertex[1]) for vertex in vertices) / len(vertices) / image_hw[0]
                if not self._spatial_accepts(str(factor["spatial"]), mean_x, mean_y):
                    continue
                mask = self._polyline_mask(vertices, image_hw, device=device)
                combined = mask if combined is None else torch.maximum(combined, mask)
        return combined

    def _match_drivable_factor(self, factor: dict[str, Any], drivable: torch.Tensor, device: torch.device) -> torch.Tensor | None:
        mask = (
            drivable.to(device=device)
            if tuple(drivable.shape) == self.grid_hw
            else F.interpolate(drivable[None, None].to(device=device), size=self.grid_hw, mode="nearest")[0, 0]
        )
        y, x = torch.meshgrid(
            torch.linspace(0.0, 1.0, self.grid_hw[0], device=device),
            torch.linspace(0.0, 1.0, self.grid_hw[1], device=device),
            indexing="ij",
        )
        spatial = str(factor["spatial"])
        if spatial == "center_corridor":
            region = (x >= 0.28) & (x <= 0.72) & (y >= 0.30)
        elif spatial == "left_corridor":
            region = (x <= 0.55) & (y >= 0.25)
        elif spatial == "right_corridor":
            region = (x >= 0.45) & (y >= 0.25)
        else:
            region = torch.ones_like(mask, dtype=torch.bool)
        result = (mask > 0).float() * region.float()
        return result if result.any() else None

    def __call__(
        self,
        reason_targets: torch.Tensor,
        grounding_records: Sequence[Any],
        *,
        split: str,
    ) -> dict[str, torch.Tensor]:
        if split != "train":
            raise ValueError("MOSAIC grounding observation builder is train-only")
        if reason_targets.ndim != 2 or reason_targets.shape[1] != 21:
            raise ValueError("MOSAIC grounding builder expects reason targets [B,21]")
        if len(grounding_records) != reason_targets.shape[0]:
            raise ValueError("grounding record count must match the reason batch")
        if not reason_targets.is_floating_point():
            raise ValueError("reason targets must be floating-point multi-hot labels")

        batch_size = reason_targets.shape[0]
        factor_count = len(self.factors)
        device = reason_targets.device
        presence_target = torch.zeros(batch_size, factor_count, device=device)
        presence_mask = torch.zeros_like(presence_target)
        visibility_target = torch.zeros_like(presence_target)
        visibility_mask = torch.zeros_like(presence_target)
        reliability = torch.zeros_like(presence_target)
        geometry_mask = torch.zeros(batch_size, factor_count, *self.grid_hw, device=device)
        geometry_valid = torch.zeros_like(presence_target)
        source_code = torch.zeros(batch_size, factor_count, dtype=torch.long, device=device)
        weak_negative_mask = torch.zeros_like(presence_target)

        for batch_index, record in enumerate(grounding_records):
            positive_reasons = set(torch.nonzero(reason_targets[batch_index] > 0.5, as_tuple=False).flatten().tolist())
            if record is None:
                for factor_index, factor in enumerate(self.factors):
                    if positive_reasons & set(factor["reason_positive_anchors"]) and self._reason_anchor_is_reliable(factor):
                        presence_target[batch_index, factor_index] = 1.0
                        presence_mask[batch_index, factor_index] = 1.0
                        visibility_target[batch_index, factor_index] = 1.0
                        visibility_mask[batch_index, factor_index] = 1.0
                        reliability[batch_index, factor_index] = self._RELIABILITY[self.SOURCE_REASON_ANCHOR]
                        source_code[batch_index, factor_index] = self.SOURCE_REASON_ANCHOR
                continue
            raw_image_hw = self._record_value(record, "image_size", (720, 1280))
            if not isinstance(raw_image_hw, (list, tuple)) or len(raw_image_hw) != 2:
                raise ValueError("grounding image_size must be (height,width)")
            image_hw = (int(raw_image_hw[0]), int(raw_image_hw[1]))
            objects, lanes = self._objects_and_lanes(record)
            drivable_value = self._record_value(record, "drivable_mask")
            if drivable_value is None:
                drivable_value = self._record_value(record, "drivable_map")
            drivable = self._load_drivable_mask(drivable_value)
            label_json = self._record_value(record, "label_json")
            box_source_available = label_json is not None or self._record_value(record, "objects") is not None
            lane_source_available = label_json is not None or self._record_value(record, "lanes") is not None

            for factor_index, factor in enumerate(self.factors):
                matched_mask: torch.Tensor | None = None
                matched_source = self.SOURCE_UNKNOWN
                available_source = self.SOURCE_UNKNOWN
                sources = set(factor["geometry_sources"])
                if "box2d" in sources:
                    if box_source_available:
                        available_source = self.SOURCE_BOX2D
                    matched_mask = self._match_box_factor(factor, objects, image_hw, device)
                    if matched_mask is not None:
                        matched_source = self.SOURCE_BOX2D
                if matched_mask is None and "lane_polyline" in sources:
                    if lane_source_available:
                        available_source = self.SOURCE_LANE_POLYLINE
                    matched_mask = self._match_lane_factor(factor, lanes, image_hw, device)
                    if matched_mask is not None:
                        matched_source = self.SOURCE_LANE_POLYLINE
                if matched_mask is None and "drivable_mask" in sources and drivable is not None:
                    available_source = self.SOURCE_DRIVABLE_MASK
                    matched_mask = self._match_drivable_factor(factor, drivable, device)
                    if matched_mask is not None:
                        matched_source = self.SOURCE_DRIVABLE_MASK

                if matched_mask is not None:
                    presence_target[batch_index, factor_index] = 1.0
                    presence_mask[batch_index, factor_index] = 1.0
                    visibility_target[batch_index, factor_index] = 1.0
                    visibility_mask[batch_index, factor_index] = 1.0
                    reliability[batch_index, factor_index] = self._RELIABILITY[matched_source]
                    geometry_mask[batch_index, factor_index] = matched_mask
                    geometry_valid[batch_index, factor_index] = 1.0
                    source_code[batch_index, factor_index] = matched_source
                    continue

                # Only attribute-free factors support absence inference from a
                # complete geometry source. Missing color/distance/indicator
                # attributes remain unknown, never hard negatives.
                if available_source != self.SOURCE_UNKNOWN and factor.get("attribute") is None:
                    presence_target[batch_index, factor_index] = 0.0
                    presence_mask[batch_index, factor_index] = 1.0
                    visibility_target[batch_index, factor_index] = 1.0
                    visibility_mask[batch_index, factor_index] = 1.0
                    reliability[batch_index, factor_index] = self._NEGATIVE_RELIABILITY[available_source]
                    source_code[batch_index, factor_index] = available_source
                    weak_negative_mask[batch_index, factor_index] = 1.0
                    continue

                if positive_reasons & set(factor["reason_positive_anchors"]) and self._reason_anchor_is_reliable(factor):
                    presence_target[batch_index, factor_index] = 1.0
                    presence_mask[batch_index, factor_index] = 1.0
                    visibility_target[batch_index, factor_index] = 1.0
                    visibility_mask[batch_index, factor_index] = 1.0
                    reliability[batch_index, factor_index] = self._RELIABILITY[self.SOURCE_REASON_ANCHOR]
                    source_code[batch_index, factor_index] = self.SOURCE_REASON_ANCHOR

        return {
            "presence_target": presence_target,
            "presence_mask": presence_mask,
            "visibility_target": visibility_target,
            "visibility_mask": visibility_mask,
            "source_reliability": reliability,
            "geometry_mask": geometry_mask,
            "geometry_mask_valid": geometry_valid,
            "source_code": source_code,
            "weak_negative_mask": weak_negative_mask,
        }
