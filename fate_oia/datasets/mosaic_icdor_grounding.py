from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F


def is_supported_lane_direction(value: Any) -> bool:
    """BDD100K lanes use only ``parallel`` or ``vertical`` direction labels."""
    return str(value).lower() in {"parallel", "vertical"}


def reliable_absence_observation(*, source_complete: bool, region_visible: bool, no_footpoint: bool) -> dict[str, float | bool]:
    """Encode a real negative observation, not an unknown missing source."""
    known = bool(source_complete and region_visible and no_footpoint)
    return {"presence": 0.0, "observability": 1.0 if known else 0.0, "known": known}


def object_corridor_overlap(
    box: Mapping[str, Any], *, corridor: tuple[float, float, float, float]
) -> float:
    """Footprint/box corridor overlap used for explicit occupancy predicates."""
    try:
        x1, y1, x2, y2 = (float(box[key]) for key in ("x1", "y1", "x2", "y2"))
    except (KeyError, TypeError, ValueError):
        return 0.0
    cx1, cy1, cx2, cy2 = corridor
    foot_x, foot_y = (x1 + x2) * 0.5, y2
    footprint = 1.0 if cx1 <= foot_x <= cx2 and cy1 <= foot_y <= cy2 else 0.0
    intersection = max(0.0, min(cx2, x2) - max(cx1, x1)) * max(0.0, min(cy2, y2) - max(cy1, y1))
    box_area = max((x2 - x1) * (y2 - y1), 1e-6)
    return max(footprint, intersection / box_area)


class ICDORGroundingObservationBuilder:
    """Train-only BDD100K observations with explicit unknown/weak-negative states.

    It never receives reason labels and never turns an unavailable source into a
    hard negative. Geometry is supervision/audit only, never a test forward input.
    """

    _RELIABLE_SOURCE_POLICY = "reliable_if_source_complete"
    _RELIABLE_ATTRIBUTE_POLICY = "reliable_if_attribute_complete"
    _UNKNOWN_POLICIES = frozenset(("unknown_without_depth", "unknown_without_direct_visual_evidence"))
    _DEFAULT_POLICY = "weak_if_source_complete"

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
        if "traffic_light" in name or name in {"red_light_visible", "green_light_visible"}:
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

    @classmethod
    def _negative_policy(cls, factor: Mapping[str, Any]) -> str:
        policy = str(factor.get("observability_policy", factor.get("negative_policy", cls._DEFAULT_POLICY)))
        valid = {cls._RELIABLE_SOURCE_POLICY, cls._RELIABLE_ATTRIBUTE_POLICY, *cls._UNKNOWN_POLICIES, cls._DEFAULT_POLICY}
        if policy not in valid:
            raise ValueError(f"unsupported IC-DOR negative policy: {policy}")
        return policy

    @staticmethod
    def _constraint_match(attributes: Mapping[str, Any], constraints: Mapping[str, Any]) -> tuple[bool, bool]:
        """Return ``(matches, complete)`` without inventing missing attributes."""
        keys = {
            "trafficLightColor": "traffic_light_color",
            "laneStyle": "lane_style",
            "laneDirection": "lane_direction",
            "areaType": "area_type",
        }
        complete = True
        for name, expected in constraints.items():
            if name == "proxy":
                continue
            key = keys.get(str(name))
            if key is None:
                raise ValueError(f"unsupported IC-DOR attribute constraint: {name}")
            actual = attributes.get(key)
            if actual is None:
                return False, False
            if str(actual).lower() != str(expected).lower():
                return False, complete
        return True, complete

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
                policy = self._negative_policy(factor)
                constraints = factor.get("source_attributes", factor.get("attribute_constraints", {}))
                if not isinstance(constraints, Mapping):
                    raise ValueError("IC-DOR factor source_attributes must be a mapping")
                positive_mask: torch.Tensor | None = None
                available = False
                attribute_complete = True
                if "box2d" in sources and objects_available:
                    available = True
                    for item in objects:
                        if not isinstance(item, dict):
                            continue
                        attributes = parse_bdd100k_attributes(item)
                        if self._matches(name, str(item.get("category", ""))):
                            attributes_match, complete = self._constraint_match(attributes, constraints)
                            attribute_complete = attribute_complete and complete
                            # Occluded/truncated traffic-control boxes cannot
                            # certify either presence or a reliable absence.
                            if "light" in name and (attributes.get("occluded") is True or attributes.get("truncated") is True):
                                attribute_complete = False
                                continue
                            if not attributes_match:
                                continue
                            box = item.get("box2d", {})
                            if ("occupied" in name or "obstacle" in name) and isinstance(box, Mapping):
                                regions = factor.get("weak_regions", [])
                                region = str(regions[0]) if isinstance(regions, (list, tuple)) and regions else "front_center"
                                region_mask = self._region_mask(region, self.grid_hw, device)
                                ys, xs = torch.nonzero(region_mask > 0, as_tuple=True)
                                if not ys.numel():
                                    continue
                                corridor = (
                                    float(xs.min()) / self.grid_hw[1] * image_hw[1],
                                    float(ys.min()) / self.grid_hw[0] * image_hw[0],
                                    float(xs.max() + 1) / self.grid_hw[1] * image_hw[1],
                                    float(ys.max() + 1) / self.grid_hw[0] * image_hw[0],
                                )
                                if object_corridor_overlap(box, corridor=corridor) < 0.10:
                                    continue
                            candidate = self._box_mask(box, image_hw, self.grid_hw, device)
                            if candidate is not None:
                                candidate = self._restrict_to_declared_region(candidate, factor, device)
                                if bool(candidate.any()):
                                    positive_mask = candidate if positive_mask is None else torch.maximum(positive_mask, candidate)
                elif "lane_polyline" in sources and lanes_available:
                    available = True
                    for lane in lanes:
                        if not isinstance(lane, dict) or "lane" not in str(lane.get("category", "")).lower():
                            continue
                        attributes = parse_bdd100k_attributes(lane)
                        lane_types = attributes.get("lane_types")
                        if attributes.get("lane_direction") != "parallel" or (
                            isinstance(lane_types, (list, tuple)) and any(str(value).lower() == "crosswalk" for value in lane_types)
                        ):
                            continue
                        attributes_match, complete = self._constraint_match(attributes, constraints)
                        attribute_complete = attribute_complete and complete
                        if not attributes_match:
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
                    attributes_match, complete = self._constraint_match(
                        {"area_type": "direct"}, constraints
                    )
                    attribute_complete = attribute_complete and complete
                    if not attributes_match:
                        continue
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
                    # The ontology, not a generic default, decides whether a
                    # complete source certifies absence. Unsupported proxies
                    # remain unknown rather than becoming fabricated negatives.
                    if policy == self._RELIABLE_SOURCE_POLICY or (
                        policy == self._RELIABLE_ATTRIBUTE_POLICY and attribute_complete
                    ):
                        presence_known_mask[row, column] = 1.0
                        # Reliable absence is evidence about both state and
                        # observability: p=0, v=1.  Leaving visibility unknown
                        # made no-lane/no-obstacle predicates impossible to
                        # learn despite complete BDD100K coverage.
                        visibility_target[row, column] = 1.0
                        visibility_known_mask[row, column] = 1.0
                    elif policy not in self._UNKNOWN_POLICIES:
                        weak_negative_mask[row, column] = 1.0
        return {"presence_target": presence_target, "presence_known_mask": presence_known_mask, "visibility_target": visibility_target, "visibility_known_mask": visibility_known_mask, "weak_negative_mask": weak_negative_mask, "geometry_known_mask": geometry_known_mask, "geometry_masks": geometry_masks, "source_available": source_available}


def parse_bdd100k_attributes(row: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize BDD100K attributes without inventing missing semantics."""
    raw = row.get("attributes", {}) if isinstance(row, Mapping) else {}
    raw = raw if isinstance(raw, Mapping) else {}

    def _text(name: str) -> str | None:
        value = raw.get(name, row.get(name) if isinstance(row, Mapping) else None)
        return str(value).lower() if value not in (None, "", "unknown") else None

    def _bool(name: str) -> bool | None:
        value = raw.get(name, row.get(name) if isinstance(row, Mapping) else None)
        if value is None:
            return None
        if isinstance(value, str):
            if value.lower() in {"true", "1", "yes"}:
                return True
            if value.lower() in {"false", "0", "no"}:
                return False
            return None
        return bool(value)

    return {
        "traffic_light_color": _text("trafficLightColor"),
        "lane_direction": _text("laneDirection"),
        "lane_style": _text("laneStyle"),
        "lane_types": raw.get("laneTypes"),
        "area_type": _text("areaType"),
        "occluded": _bool("occluded"),
        "truncated": _bool("truncated"),
        "source_attributes_complete": all(
            raw.get(name) is not None for name in ("occluded", "truncated")
        ),
    }


def corridor_occupancy_observation(
    boxes: Sequence[tuple[float, float, float, float]],
    *,
    corridor: tuple[float, float, float, float],
) -> dict[str, Any]:
    """Return occupancy with a reliable negative only when source is complete."""
    cx0, cy0, cx1, cy1 = corridor
    occupied = False
    for x0, y0, x1, y1 in boxes:
        overlap_x = max(0.0, min(cx1, x1) - max(cx0, x0))
        overlap_y = max(0.0, min(cy1, y1) - max(cy0, y0))
        corridor_area = max((cx1 - cx0) * (cy1 - cy0), 1e-6)
        if overlap_x * overlap_y / corridor_area >= 0.05:
            occupied = True
            break
    return {
        "presence": 1.0 if occupied else 0.0,
        "reliable_negative": not occupied,
        "source_complete": True,
        "overlap_threshold": 0.05,
    }
