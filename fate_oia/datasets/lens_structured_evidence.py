from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
import yaml

from fate_oia.datasets.bdd100k_grounding import BDD100KGroundingIndex, load_bdd100k_objects
from fate_oia.grounding.mask_builder import drivable_map_to_mask, objects_to_mask


POSITIVE, COUNTER, UNKNOWN = 0, 1, 2


@dataclass(frozen=True)
class StructuredEvidence:
    state_target: torch.Tensor
    state_mask: torch.Tensor
    map_target: torch.Tensor
    map_mask: torch.Tensor
    source_reliability: torch.Tensor
    source_id: torch.Tensor
    source_complete: torch.Tensor
    coverage: dict[str, int]


class LENSStructuredEvidenceBuilder:
    """Fail closed: lack of an explicit, complete source always remains unknown."""

    def __init__(self, schema_path: str | Path, grid_hw: tuple[int, int] = (45, 80)) -> None:
        raw = yaml.safe_load(Path(schema_path).read_text(encoding="utf-8")) or {}
        self.schema = {int(k): v for k, v in (raw.get("reasons") or {}).items()}
        if len(self.schema) != 21:
            raise ValueError("LENS schema must define all 21 reasons")
        self.grid_hw = grid_hw

    def build(self, records: list[dict[str, Any]]) -> StructuredEvidence:
        b, r, n = len(records), 21, self.grid_hw[0] * self.grid_hw[1]
        state = torch.zeros(b, r, 3)
        state[..., UNKNOWN] = 1.0
        state_mask = torch.zeros(b, r)
        map_target = torch.zeros(b, r, n)
        map_mask = torch.zeros(b, r)
        reliability = torch.zeros(b, r)
        source_id = torch.full((b, r), -1, dtype=torch.long)
        complete = torch.zeros(b, r, dtype=torch.bool)
        for bi, record in enumerate(records):
            attributes = record.get("explicit_attributes", {})
            sources = record.get("complete_sources", {})
            for ri, spec in self.schema.items():
                support = any(bool(attributes.get(name, False)) for name in spec.get("support_sources", []))
                counter = any(bool(attributes.get(name, False)) for name in spec.get("counter_sources", []))
                needs_complete = bool(spec.get("complete_source_required", True))
                is_complete = bool(sources.get(spec.get("default_region", ""), False))
                complete[bi, ri] = is_complete
                if support and not counter:
                    state[bi, ri] = torch.tensor([1.0, 0.0, 0.0])
                    state_mask[bi, ri] = 1.0; reliability[bi, ri] = 1.0; source_id[bi, ri] = 1
                elif counter and (is_complete or not needs_complete):
                    state[bi, ri] = torch.tensor([0.0, 1.0, 0.0])
                    state_mask[bi, ri] = 1.0; reliability[bi, ri] = 0.9; source_id[bi, ri] = 2
                # Unknown is deliberately not made a hard target.
                selected_sources = spec.get("support_sources", []) if support and not counter else spec.get("counter_sources", []) if counter else []
                source_maps = record.get("attribute_maps", {})
                available = [torch.as_tensor(source_maps[name]).float().reshape(-1) for name in selected_sources if name in source_maps]
                if available:
                    merged = torch.stack(available).amax(0)
                    if merged.numel() == n and float(merged.sum()) > 0:
                        map_target[bi, ri] = merged / merged.sum().clamp_min(1e-8)
                        map_mask[bi, ri] = 1.0
        return StructuredEvidence(state, state_mask, map_target, map_mask, reliability, source_id, complete, {
            "known": int(state_mask.sum().item()), "unknown": int((1 - state_mask).sum().item())
        })


class LENSStructuredRecordAdapter:
    """Converts only explicit BDD100K geometry/attributes into conservative records."""

    def __init__(self, bdd100k_root: str | Path, grid_hw: tuple[int, int] = (45, 80)) -> None:
        self.index = BDD100KGroundingIndex(bdd100k_root)
        self.grid_hw = grid_hw

    @staticmethod
    def _box_center(obj: dict[str, Any]) -> tuple[float, float]:
        box = obj.get("box2d") or {}
        return ((float(box.get("x1", 0)) + float(box.get("x2", 0))) / 2560.0,
                (float(box.get("y1", 0)) + float(box.get("y2", 0))) / 1440.0)

    @lru_cache(maxsize=20000)
    def build_record(self, file_name: str) -> dict[str, Any]:
        paths = self.index.lookup(file_name)
        objects = load_bdd100k_objects(paths.label_json) if paths.label_json else []
        attributes: dict[str, bool] = {}
        attribute_maps: dict[str, torch.Tensor] = {}
        complete = {
            "front_center": bool(paths.label_json),
            "upper_traffic_region": bool(paths.label_json),
            "left_corridor": bool(paths.drivable_map),
            "right_corridor": bool(paths.drivable_map),
        }
        categories: dict[str, list[dict[str, Any]]] = {}
        source_objects: dict[str, list[dict[str, Any]]] = {}
        for obj in objects:
            cat = str(obj.get("category", "")).lower()
            categories.setdefault(cat, []).append(obj)
            x, y = self._box_center(obj)
            attrs = obj.get("attributes") or {}
            if "traffic light" in cat:
                attributes["traffic_light"] = True
                source_objects.setdefault("traffic_light",[]).append(obj)
                color = str(attrs.get("trafficLightColor", attrs.get("color", ""))).lower()
                if color in {"green", "red", "yellow"}:
                    attributes[f"traffic_light_color_{color}"] = True
                    source_objects.setdefault(f"traffic_light_color_{color}",[]).append(obj)
            if "traffic sign" in cat:
                attributes["traffic_sign"] = True
                source_objects.setdefault("traffic_sign",[]).append(obj)
            if any(name in cat for name in ("car", "truck", "bus", "vehicle")):
                source="left_vehicle" if x < 0.4 else "right_vehicle" if x > 0.6 else "front_vehicle"
                attributes[source] = True; source_objects.setdefault(source,[]).append(obj)
            if "person" in cat and 0.25 <= x <= 0.75 and y >= 0.25:
                attributes["front_person"] = True
                source_objects.setdefault("front_person",[]).append(obj)
            if any(name in cat for name in ("rider", "bike", "motor")) and 0.25 <= x <= 0.75 and y >= 0.25:
                attributes["front_rider"] = True
                source_objects.setdefault("front_rider",[]).append(obj)
            if "obstacle" in cat and 0.25 <= x <= 0.75:
                attributes["front_obstacle"] = True
                source_objects.setdefault("front_obstacle",[]).append(obj)
            lane_style = str(attrs.get("laneStyle", attrs.get("style", ""))).lower()
            if "lane" in cat and lane_style in {"solid", "dashed"}:
                source=("left" if x < 0.5 else "right") + f"_lane_style_{lane_style}"
                attributes[source] = True; source_objects.setdefault(source,[]).append(obj)
        for source, selected_objects in source_objects.items():
            attribute_maps[source] = objects_to_mask(selected_objects, (1280, 720), self.grid_hw).reshape(-1)
        if paths.drivable_map:
            drive = drivable_map_to_mask(paths.drivable_map, self.grid_hw)
            width = drive.shape[1]
            for side, region in (("left", drive[:, : width // 2]), ("right", drive[:, width // 2 :])):
                present = bool(float(region.mean()) > 0.01)
                source = f"complete_drivable_{side}_{'present' if present else 'absent'}"
                attributes[source] = True
                side_map = torch.zeros_like(drive)
                if side == "left": side_map[:, : width // 2] = region
                else: side_map[:, width // 2 :] = region
                attribute_maps[source] = side_map.reshape(-1)
        if paths.label_json and not any(attributes.get(name, False) for name in ("front_vehicle", "front_person", "front_rider", "front_obstacle")):
            attributes["complete_front_corridor_clear"] = True
        return {"explicit_attributes": attributes, "attribute_maps": attribute_maps, "complete_sources": complete}
