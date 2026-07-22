from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image

from fate_oia.datasets.bdd100k_task_aware_index import TaskAwareGroundingRecord


def _scalar(value: float) -> torch.Tensor:
    return torch.tensor(float(value), dtype=torch.float32)


def _flatten_labels(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, dict):
        if raw.get("category") or raw.get("name"):
            return [raw]
        if isinstance(raw.get("labels"), list):
            return [item for item in raw["labels"] if isinstance(item, dict)]
        if isinstance(raw.get("frames"), list):
            return [
                item
                for frame in raw["frames"] if isinstance(frame, dict)
                for item in frame.get("objects", frame.get("labels", [])) if isinstance(item, dict)
            ]
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    return []


class PRECISEGroundingAdapter:
    """Converts task-aware metadata into reliable-positive/negative/unknown targets."""

    def __init__(self, fields: list[dict[str, Any]]) -> None:
        self.fields = fields

    @staticmethod
    def _sector(label: dict[str, Any]) -> str:
        box = label.get("box2d") or label.get("box") or {}
        try:
            center = (float(box["x1"]) + float(box["x2"])) * 0.5
            width = max(float(box.get("image_width", 1280.0)), 1.0)
            ratio = center / width
        except (KeyError, TypeError, ValueError):
            return "center"
        return "left" if ratio < 1.0 / 3.0 else "right" if ratio > 2.0 / 3.0 else "center"

    @staticmethod
    def _is_category(label: dict[str, Any], words: tuple[str, ...]) -> bool:
        name = str(label.get("category") or label.get("name") or "").lower()
        return any(word in name for word in words)

    def _base(self, field: dict[str, Any], record: TaskAwareGroundingRecord, present: bool, geometry: bool, state: torch.Tensor | None = None, state_valid: float = 0.0) -> dict[str, Any]:
        complete = all(record.source_complete.get(source, False) for source in field["supervision_sources"])
        valid = float(complete)
        obs = float(complete)
        parts = int(field["num_parts"])
        state = state if state is not None else torch.zeros(len(field.get("state_schema", field.get("type_schema", []))), dtype=torch.float32)
        return {
            "target": _scalar(float(present)), "valid_mask": _scalar(valid), "reliability": _scalar(valid), "source_id": "+".join(field["supervision_sources"]),
            "geometry_valid": _scalar(float(geometry and complete)), "observability_target": _scalar(obs), "presence": _scalar(float(present)),
            "presence_valid": _scalar(valid), "observability": _scalar(obs), "state": state, "state_valid": _scalar(state_valid),
            "part_coordinates": torch.zeros(parts, 2), "part_scales": torch.zeros(parts, 2), "part_valid": _scalar(0.0),
        }

    @staticmethod
    def _box_parts(label: dict[str, Any], parts: int) -> torch.Tensor | None:
        box = label.get("box2d") or label.get("box") or {}
        try:
            x0, x1 = float(box["x1"]) / 1280.0, float(box["x2"]) / 1280.0
            y0, y1 = float(box["y1"]) / 720.0, float(box["y2"]) / 720.0
        except (KeyError, TypeError, ValueError):
            return None
        points = torch.tensor([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=torch.float32)
        return points[:parts] if parts <= 4 else points.repeat((parts + 3) // 4, 1)[:parts]

    @staticmethod
    def _curve_parts(labels: list[dict[str, Any]], parts: int) -> torch.Tensor | None:
        points = [point for label in labels for poly in (label.get("poly2d") or []) if isinstance(poly, dict) for point in poly.get("vertices", []) if isinstance(point, (list, tuple)) and len(point) >= 2]
        if len(points) < 2:
            return None
        raw = torch.tensor([[float(point[0]) / 1280.0, float(point[1]) / 720.0] for point in points], dtype=torch.float32)
        order = torch.argsort(raw[:, 1])
        raw = raw[order]
        positions = torch.linspace(0, raw.shape[0] - 1, parts).round().long()
        return raw[positions]

    @staticmethod
    def _region_parts(sector: str, parts: int) -> torch.Tensor:
        ranges = {"left": (0.0, 1 / 3), "center": (1 / 3, 2 / 3), "right": (2 / 3, 1.0)}
        x0, x1 = ranges[sector]
        yy, xx = torch.meshgrid(torch.linspace(0.45, 0.95, 2), torch.linspace(x0 + 0.04, x1 - 0.04, 4), indexing="ij")
        return torch.stack([xx, yy], dim=-1).reshape(-1, 2)[:parts]

    @staticmethod
    def _poly_sector(label: dict[str, Any]) -> str:
        polys = label.get("poly2d") or []
        points = [point for poly in polys if isinstance(poly, dict) for point in poly.get("vertices", []) if isinstance(point, (list, tuple)) and len(point) >= 2]
        if not points:
            return "center"
        x_mean = sum(float(point[0]) for point in points) / len(points)
        return "left" if x_mean < 1280.0 / 3.0 else "right" if x_mean > 2.0 * 1280.0 / 3.0 else "center"

    @staticmethod
    def _drivable_fraction(paths: tuple[str, ...], sector: str) -> tuple[float, bool]:
        """Read the official red direct-drivable map once during target build."""
        for raw_path in paths:
            try:
                image = Image.open(Path(raw_path)).convert("RGB")
                width, height = image.size
                thirds = {"left": (0, width // 3), "center": (width // 3, 2 * width // 3), "right": (2 * width // 3, width)}
                x0, x1 = thirds[sector]
                # Lower road region avoids treating the upper sky/background as a negative.
                pixels = list(image.crop((x0, int(height * 0.35), x1, height)).getdata())
                if not pixels:
                    continue
                direct = sum(1 for red, green, blue in pixels if red >= 200 and green <= 80 and blue <= 80)
                return direct / len(pixels), True
            except (OSError, ValueError):
                continue
        return 0.0, False

    def from_metadata(self, record: TaskAwareGroundingRecord, metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
        detections = [label for source in metadata.get("detection", []) for label in _flatten_labels(source)]
        lane_objects = [label for source in metadata.get("lane", metadata.get("detection", [])) for label in _flatten_labels(source) if label.get("poly2d")]
        result: dict[str, dict[str, Any]] = {}
        for field in self.fields:
            name, sector = field["name"], field["sector"]
            labels: list[dict[str, Any]] = []
            if name == "traffic_light":
                labels = [item for item in detections if self._is_category(item, ("traffic light",))]
            elif name == "traffic_sign":
                labels = [item for item in detections if self._is_category(item, ("traffic sign", "sign"))]
            elif name.startswith("actor_"):
                labels = [item for item in detections if self._is_category(item, ("car", "truck", "bus", "vehicle", "pedestrian", "rider", "bike", "motor")) and self._sector(item) == sector]
            if name.startswith("drivable_"):
                fraction, readable = self._drivable_fraction(record.drivable_maps, sector)
                target = self._base(field, record, fraction > 0.0, readable, state_valid=0.0)
                target["presence"] = _scalar(fraction)
                target["target"] = _scalar(fraction)
                target["geometry_valid"] = _scalar(float(readable and record.source_complete.get("drivable", False)))
                target["part_coordinates"] = self._region_parts(sector, int(field["num_parts"]))
                target["part_valid"] = _scalar(float(readable and record.source_complete.get("drivable", False)))
                result[name] = target
                continue
            if name.startswith("boundary_"):
                labels = [item for item in lane_objects if self._poly_sector(item) == sector and "lane" in str(item.get("category", "")).lower()]
            present = bool(labels)
            geometry = any(bool(item.get("box2d") or item.get("box") or item.get("poly2d")) for item in labels)
            state = None
            state_valid = 0.0
            if name == "traffic_light" and labels:
                color = str((labels[0].get("attributes") or {}).get("trafficLightColor") or "").lower()
                mapping = {"red": 0, "green": 1, "yellow": 2}
                if color in mapping:
                    state = torch.zeros(4)
                    state[mapping[color]] = 1.0
                    state_valid = 1.0
            elif name.startswith("actor_") and labels:
                state = torch.zeros(4)
                categories = [str(item.get("category", "")).lower() for item in labels]
                state[0] = float(any(word in category for category in categories for word in ("car", "truck", "bus", "vehicle")))
                state[1] = float(any("pedestrian" in category for category in categories))
                state[2] = float(any(word in category for category in categories for word in ("rider", "bike", "motor")))
                state[3] = float(any(state[index] == 0 for index in range(3)))
                state_valid = 1.0
            elif name.startswith("boundary_") and labels:
                state = torch.zeros(3)
                styles = [str((item.get("attributes") or {}).get("style") or "").lower() for item in labels]
                state[0] = float(any(style == "solid" for style in styles))
                state[1] = float(any(style and style != "solid" for style in styles))
                state_valid = float(any(styles))
            target = self._base(field, record, present, geometry, state, state_valid)
            if labels:
                coords = self._curve_parts(labels, int(field["num_parts"])) if name.startswith("boundary_") else self._box_parts(labels[0], int(field["num_parts"]))
                if coords is not None:
                    target["part_coordinates"] = coords
                    target["part_valid"] = _scalar(float(target["geometry_valid"].item() > 0))
            result[name] = target
        return result

    def stack_batch(self, samples: list[dict[str, dict[str, Any]]], device: torch.device | None = None) -> dict[str, torch.Tensor]:
        """Stack only tensor targets; provenance remains outside the model forward."""
        if not samples:
            raise ValueError("PRECISE grounding batch cannot be empty")
        keys = ("presence", "presence_valid", "observability", "geometry_valid", "reliability", "state_valid", "part_valid")
        result: dict[str, torch.Tensor] = {}
        for key in keys:
            value = torch.stack([torch.stack([sample[field["name"]][key] for field in self.fields]) for sample in samples])
            result[key] = value.to(device) if device is not None else value
        max_state = max(len(field.get("state_schema", field.get("type_schema", []))) for field in self.fields)
        max_parts = max(int(field["num_parts"]) for field in self.fields)
        state = torch.zeros(len(samples), len(self.fields), max_state)
        parts = torch.zeros(len(samples), len(self.fields), max_parts, 2)
        for row, sample in enumerate(samples):
            for column, field in enumerate(self.fields):
                item = sample[field["name"]]
                state[row, column, : item["state"].numel()] = item["state"]
                parts[row, column, : item["part_coordinates"].shape[0]] = item["part_coordinates"]
        result["state"] = state.to(device) if device is not None else state
        result["part_coordinates"] = parts.to(device) if device is not None else parts
        return result

    def coverage(self, samples: list[dict[str, dict[str, Any]]]) -> dict[str, dict[str, int]]:
        report: dict[str, dict[str, int]] = {}
        for field in self.fields:
            name = field["name"]
            values = [sample[name] for sample in samples]
            positive = sum(int(item["presence"].item() > 0 and item["presence_valid"].item() > 0) for item in values)
            negative = sum(int(item["presence"].item() == 0 and item["presence_valid"].item() > 0) for item in values)
            geometry = sum(int(item["geometry_valid"].item() > 0) for item in values)
            unknown = sum(int(item["presence_valid"].item() == 0) for item in values)
            report[name] = {"positive_count": positive, "reliable_negative_count": negative, "geometry_valid_count": geometry, "unknown_count": unknown}
        return report
