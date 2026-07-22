from __future__ import annotations

from typing import Any

import torch

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
            return [item for frame in raw["frames"] if isinstance(frame, dict) for item in frame.get("labels", []) if isinstance(item, dict)]
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
            "part_coordinates": torch.zeros(parts, 2), "part_scales": torch.zeros(parts, 2),
        }

    def from_metadata(self, record: TaskAwareGroundingRecord, metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
        detections = [label for source in metadata.get("detection", []) for label in _flatten_labels(source)]
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
            present = bool(labels)
            geometry = any(bool(item.get("box2d") or item.get("box")) for item in labels)
            state = None
            state_valid = 0.0
            if name == "traffic_light" and labels:
                color = str((labels[0].get("attributes") or {}).get("trafficLightColor") or "").lower()
                mapping = {"red": 0, "green": 1, "yellow": 2}
                if color in mapping:
                    state = torch.zeros(4)
                    state[mapping[color]] = 1.0
                    state_valid = 1.0
            result[name] = self._base(field, record, present, geometry, state, state_valid)
        return result
