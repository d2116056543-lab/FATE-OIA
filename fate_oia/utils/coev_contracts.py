from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


ACTION_FLIP = (0, 1, 3, 2)
REASON_FLIP = (0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 18, 19, 20, 9, 10, 11, 12, 13, 14)
PREDICATE_NAMES = (
    "vehicle", "pedestrian", "bicycle", "traffic_light_red",
    "traffic_light_green", "traffic_sign", "drivable", "lane",
)
PRIMITIVE_NAMES = (
    "vehicle_front", "vehicle_left", "vehicle_right",
    "pedestrian_front", "pedestrian_left", "pedestrian_right",
    "vehicle_front_expansion", "vehicle_left_to_center", "vehicle_right_to_center",
    "traffic_red_upper", "traffic_green_upper", "camera_lateral_rate",
    "camera_vertical_rate", "camera_log_scale_rate",
)


@dataclass
class CoEVInputs:
    target_rgb: Tensor
    history_rgb: Tensor
    actual_t: Tensor
    valid: Tensor
    geometry_meta: dict[str, Any]

    def validate(self) -> "CoEVInputs":
        b = self.target_rgb.shape[0]
        if self.target_rgb.shape != (b, 3, 360, 640):
            raise ValueError(f"target_rgb must be [B,3,360,640], got {tuple(self.target_rgb.shape)}")
        if self.history_rgb.shape != (b, 14, 3, 256, 448):
            raise ValueError(f"history_rgb must be [B,14,3,256,448], got {tuple(self.history_rgb.shape)}")
        if self.actual_t.shape != (b, 15) or self.actual_t.dtype != torch.float32:
            raise ValueError("actual_t must be FP32 [B,15]")
        if self.valid.shape != (b, 15) or self.valid.dtype != torch.bool:
            raise ValueError("valid must be bool [B,15]")
        if not torch.allclose(self.actual_t[:, -1], torch.zeros_like(self.actual_t[:, -1])):
            raise ValueError("target time must be exactly zero")
        return self

    def to(self, device: torch.device | str) -> "CoEVInputs":
        return CoEVInputs(
            self.target_rgb.to(device), self.history_rgb.to(device),
            self.actual_t.to(device), self.valid.to(device), self.geometry_meta,
        )


@dataclass
class CoEVTargets:
    action: Tensor
    reason: Tensor
    grounding_value: Tensor
    grounding_known: Tensor
    grounding_weight: Tensor
    ids: list[str]

    def to(self, device: torch.device | str) -> "CoEVTargets":
        return CoEVTargets(
            self.action.to(device), self.reason.to(device), self.grounding_value.to(device),
            self.grounding_known.to(device), self.grounding_weight.to(device), self.ids,
        )


def flip_labels(action: Tensor, reason: Tensor) -> tuple[Tensor, Tensor]:
    return action[..., list(ACTION_FLIP)], reason[..., list(REASON_FLIP)]


def formal_total_updates(train_count: int, effective_batch: int = 32, epochs: int = 24) -> tuple[int, int]:
    per_epoch = (int(train_count) + effective_batch - 1) // effective_batch
    return per_epoch, per_epoch * epochs
