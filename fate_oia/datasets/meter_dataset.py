from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .bdd_oia_multitask import BDDOIAMultiTaskDataset
from .meter_grounding_index import METERGroundingIndex

ACTION_MIRROR_PAIRS = ((2, 3),)
REASON_MIRROR_PAIRS = ((9, 15), (10, 16), (11, 17), (12, 18), (13, 19), (14, 20))


def _swap_pairs(value: torch.Tensor, pairs: tuple[tuple[int, int], ...]) -> torch.Tensor:
    result = value.clone()
    for left, right in pairs:
        result[left], result[right] = value[right].clone(), value[left].clone()
    return result


def mirror_typed_target(target: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result = {key: value.clone() for key, value in target.items()}
    for key in (
        "factor_anchor_map",
        "factor_anchor_valid",
        "factor_state_target",
        "factor_state_valid",
        "factor_observability",
        "factor_observability_valid",
        "factor_source_weight",
    ):
        if key not in result:
            continue
        result[key] = _swap_pairs(result[key], REASON_MIRROR_PAIRS)
    if "factor_anchor_map" in result:
        result["factor_anchor_map"] = torch.flip(
            result["factor_anchor_map"], dims=[-1]
        )
    return result


def fixed_meter_split_indices(
    file_names: list[str],
    *,
    audit_fraction: float,
    calib_fraction: float,
    seed: int,
) -> dict[str, list[int]]:
    if audit_fraction <= 0 or calib_fraction <= 0 or audit_fraction + calib_fraction >= 1:
        raise ValueError("METER requires positive disjoint audit/calibration fractions")
    ranked = sorted(
        range(len(file_names)),
        key=lambda index: hashlib.sha256(
            f"{seed}:{file_names[index]}".encode("utf-8")
        ).hexdigest(),
    )
    audit_count = max(1, int(round(len(ranked) * audit_fraction)))
    calib_count = max(1, int(round(len(ranked) * calib_fraction)))
    return {
        "audit": ranked[:audit_count],
        "calib": ranked[audit_count : audit_count + calib_count],
        "main": ranked[audit_count + calib_count :],
    }


def meter_split_manifest(
    file_names: list[str], split: dict[str, list[int]]
) -> dict[str, Any]:
    owners: dict[int, str] = {}
    manifest: dict[str, Any] = {}
    for name, indices in split.items():
        if any(index in owners for index in indices):
            raise ValueError("METER split partitions overlap")
        owners.update({index: name for index in indices})
        names = [str(file_names[index]) for index in indices]
        manifest[name] = {
            "count": len(indices),
            "sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(),
            "first": names[:3],
        }
    if len(owners) != len(file_names):
        raise ValueError("METER split partitions do not cover all samples")
    manifest["disjoint"] = True
    return manifest


class METERDataset(Dataset):
    """BDD-OIA RGB samples with train-only typed weak grounding."""

    def __init__(
        self,
        *,
        data_root: str | Path,
        raw_root: str | Path,
        split: str,
        transform: Any,
        grounding_index: METERGroundingIndex | None = None,
        include_grounding: bool = False,
        mirror_probability: float = 0.0,
    ) -> None:
        self.base = BDDOIAMultiTaskDataset(
            data_root=data_root,
            raw_root=raw_root,
            split=split,
            action_dim=4,
            reason_dim=21,
            load_image=True,
            transform=transform,
        )
        self.split = split
        self.grounding_index = grounding_index
        self.include_grounding = bool(include_grounding)
        self.mirror_probability = float(mirror_probability)
        if self.include_grounding and split != "train":
            raise ValueError("Typed grounding is train-only")
        if not 0.0 <= self.mirror_probability <= 1.0:
            raise ValueError("mirror_probability must be in [0,1]")

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = dict(self.base[index])
        if self.include_grounding and self.grounding_index is not None:
            target = self.grounding_index.typed_target(
                item["file_name"], split=self.split
            )
            if target is not None:
                item["meter_grounding"] = target
        if self.split == "train" and random.random() < self.mirror_probability:
            item["image"] = torch.flip(item["image"], dims=[-1])
            item["action"] = _swap_pairs(item["action"], ACTION_MIRROR_PAIRS)
            item["reason"] = _swap_pairs(item["reason"], REASON_MIRROR_PAIRS)
            if "meter_grounding" in item:
                item["meter_grounding"] = mirror_typed_target(
                    item["meter_grounding"]
                )
            item["meter_mirrored"] = True
        else:
            item["meter_mirrored"] = False
        return item
