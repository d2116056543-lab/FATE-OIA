from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from .bdd_oia_multitask import BDDOIAMultiTaskDataset
from .meter_grounding_index import METERGroundingIndex


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
        if self.include_grounding and split != "train":
            raise ValueError("Typed grounding is train-only")

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
        return item
