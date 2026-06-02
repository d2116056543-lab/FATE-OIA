from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from fate_oia.datasets.bdd_oia_multitask import BDDOIAMultiTaskDataset
from fate_oia.datasets.bdd100k_structured import BDD100KStructuredIndex


class CAREMoEOIADataset(BDDOIAMultiTaskDataset):
    def __init__(
        self,
        data_root: str | Path,
        raw_root: str | Path,
        bdd100k_root: str | Path,
        split: str = "train",
        action_dim: int = 4,
        reason_dim: int = 21,
        load_image: bool = True,
        transform: Any | None = None,
        include_structured: bool = True,
    ) -> None:
        super().__init__(
            data_root=data_root,
            raw_root=raw_root,
            split=split,
            action_dim=action_dim,
            reason_dim=reason_dim,
            load_image=load_image,
            transform=transform,
        )
        self.include_structured = include_structured
        self.structured_index = BDD100KStructuredIndex(bdd100k_root) if include_structured else None

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = super().__getitem__(idx)
        rec = self.structured_index.resolve(item["file_name"], self.split) if self.structured_index is not None else None
        item["bdd100k_structured"] = rec.to_dict() if rec is not None else None
        item["has_structured"] = rec is not None
        counts = rec.counts if rec is not None else {}
        item["object_count"] = int(counts.get("object_count", 0))
        item["lane_count"] = int(counts.get("lane_count", 0))
        item["drivable_count"] = int(counts.get("drivable_count", 0))
        item["attribute_count"] = int(counts.get("attribute_count", 0))
        return item


def care_moe_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    tensor_keys = ["image", "action", "reason"]
    for key in tensor_keys:
        if key in batch[0]:
            out[key] = torch.stack([x[key] for x in batch])
    for key in ["split", "file_name", "image_path", "image_meta", "bdd100k_structured"]:
        out[key] = [x.get(key) for x in batch]
    for key in ["has_structured", "object_count", "lane_count", "drivable_count", "attribute_count"]:
        out[key] = torch.tensor([int(x.get(key, 0)) for x in batch], dtype=torch.long)
    return out
