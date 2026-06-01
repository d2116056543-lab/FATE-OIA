from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from fate_oia.datasets.bdd100k_structured import BDD100KStructuredIndex
from fate_oia.datasets.bdd_oia_multitask import BDDOIAMultiTaskDataset
from fate_oia.engine.train_fate_oia import build_transform


class SUREOIADataset(Dataset):
    """Direct-image BDD-OIA dataset with optional BDD100K structured metadata."""

    def __init__(
        self,
        data_root: str | Path,
        raw_root: str | Path,
        bdd100k_root: str | Path,
        split: str,
        action_dim: int = 4,
        reason_dim: int = 21,
        image_height: int = 360,
        image_width: int = 640,
        patch_size: int = 8,
        preserve_aspect_ratio: bool = True,
    ) -> None:
        self.base = BDDOIAMultiTaskDataset(
            data_root=data_root,
            raw_root=raw_root,
            split=split,
            action_dim=action_dim,
            reason_dim=reason_dim,
            load_image=True,
            transform=build_transform(image_height, image_width, patch_size, preserve_aspect_ratio, return_meta=True),
        )
        self.structured_index = BDD100KStructuredIndex(bdd100k_root)
        self.split = split

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.base[index]
        item["bdd100k_structured"] = self.structured_index.lookup(item["file_name"], self.split).to_dict()
        return item

    def audit(self) -> dict[str, Any]:
        base_audit = self.base.audit()
        sample_names = [s.file_name for s in self.base.samples[: min(len(self.base.samples), 512)]]
        base_audit["bdd100k_structured_sample_audit"] = self.structured_index.audit_samples(sample_names, self.split)
        return base_audit


def sure_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    tensor_keys = {"image", "action", "reason"}
    for key in tensor_keys:
        if key in batch[0]:
            out[key] = torch.stack([item[key] for item in batch], dim=0)
    for key in ["file_name", "image_path", "split", "image_meta", "bdd100k_structured"]:
        if key in batch[0]:
            out[key] = [item[key] for item in batch]
    return out
