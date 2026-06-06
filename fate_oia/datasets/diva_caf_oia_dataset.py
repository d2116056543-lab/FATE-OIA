from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image

from fate_oia.datasets.bdd_oia_multitask import BDDOIAMultiTaskDataset


class SimpleResizeToTensor:
    def __init__(self, height: int = 360, width: int = 640) -> None:
        self.height = int(height)
        self.width = int(width)

    def __call__(self, image: Image.Image):
        original_size = tuple(image.size)
        resized = image.resize((self.width, self.height), Image.BILINEAR)
        data = torch.ByteTensor(torch.ByteStorage.from_buffer(resized.tobytes()))
        tensor = data.view(self.height, self.width, 3).permute(2, 0, 1).float().div(255.0)
        meta = {
            "original_size": original_size,
            "resized_size": (self.width, self.height),
            "pad": (0, 0, 0, 0),
            "patch_grid": (self.height // 8, self.width // 8),
        }
        return tensor, meta


def build_diva_caf_dataset(data_root: str | Path, raw_root: str | Path, split: str, height: int = 360, width: int = 640, max_samples: int | None = None):
    ds = BDDOIAMultiTaskDataset(data_root=data_root, raw_root=raw_root, split=split, action_dim=4, reason_dim=21, load_image=True, transform=SimpleResizeToTensor(height, width))
    if max_samples is not None and max_samples > 0:
        from torch.utils.data import Subset
        return Subset(ds, list(range(min(int(max_samples), len(ds)))))
    return ds


def collate_diva_caf(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image": torch.stack([x["image"] for x in batch]),
        "action": torch.stack([x["action"] for x in batch]),
        "reason": torch.stack([x["reason"] for x in batch]),
        "file_name": [x["file_name"] for x in batch],
        "image_meta": [x.get("image_meta", {}) for x in batch],
    }
