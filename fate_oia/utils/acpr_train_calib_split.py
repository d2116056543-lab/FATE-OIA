from __future__ import annotations

import hashlib
from typing import Any

from torch.utils.data import DataLoader, Subset


def _item_key(dataset: Any, idx: int, key: str) -> str:
    base = getattr(dataset, "dataset", dataset)
    mapped_idx = idx
    if hasattr(dataset, "indices"):
        mapped_idx = int(dataset.indices[idx])
    if hasattr(base, "items"):
        item = base.items[mapped_idx]
        if isinstance(item, dict) and key in item:
            return str(item[key])
    if hasattr(base, "samples"):
        sample = base.samples[mapped_idx]
        if hasattr(sample, key):
            return str(getattr(sample, key))
        if isinstance(sample, dict) and key in sample:
            return str(sample[key])
    try:
        item = dataset[idx]
        if isinstance(item, dict) and key in item:
            return str(item[key])
    except Exception:
        pass
    return str(idx)


def make_train_calib_indices(
    dataset: Any,
    calib_fraction: float = 0.10,
    seed: int = 20260615,
    key: str = "file_name",
) -> tuple[list[int], list[int]]:
    n = len(dataset)
    if n <= 1:
        return list(range(n)), []
    calib_count = max(1, int(round(n * float(calib_fraction))))
    scored = []
    for idx in range(n):
        raw = f"{seed}:{_item_key(dataset, idx, key)}".encode("utf-8")
        score = int(hashlib.sha1(raw).hexdigest(), 16)
        scored.append((score, idx))
    calib = sorted(idx for _, idx in sorted(scored)[:calib_count])
    calib_set = set(calib)
    main = [idx for idx in range(n) if idx not in calib_set]
    return main, calib


def make_subset_loader(dataset: Any, indices: list[int], **loader_kwargs: Any) -> DataLoader:
    return DataLoader(Subset(dataset, indices), **loader_kwargs)
