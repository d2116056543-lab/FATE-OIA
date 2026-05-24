from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import Dataset


ACTION_NAMES_4 = ["forward", "stop", "left", "right"]
ACTION_NAMES_5 = ["forward", "stop", "left", "right", "confuse"]


@dataclass(frozen=True)
class BDDOIASample:
    split: str
    file_name: str
    image_path: str
    action: tuple[float, ...]
    reason: tuple[float, ...]


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return json.load(f)


def _find_file(root: Path, candidates: list[str]) -> Path:
    for name in candidates:
        p = root / name
        if p.exists():
            return p
    raise FileNotFoundError(f"None of these files exist under {root}: {candidates}")


def _as_vec(value: Any, dim: int, field: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be a list/tuple, got {type(value)}")
    if len(value) < dim:
        # BDD-OIA action vectors may appear as 4-dim or 5-dim depending on
        # whether the optional confuse label is present. Pad absent labels
        # with 0 so action_dim=5 remains usable.
        value = list(value) + [0.0] * (dim - len(value))
    return tuple(float(x) for x in value[:dim])


def _load_action_rows(raw_root: Path, split: str, action_dim: int) -> list[dict[str, Any]]:
    path = _find_file(raw_root, [f"{split}_25k_images_actions.json", f"{split}_images_actions.json", f"{split}.json"])
    data = _load_json(path)
    images = data.get("images", []) if isinstance(data, dict) else []
    anns = data.get("annotations", []) if isinstance(data, dict) else []
    if len(images) != len(anns):
        raise ValueError(f"Action JSON image/annotation count mismatch in {path}: {len(images)} vs {len(anns)}")
    rows = []
    for img, ann in zip(images, anns):
        file_name = img.get("file_name") or img.get("filename")
        if not file_name:
            raise ValueError(f"Missing file_name in {path}")
        rows.append({"file_name": str(file_name), "action": _as_vec(ann.get("category"), action_dim, "category")})
    return rows


def _load_reason_map(raw_root: Path, split: str, reason_dim: int) -> dict[str, tuple[float, ...]]:
    path = _find_file(raw_root, [f"{split}_25k_images_reasons.json", f"{split}_images_reasons.json"])
    data = _load_json(path)
    if isinstance(data, dict):
        records = data.get("annotations") or data.get("reasons") or data.get("images") or []
    else:
        records = data
    out: dict[str, tuple[float, ...]] = {}
    for rec in records:
        fn = rec.get("file_name") or rec.get("filename") or rec.get("image")
        reason = rec.get("reason") or rec.get("category")
        if fn is None or reason is None:
            continue
        out[str(fn)] = _as_vec(reason, reason_dim, "reason")
    return out


class BDDOIAMultiTaskDataset(Dataset):
    """BDD-OIA action/reason multi-label dataset.

    This loader intentionally treats BDD-OIA as multi-label:
    action category vectors use sigmoid/BCE paths downstream, not softmax.
    """

    def __init__(
        self,
        data_root: str | Path = "dataset/BDD-OIA",
        raw_root: str | Path | None = None,
        split: str = "train",
        action_dim: int = 4,
        reason_dim: int = 21,
        load_image: bool = False,
        transform: Callable[[Any], Any] | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.raw_root = Path(raw_root) if raw_root is not None else self.data_root
        self.split = split
        self.action_dim = action_dim
        self.reason_dim = reason_dim
        self.load_image = load_image
        self.transform = transform
        if action_dim not in (4, 5):
            raise ValueError("action_dim must be 4 or 5")
        action_rows = _load_action_rows(self.raw_root, split, action_dim)
        reason_map = _load_reason_map(self.raw_root, split, reason_dim)
        samples: list[BDDOIASample] = []
        missing_reason: list[str] = []
        missing_image: list[str] = []
        for row in action_rows:
            fn = row["file_name"]
            if fn not in reason_map:
                missing_reason.append(fn)
                reason = tuple(0.0 for _ in range(reason_dim))
            else:
                reason = reason_map[fn]
            img_path = self._resolve_image_path(fn)
            if not Path(img_path).exists():
                missing_image.append(fn)
            samples.append(BDDOIASample(split, fn, img_path, row["action"], reason))
        self.samples = samples
        self.missing_reason = missing_reason
        self.missing_image = missing_image

    def _resolve_image_path(self, file_name: str) -> str:
        candidates = [
            self.data_root / self.split / file_name,
            self.data_root / "data" / file_name,
            self.raw_root / self.split / file_name,
            self.raw_root / "data" / file_name,
        ]
        for p in candidates:
            if p.exists():
                return str(p)
        return str(candidates[0])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        s = self.samples[idx]
        item: dict[str, Any] = {
            "split": s.split,
            "file_name": s.file_name,
            "image_path": s.image_path,
            "action": torch.tensor(s.action, dtype=torch.float32),
            "reason": torch.tensor(s.reason, dtype=torch.float32),
        }
        if self.load_image:
            from PIL import Image
            img = Image.open(s.image_path).convert("RGB")
            item["image"] = self.transform(img) if self.transform else img
        return item

    def audit(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "count": len(self),
            "action_dim": self.action_dim,
            "reason_dim": self.reason_dim,
            "missing_reason": len(self.missing_reason),
            "missing_image": len(self.missing_image),
            "action_positive_counts": torch.stack([torch.tensor(s.action) for s in self.samples]).sum(0).tolist()
            if self.samples
            else [],
            "reason_positive_counts": torch.stack([torch.tensor(s.reason) for s in self.samples]).sum(0).tolist()
            if self.samples
            else [],
        }
