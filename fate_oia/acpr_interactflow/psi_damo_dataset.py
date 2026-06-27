from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

from .psi_frame_resolver import PSIFrameResolver, assert_target_not_in_inputs
from .types import PSIInteractFlowBatch


def _load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".pkl", ".pickle"}:
        with path.open("rb") as f:
            obj = pickle.load(f)
    else:
        text = path.read_text(encoding="utf-8")
        obj = [json.loads(line) for line in text.splitlines() if line.strip()] if path.suffix.lower() == ".jsonl" else json.loads(text)
    if isinstance(obj, dict):
        for key in ("samples", "records", "data"):
            if key in obj:
                obj = obj[key]
                break
    if not isinstance(obj, list):
        raise TypeError(f"Expected list records in {path}, got {type(obj)!r}")
    return [dict(x) for x in obj]


def _load_exp29_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "labels" in obj:
        labels = list(obj["labels"])
        masks = list(obj.get("masks", [1.0] * len(labels)))
        records = []
        for label, mask in zip(labels, masks):
            if isinstance(mask, (list, tuple)):
                mask_values = mask
            else:
                mask_values = [float(mask)] * 29
            records.append({"exp29": label, "exp29_mask": mask_values})
        return records
    if isinstance(obj, list):
        return [dict(x) for x in obj]
    raise TypeError(f"Expected exp29 dict/list records in {path}, got {type(obj)!r}")


def _first(record: dict[str, Any], names: tuple[str, ...], default: Any = None) -> Any:
    for name in names:
        if name in record:
            return record[name]
    return default


def _letterbox_tensor(path: Path, image_size: tuple[int, int]) -> torch.Tensor:
    width, height = image_size[1], image_size[0]
    image = Image.open(path).convert("RGB")
    scale = min(width / image.width, height / image.height)
    new_size = (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale))))
    image = image.resize(new_size, Image.BILINEAR)
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    canvas.paste(image, ((width - new_size[0]) // 2, (height - new_size[1]) // 2))
    data = torch.ByteTensor(torch.ByteStorage.from_buffer(canvas.tobytes()))
    tensor = data.view(height, width, 3).permute(2, 0, 1).float().div(255.0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor - mean) / std


class PSIDAMO11902Dataset(Dataset):
    SPLIT_COUNTS = {"train": 8873, "val": 612, "test": 2417}

    def __init__(
        self,
        package_root: str | Path,
        split: str,
        frames_root: str | Path | None = None,
        image_size: tuple[int, int] = (320, 576),
        action_dim: int = 3,
        strict_counts: bool = False,
        max_samples: int | None = None,
    ) -> None:
        self.package_root = Path(package_root)
        self.split = split
        self.image_size = image_size
        self.action_dim = action_dim
        split_path = self.package_root / "samples" / f"{split}.pkl"
        if not split_path.exists():
            split_path = self.package_root / f"{split}.pkl"
        self.records = _load_records(split_path)
        if max_samples is not None:
            self.records = self.records[: int(max_samples)]
        if strict_counts and max_samples is None and split in self.SPLIT_COUNTS and len(self.records) != self.SPLIT_COUNTS[split]:
            raise ValueError(f"{split} count mismatch: expected {self.SPLIT_COUNTS[split]}, got {len(self.records)}")
        self.exp_records: list[dict[str, Any]] | None = None
        exp_path = self.package_root / "reason_exp29" / f"{split}.pkl"
        if exp_path.exists():
            self.exp_records = _load_exp29_records(exp_path)
        self.resolver = PSIFrameResolver(frames_root or self.package_root.parent / "PSI_data" / "frames")

    def __len__(self) -> int:
        return len(self.records)

    def _record_exp29(self, idx: int, record: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        source = record
        if self.exp_records is not None and idx < len(self.exp_records):
            source = {**record, **self.exp_records[idx]}
        values = _first(source, ("exp29", "reason_exp29", "reason_labels", "explanation_labels"), None)
        if values is None:
            values = [0.0] * 29
        exp = torch.tensor(values, dtype=torch.float32)
        if exp.numel() != 29:
            raise ValueError(f"Exp29 target must have 29 labels, got {exp.numel()}")
        mask_values = _first(source, ("exp29_mask", "reason_mask", "explanation_mask"), None)
        if mask_values is None:
            mask = torch.ones_like(exp)
            if float(exp.sum()) == 0.0:
                mask.zero_()
        else:
            mask = torch.tensor(mask_values, dtype=torch.float32)
        return exp, mask

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]
        frame_values = _first(record, ("input_frames", "frames", "observed_frames"), None)
        if frame_values is None:
            raise KeyError("sample missing input_frames/frames/observed_frames")
        video_id = str(_first(record, ("video_id", "video", "vid"), ""))
        frame_paths = self.resolver.resolve_sequence(list(frame_values), expected_count=15, video_id=video_id)
        target_frame = _first(record, ("target_frame", "target_frame_path"), None)
        if target_frame is not None:
            if isinstance(target_frame, int) or (isinstance(target_frame, str) and str(target_frame).isdigit()):
                target_path = self.resolver.resolve_frame_id(video_id, target_frame)
            else:
                target_path = self.resolver.resolve(str(target_frame))
            assert_target_not_in_inputs(frame_paths, target_path)
        else:
            target_path = Path("")
        frames = torch.stack([_letterbox_tensor(p, self.image_size) for p in frame_paths], dim=0)
        soft = torch.tensor(_first(record, ("action_soft_target", "action_soft", "action_distribution", "soft_action"), [1.0 / self.action_dim] * self.action_dim), dtype=torch.float32)
        if soft.numel() != self.action_dim:
            raise ValueError(f"Action soft target must have {self.action_dim} classes, got {soft.numel()}")
        majority = int(_first(record, ("action_majority", "action_label", "majority_action"), int(soft.argmax().item())))
        exp, exp_mask = self._record_exp29(idx, record)
        return {
            "input_frames": frames,
            "action_soft": soft,
            "action_majority": torch.tensor(majority, dtype=torch.long),
            "exp29": exp,
            "exp29_mask": exp_mask,
            "paper_effective_weight": torch.tensor(float(_first(record, ("paper_effective_weight", "effective_weight", "weight"), 1.0)), dtype=torch.float32),
            "video_id": video_id,
            "start_frame": torch.tensor(int(_first(record, ("start_frame", "start", "frame_start"), 0)), dtype=torch.long),
            "target_frame_index": torch.tensor(int(_first(record, ("target_frame_index", "target_index", "target_frame_id"), 0)), dtype=torch.long),
            "target_frame_path": str(target_path),
            "frame_paths": [str(p) for p in frame_paths],
            "explanation_text": str(_first(record, ("explanation_text", "explanation", "description"), "")),
            "reasoning_text": str(_first(record, ("reasoning_text", "reason", "reasoning"), "")),
            "sample_id": str(_first(record, ("sample_id", "id"), f"{self.split}_{idx}")),
            "meta": record,
        }


def psi_interactflow_collate(items: list[dict[str, Any]]) -> PSIInteractFlowBatch:
    return PSIInteractFlowBatch(
        input_frames=torch.stack([x["input_frames"] for x in items], 0),
        action_soft=torch.stack([x["action_soft"] for x in items], 0),
        action_majority=torch.stack([x["action_majority"] for x in items], 0),
        exp29=torch.stack([x["exp29"] for x in items], 0),
        exp29_mask=torch.stack([x["exp29_mask"] for x in items], 0),
        paper_effective_weight=torch.stack([x["paper_effective_weight"] for x in items], 0),
        video_id=[x["video_id"] for x in items],
        start_frame=torch.stack([x["start_frame"] for x in items], 0),
        target_frame_index=torch.stack([x["target_frame_index"] for x in items], 0),
        target_frame_path=[x["target_frame_path"] for x in items],
        frame_paths=[x["frame_paths"] for x in items],
        explanation_text=[x["explanation_text"] for x in items],
        reasoning_text=[x["reasoning_text"] for x in items],
        sample_id=[x["sample_id"] for x in items],
        meta=[x["meta"] for x in items],
    )
