from __future__ import annotations

from pathlib import Path
import random
from typing import Any, Callable

import torch
from torch.utils.data import Dataset

from .tida_clip_manifest import TIDAClipRecord, load_manifest
from ..transforms_video import SynchronizedVideoTransform


ACTION_FLIP_PERMUTATION = (0, 1, 3, 2)
REASON_FLIP_PERMUTATION = (0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 18, 19, 20, 9, 10, 11, 12, 13, 14)


def remap_horizontal_flip_labels(values: tuple[float, ...], permutation: tuple[int, ...]) -> tuple[float, ...]:
    if len(values) != len(permutation):
        raise ValueError("label vector and horizontal-flip permutation differ")
    return tuple(float(values[index]) for index in permutation)


def quadratic_multirate_timestamps(num_frames: int = 15, history_seconds: float = 5.0) -> torch.Tensor:
    if num_frames < 2:
        raise ValueError("num_frames must be at least 2")
    index = torch.arange(num_frames, dtype=torch.float64)
    return (-history_seconds * (1.0 - index / (num_frames - 1)) ** 2).to(torch.float32)


def jitter_timestamps(timestamps: torch.Tensor, rng: random.Random, fraction: float = 0.10) -> torch.Tensor:
    result = timestamps.clone()
    for index in range(1, len(result) - 1):
        left = float(result[index] - result[index - 1])
        right = float(result[index + 1] - result[index])
        radius = fraction * min(left, right)
        result[index] += rng.uniform(-radius, radius)
    result[-1] = 0.0
    if not torch.all(result[1:] > result[:-1]):
        raise RuntimeError("timestamp jitter changed sampling order")
    return result


def timestamps_to_indices(timestamps: torch.Tensor, fps: float, target_frame_index: int) -> torch.Tensor:
    indices = torch.round(target_frame_index + timestamps * float(fps)).to(torch.long)
    indices.clamp_(0, int(target_frame_index))
    if int(target_frame_index) < len(indices) - 1:
        raise ValueError("clip has too few frames for unique temporal sampling")
    indices[-1] = int(target_frame_index)
    for position in range(len(indices) - 2, -1, -1):
        indices[position] = min(int(indices[position]), int(indices[position + 1]) - 1)
    if int(indices[0]) < 0:
        raise ValueError("quadratic sampling cannot form strictly ordered frame indices")
    return indices


def decode_selected_frames(path: str | Path, indices: torch.Tensor) -> tuple[list[Any], torch.Tensor]:
    import cv2
    from PIL import Image

    requested = [int(value) for value in indices.tolist()]
    wanted = set(requested)
    decoded: dict[int, Image.Image] = {}
    capture = cv2.VideoCapture(str(path))
    frame_index = 0
    while capture.isOpened() and wanted:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index in wanted:
            decoded[frame_index] = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            wanted.remove(frame_index)
        frame_index += 1
    capture.release()
    frames: list[Image.Image] = []
    valid = torch.zeros(len(requested), dtype=torch.bool)
    previous: Image.Image | None = None
    for position, index in enumerate(requested):
        image = decoded.get(index)
        if image is not None:
            previous = image
            valid[position] = True
        if previous is None:
            previous = Image.new("RGB", (1280, 720))
        frames.append(previous.copy())
    return frames, valid


class BDDOIAVideoDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        partition: str,
        *,
        training: bool = False,
        transform: SynchronizedVideoTransform | None = None,
        decoder: Callable[[str | Path, torch.Tensor], tuple[list[Any], torch.Tensor]] = decode_selected_frames,
        seed: int = 20260821,
    ) -> None:
        self.records = [record for record in load_manifest(manifest_path) if record.partition == partition]
        self.partition = partition
        self.training = training
        self.transform = transform or SynchronizedVideoTransform()
        self.decoder = decoder
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        from PIL import Image

        record = self.records[index]
        requested_timestamps = quadratic_multirate_timestamps()
        if self.training:
            requested_timestamps = jitter_timestamps(requested_timestamps, random.Random(self.seed + index))
        frame_indices = timestamps_to_indices(requested_timestamps, record.fps, record.target_frame_index)
        timestamps = (frame_indices.to(torch.float32) - float(record.target_frame_index)) / float(record.fps)
        timestamps[-1] = 0.0
        if not torch.all(timestamps[1:] > timestamps[:-1]):
            raise ValueError(f"actual decoded timestamps are not strictly increasing: {record.file_name}")
        decoded, decoded_valid = self.decoder(record.clip_path, frame_indices)
        target = Image.open(record.target_image_path).convert("RGB")
        frames = decoded[:14] + [target]
        transformed = self.transform(frames, training=self.training)
        action = record.action
        reason = record.reason
        if transformed["meta"]["flipped"]:
            action = remap_horizontal_flip_labels(action, ACTION_FLIP_PERMUTATION)
            reason = remap_horizontal_flip_labels(reason, REASON_FLIP_PERMUTATION)
        frame_valid = decoded_valid.clone()
        frame_valid[-1] = True
        return {
            "target_image": transformed["target_image"],
            "context_images": transformed["context_images"],
            "timestamps": timestamps,
            "requested_timestamps": requested_timestamps,
            "frame_indices": frame_indices,
            "frame_valid_mask": frame_valid,
            "action": torch.tensor(action, dtype=torch.float32),
            "reason": torch.tensor(reason, dtype=torch.float32),
            "file_name": record.file_name,
            "target_image_path": str(record.target_image_path),
            "clip_path": str(record.clip_path),
            "source_video_id": record.source_video_id,
            "clip_meta": record.to_dict(),
            "transform_meta": transformed["meta"],
        }


def tida_video_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    tensor_keys = ("target_image", "context_images", "timestamps", "requested_timestamps", "frame_indices", "frame_valid_mask", "action", "reason")
    result = {key: torch.stack([row[key] for row in batch]) for key in tensor_keys}
    for key in ("file_name", "target_image_path", "clip_path", "source_video_id", "clip_meta", "transform_meta"):
        result[key] = [row[key] for row in batch]
    return result
