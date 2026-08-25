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
    # Project rounded/clamped ideal timestamps onto the feasible set of
    # strictly ordered indices. The forward pass resolves duplicated early
    # frames; the backward pass preserves the exact terminal frame.
    for position in range(1, len(indices)):
        indices[position] = max(int(indices[position]), int(indices[position - 1]) + 1)
    indices[-1] = int(target_frame_index)
    for position in range(len(indices) - 2, -1, -1):
        indices[position] = min(int(indices[position]), int(indices[position + 1]) - 1)
    if int(indices[0]) < 0:
        raise ValueError("quadratic sampling cannot form strictly ordered frame indices")
    return indices


def _decode_selected_frames_from_capture(
    capture: Any,
    indices: torch.Tensor,
    *,
    bgr_to_rgb: Callable[[Any], Any],
    seek_frame: Callable[[Any, int], bool] | None = None,
    sequential_gap: int = 8,
) -> tuple[list[Any], torch.Tensor]:
    """Decode sparse frames with seeks across large gaps and grabs across short gaps."""
    from PIL import Image

    requested = [int(value) for value in indices.tolist()]
    decoded: dict[int, Image.Image] = {}
    current_index = -1
    for index in requested:
        gap = index - current_index
        if seek_frame is not None and (current_index < 0 or gap > int(sequential_gap)):
            if not seek_frame(capture, index):
                continue
            current_index = index - 1
            gap = 1
        reached = True
        for _ in range(gap):
            if not capture.isOpened() or not capture.grab():
                reached = False
                break
            current_index += 1
        if not reached:
            break
        ok, frame = capture.retrieve()
        if ok:
            decoded[index] = Image.fromarray(bgr_to_rgb(frame))
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


def decode_selected_frames(path: str | Path, indices: torch.Tensor) -> tuple[list[Any], torch.Tensor]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        return _decode_selected_frames_from_capture(
            capture,
            indices,
            bgr_to_rgb=lambda frame: cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            seek_frame=lambda stream, position: stream.set(cv2.CAP_PROP_POS_FRAMES, position),
        )
    finally:
        capture.release()


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
        max_samples: int | None = None,
        object_track_store_path: str | Path | None = None,
        frame_store_root: str | Path | None = None,
    ) -> None:
        self.records = [record for record in load_manifest(manifest_path) if record.partition == partition]
        if max_samples is not None:
            self.records = self.records[: max(0, int(max_samples))]
        self.partition = partition
        self.training = training
        self.transform = transform or SynchronizedVideoTransform()
        self.decoder = decoder
        self.seed = int(seed)
        # Keep precomputed tracks in two contiguous shared tensors. A Python
        # dict of per-sample tensor views is re-serialized into every spawned
        # Windows DataLoader worker and can exhaust native memory.
        self.object_tracks = None
        self.object_track_indices: dict[str, int] | None = None
        self.object_tracks_xy: torch.Tensor | None = None
        self.object_tracks_visibility: torch.Tensor | None = None
        self.frame_store_root = Path(frame_store_root) if frame_store_root is not None else None
        if self.frame_store_root is not None and not self.frame_store_root.is_dir():
            raise FileNotFoundError(f"raw frame store does not exist: {self.frame_store_root}")
        if object_track_store_path is not None:
            payload = torch.load(object_track_store_path, map_location="cpu", weights_only=True)
            names = payload["file_names"]
            if len(names) != len(payload["tracks_xy"]) or len(names) != len(payload["visibility"]):
                raise ValueError("object track store arrays have inconsistent lengths")
            tracks_xy = payload["tracks_xy"]
            visibility = payload["visibility"]
            if not torch.is_tensor(tracks_xy):
                tracks_xy = torch.stack(list(tracks_xy))
            if not torch.is_tensor(visibility):
                visibility = torch.stack(list(visibility))
            self.object_tracks_xy = tracks_xy.float().contiguous().share_memory_()
            self.object_tracks_visibility = visibility.bool().contiguous().share_memory_()
            self.object_track_indices = {
                str(name).lower(): index for index, name in enumerate(names)
            }

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int | tuple[int, int]) -> dict[str, Any]:
        from PIL import Image

        augmentation_seed = self.seed
        if isinstance(index, tuple):
            index, augmentation_seed = int(index[0]), int(index[1])
        record = self.records[index]
        requested_timestamps = quadratic_multirate_timestamps()
        if self.training and self.object_track_indices is None and self.frame_store_root is None:
            requested_timestamps = jitter_timestamps(requested_timestamps, random.Random(augmentation_seed))
        frame_indices = timestamps_to_indices(requested_timestamps, record.fps, record.target_frame_index)
        timestamps = (frame_indices.to(torch.float32) - float(record.target_frame_index)) / float(record.fps)
        timestamps[-1] = 0.0
        if not torch.all(timestamps[1:] > timestamps[:-1]):
            raise ValueError(f"actual decoded timestamps are not strictly increasing: {record.file_name}")
        # The audited terminal JPEG is the prediction frame. Decoding the same
        # terminal video frame again wastes work and is discarded below.
        if self.frame_store_root is None:
            decoded, decoded_valid = self.decoder(record.clip_path, frame_indices[:-1])
        else:
            case_dir = self.frame_store_root / record.partition / Path(record.file_name).stem
            decoded = [Image.open(case_dir / f"{position:02d}.jpg").convert("RGB") for position in range(14)]
            decoded_valid = torch.ones(14, dtype=torch.bool)
        target = Image.open(record.target_image_path).convert("RGB")
        frames = decoded + [target]
        transformed = self.transform(
            frames, training=self.training, random_value=random.Random(augmentation_seed + 1).random()
        )
        action = record.action
        reason = record.reason
        if transformed["meta"]["flipped"]:
            action = remap_horizontal_flip_labels(action, ACTION_FLIP_PERMUTATION)
            reason = remap_horizontal_flip_labels(reason, REASON_FLIP_PERMUTATION)
        frame_valid = torch.cat((decoded_valid, torch.ones(1, dtype=torch.bool)))
        result = {
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
        if self.object_track_indices is not None:
            key = record.file_name.lower()
            if key not in self.object_track_indices:
                raise KeyError(f"object track store missing {record.file_name}")
            track_index = self.object_track_indices[key]
            tracks_xy = self.object_tracks_xy[track_index].clone()
            visibility = self.object_tracks_visibility[track_index]
            if transformed["meta"]["flipped"]:
                tracks_xy[..., 0] = -tracks_xy[..., 0]
            result["object_tracks_xy"] = tracks_xy
            result["object_tracks_visibility"] = visibility
        return result


def tida_video_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    tensor_keys = ["target_image", "context_images", "timestamps", "requested_timestamps", "frame_indices", "frame_valid_mask", "action", "reason"]
    if "object_tracks_xy" in batch[0]:
        tensor_keys.extend(("object_tracks_xy", "object_tracks_visibility"))
    result = {key: torch.stack([row[key] for row in batch]) for key in tensor_keys}
    for key in ("file_name", "target_image_path", "clip_path", "source_video_id", "clip_meta", "transform_meta"):
        result[key] = [row[key] for row in batch]
    return result
