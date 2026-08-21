from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Sequence

from PIL import Image, ImageOps
import torch

from .transforms import IMAGENET_MEAN, IMAGENET_STD, _to_tensor


def _letterbox(image: Image.Image, hw: tuple[int, int]) -> tuple[Image.Image, dict[str, Any]]:
    height, width = hw
    scale = min(width / image.width, height / image.height)
    resized = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    image = image.resize(resized, Image.BILINEAR)
    left = (width - resized[0]) // 2
    top = (height - resized[1]) // 2
    boxed = ImageOps.expand(image, (left, top, width - resized[0] - left, height - resized[1] - top), fill=(0, 0, 0))
    return boxed, {"resized": resized, "padding": (left, top, width - resized[0] - left, height - resized[1] - top)}


def _normalize(image: Image.Image) -> torch.Tensor:
    tensor = _to_tensor(image)
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


@dataclass
class SynchronizedVideoTransform:
    target_hw: tuple[int, int] = (360, 640)
    context_hw: tuple[int, int] = (192, 344)
    flip_probability: float = 0.5

    def __call__(
        self, frames: Sequence[Image.Image], *, training: bool, random_value: float | None = None
    ) -> dict[str, Any]:
        if len(frames) != 15:
            raise ValueError(f"expected 15 frames, got {len(frames)}")
        value = random.random() if random_value is None else float(random_value)
        flipped = bool(training and value < self.flip_probability)
        normalized_geometry = (frames[-1].width / max(frames[-1].height, 1), int(flipped))
        context: list[torch.Tensor] = []
        meta: list[dict[str, Any]] = []
        for index, source in enumerate(frames):
            image = source.convert("RGB")
            if flipped:
                image = ImageOps.mirror(image)
            hw = self.target_hw if index == 14 else self.context_hw
            boxed, geometry = _letterbox(image, hw)
            geometry["normalized_geometry"] = normalized_geometry
            meta.append(geometry)
            if index < 14:
                context.append(_normalize(boxed))
            else:
                target = _normalize(boxed)
        return {
            "target_image": target,
            "context_images": torch.stack(context),
            "meta": {"flipped": flipped, "frames": meta},
        }
