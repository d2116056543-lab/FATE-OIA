from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def audit_image_size(image: Image.Image) -> dict[str, tuple[int, int]]:
    """Return the real PIL image size as width/height."""
    return {"original_size": tuple(image.size)}


def _to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.uint8)
    tensor = torch.from_numpy(array.copy()).permute(2, 0, 1).float() / 255.0
    return tensor


@dataclass
class AspectRatioLetterboxTransform:
    image_height: int
    image_width: int
    patch_size: int = 8
    fill: tuple[int, int, int] = (0, 0, 0)
    normalize: bool = True
    return_meta: bool = False

    def _resize_with_meta(self, image: Image.Image) -> tuple[Image.Image, dict[str, Any]]:
        original_w, original_h = image.size
        scale = min(self.image_width / max(original_w, 1), self.image_height / max(original_h, 1))
        resized_w = max(1, int(round(original_w * scale)))
        resized_h = max(1, int(round(original_h * scale)))
        resized = image.resize((resized_w, resized_h), Image.BILINEAR)
        pad_left = (self.image_width - resized_w) // 2
        pad_top = (self.image_height - resized_h) // 2
        pad_right = self.image_width - resized_w - pad_left
        pad_bottom = self.image_height - resized_h - pad_top
        boxed = ImageOps.expand(resized, border=(pad_left, pad_top, pad_right, pad_bottom), fill=self.fill)
        meta = {
            "original_size": (original_w, original_h),
            "resized_size": (resized_w, resized_h),
            "padding": (pad_left, pad_top, pad_right, pad_bottom),
            "patch_grid": (self.image_height // self.patch_size, self.image_width // self.patch_size),
            "image_size": (self.image_width, self.image_height),
        }
        return boxed, meta

    def __call__(self, image: Image.Image):
        if image.mode != "RGB":
            image = image.convert("RGB")
        boxed, meta = self._resize_with_meta(image)
        tensor = _to_tensor(boxed)
        if self.normalize:
            tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
        if self.return_meta:
            return tensor, meta
        return tensor


@dataclass
class FixedSizeResizeTransform:
    image_height: int
    image_width: int
    patch_size: int = 8
    normalize: bool = True
    return_meta: bool = False

    def __call__(self, image: Image.Image):
        if image.mode != "RGB":
            image = image.convert("RGB")
        original = tuple(image.size)
        resized = image.resize((self.image_width, self.image_height), Image.BILINEAR)
        tensor = _to_tensor(resized)
        if self.normalize:
            tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
        meta = {
            "original_size": original,
            "resized_size": (self.image_width, self.image_height),
            "padding": (0, 0, 0, 0),
            "patch_grid": (self.image_height // self.patch_size, self.image_width // self.patch_size),
            "image_size": (self.image_width, self.image_height),
        }
        if self.return_meta:
            return tensor, meta
        return tensor
