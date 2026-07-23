from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import ImageEnhance, ImageOps

from fate_oia.transforms import AspectRatioLetterboxTransform


@dataclass
class PRECISEImageTransform:
    image_height: int = 360
    image_width: int = 640
    return_mirror: bool = False
    training: bool = False
    brightness: float = 0.0
    contrast: float = 0.0

    def __post_init__(self) -> None:
        self.base = AspectRatioLetterboxTransform(self.image_height, self.image_width, patch_size=8, return_meta=True)

    def __call__(self, image):
        if self.training:
            if self.brightness > 0:
                factor = 1.0 + (2.0 * torch.rand(()).item() - 1.0) * self.brightness
                image = ImageEnhance.Brightness(image).enhance(factor)
            if self.contrast > 0:
                factor = 1.0 + (2.0 * torch.rand(()).item() - 1.0) * self.contrast
                image = ImageEnhance.Contrast(image).enhance(factor)
        canonical, meta = self.base(image)
        if not self.return_mirror:
            return canonical, meta
        mirrored, mirror_meta = self.base(ImageOps.mirror(image))
        mirror_meta["horizontal_mirror"] = True
        return {"image": canonical, "mirror_image": mirrored, "image_meta": meta, "mirror_meta": mirror_meta}


def mirror_patch_coordinates(points: torch.Tensor) -> torch.Tensor:
    mirrored = points.clone()
    mirrored[..., 0] = 1.0 - mirrored[..., 0]
    return mirrored
