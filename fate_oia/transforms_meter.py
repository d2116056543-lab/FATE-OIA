from __future__ import annotations

import random
from dataclasses import dataclass

from PIL import Image, ImageEnhance

from fate_oia.transforms import AspectRatioLetterboxTransform


@dataclass
class METERImageTransform:
    training: bool = False
    brightness_contrast: float = 0.10

    def __post_init__(self) -> None:
        self.letterbox = AspectRatioLetterboxTransform(
            image_height=360,
            image_width=640,
            patch_size=8,
            normalize=True,
            return_meta=True,
        )

    def __call__(self, image: Image.Image):
        if self.training and self.brightness_contrast > 0:
            span = float(self.brightness_contrast)
            image = ImageEnhance.Brightness(image).enhance(
                1.0 + random.uniform(-span, span)
            )
            image = ImageEnhance.Contrast(image).enhance(
                1.0 + random.uniform(-span, span)
            )
        return self.letterbox(image)


def meter_image_transform(*, training: bool = False) -> METERImageTransform:
    return METERImageTransform(training=training)
