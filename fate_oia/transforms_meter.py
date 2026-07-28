from __future__ import annotations

from fate_oia.transforms import AspectRatioLetterboxTransform


def meter_image_transform() -> AspectRatioLetterboxTransform:
    return AspectRatioLetterboxTransform(image_height=360, image_width=640, patch_size=8, normalize=True, return_meta=True)
