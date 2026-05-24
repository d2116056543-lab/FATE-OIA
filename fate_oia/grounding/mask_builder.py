from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch


def _scale_box(box: dict[str, float], width: int, height: int, out_w: int, out_h: int) -> tuple[int, int, int, int]:
    x1 = int(round(float(box.get("x1", 0.0)) / max(width, 1) * out_w))
    y1 = int(round(float(box.get("y1", 0.0)) / max(height, 1) * out_h))
    x2 = int(round(float(box.get("x2", 0.0)) / max(width, 1) * out_w))
    y2 = int(round(float(box.get("y2", 0.0)) / max(height, 1) * out_h))
    x1, x2 = sorted((max(0, min(out_w, x1)), max(0, min(out_w, x2))))
    y1, y2 = sorted((max(0, min(out_h, y1)), max(0, min(out_h, y2))))
    return x1, y1, x2, y2


def objects_to_mask(
    objects: Iterable[dict[str, Any]],
    image_size: tuple[int, int],
    output_size: tuple[int, int],
    categories: set[str] | None = None,
) -> torch.Tensor:
    """Build a binary mask from BDD100K box2d objects.

    Poly2d rasterization is intentionally not approximated here; box2d is the
    stable full-split grounding primitive. Semantic/polygon subsets can be
    evaluated separately.
    """
    width, height = image_size
    out_h, out_w = output_size
    mask = torch.zeros(out_h, out_w, dtype=torch.float32)
    for obj in objects:
        cat = str(obj.get("category", ""))
        if categories is not None and cat not in categories:
            continue
        box = obj.get("box2d")
        if not isinstance(box, dict):
            continue
        x1, y1, x2, y2 = _scale_box(box, width, height, out_w, out_h)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1.0
    return mask