from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from PIL import Image, ImageDraw


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
    include_box2d: bool = True,
    include_poly2d: bool = True,
    include_drivable: bool = True,
    include_lane: bool = True,
) -> torch.Tensor:
    """Build a binary mask from BDD100K box2d/poly2d objects."""
    width, height = image_size
    out_h, out_w = output_size
    mask = torch.zeros(out_h, out_w, dtype=torch.float32)
    for obj in objects:
        cat = str(obj.get("category", ""))
        if not include_lane and cat.startswith("lane/"):
            continue
        if not include_drivable and cat.startswith("area/"):
            continue
        if categories is not None and cat not in categories:
            continue
        if include_box2d:
            box = obj.get("box2d")
            if isinstance(box, dict):
                x1, y1, x2, y2 = _scale_box(box, width, height, out_w, out_h)
                if x2 > x1 and y2 > y1:
                    mask[y1:y2, x1:x2] = 1.0
        if include_poly2d:
            poly_mask = poly2d_to_mask(obj.get("poly2d") or [], image_size=image_size, output_size=output_size)
            if poly_mask.numel():
                mask = torch.maximum(mask, poly_mask)
    return mask


def _extract_vertices(poly: Any) -> list[tuple[float, float]]:
    if isinstance(poly, dict):
        vertices = poly.get("vertices") or poly.get("verts") or poly.get("points")
    else:
        vertices = poly
    out: list[tuple[float, float]] = []
    if not isinstance(vertices, list):
        return out
    for item in vertices:
        if isinstance(item, dict):
            if "x" in item and "y" in item:
                out.append((float(item["x"]), float(item["y"])))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            out.append((float(item[0]), float(item[1])))
    return out


def poly2d_to_mask(poly2d: Any, image_size: tuple[int, int], output_size: tuple[int, int]) -> torch.Tensor:
    """Rasterize BDD100K-style poly2d entries to an output mask."""
    width, height = image_size
    out_h, out_w = output_size
    mask_img = Image.new("L", (out_w, out_h), 0)
    draw = ImageDraw.Draw(mask_img)
    polys = poly2d if isinstance(poly2d, list) else [poly2d]
    for poly in polys:
        vertices = _extract_vertices(poly)
        if len(vertices) < 3:
            continue
        scaled = [
            (
                max(0, min(out_w, int(round(x / max(width, 1) * out_w)))),
                max(0, min(out_h, int(round(y / max(height, 1) * out_h)))),
            )
            for x, y in vertices
        ]
        draw.polygon(scaled, fill=1)
    data = torch.ByteTensor(torch.ByteStorage.from_buffer(mask_img.tobytes()))
    return data.view(out_h, out_w).float()


def drivable_map_to_mask(path: str, output_size: tuple[int, int], positive_values: set[int] | None = None) -> torch.Tensor:
    """Load a BDD100K drivable/lane map and downsample it to an attention grid."""
    positive_values = positive_values or {1, 2, 255}
    image = Image.open(path).convert("L").resize((output_size[1], output_size[0]), Image.NEAREST)
    data = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes())).view(output_size[0], output_size[1])
    mask = torch.zeros_like(data, dtype=torch.float32)
    for value in positive_values:
        mask = torch.maximum(mask, (data == int(value)).float())
    return mask
