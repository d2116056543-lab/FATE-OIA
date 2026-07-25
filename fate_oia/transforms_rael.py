"""Synchronous image and task-aware grounding transforms for RAEL."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Callable, Mapping

import torch
from PIL import Image, ImageOps

from fate_oia.datasets.bdd100k_task_aware_index import RAELGroundingRecord
from fate_oia.transforms import IMAGENET_MEAN, IMAGENET_STD


def _to_tensor(image: Image.Image) -> torch.Tensor:
    data = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    return data.view(image.height, image.width, len(image.getbands())).permute(2, 0, 1).float() / 255.0


_DIRECTIONAL_FIELDS = frozenset({
    "approach_side", "direction", "lane_direction", "lane_side", "relative_position", "road_side", "sector", "side", "traffic_direction",
})


def _normalise_field_name(field: str) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", field).lower()


def _swap_direction_tokens(value: str) -> str:
    """Swap whole controlled direction tokens, never substrings such as bright/upright."""

    def swap(part: str) -> str:
        lower = part.lower()
        if lower not in {"left", "right"}:
            return part
        replacement = "right" if lower == "left" else "left"
        if part.isupper():
            return replacement.upper()
        if part.istitle():
            return replacement.title()
        return replacement

    parts = re.split(r"([_\-\s/]+)", value)
    return "".join(swap(part) for part in parts)


def _mirror_semantics(value: Any, *, field: str | None = None) -> Any:
    if isinstance(value, str):
        return _swap_direction_tokens(value) if field in _DIRECTIONAL_FIELDS else value
    if isinstance(value, Mapping):
        return {key: _mirror_semantics(item, field=_normalise_field_name(str(key))) for key, item in value.items()}
    if isinstance(value, list):
        return [_mirror_semantics(item, field=field) for item in value]
    if isinstance(value, tuple):
        return tuple(_mirror_semantics(item, field=field) for item in value)
    return value


def _points(value: Any, project: Callable[[float, float], tuple[float, float]]) -> Any:
    if isinstance(value, Mapping):
        if "x" in value and "y" in value:
            transformed = dict(value)
            transformed["x"], transformed["y"] = project(float(value["x"]), float(value["y"]))
            return transformed
        return {key: _points(item, project) for key, item in value.items()}
    if not isinstance(value, (list, tuple)):
        return value
    output = []
    for item in value:
        if isinstance(item, (list, tuple)) and len(item) >= 2 and all(isinstance(v, (int, float)) for v in item[:2]):
            x, y = project(float(item[0]), float(item[1]))
            output.append([x, y, *item[2:]])
        else:
            output.append(_points(item, project))
    return output


@dataclass(frozen=True)
class RAELGroundingTransformResult:
    image: torch.Tensor
    record: RAELGroundingRecord
    meta: dict[str, Any]


@dataclass
class RAELGroundingTransform:
    image_height: int = 360
    image_width: int = 640
    patch_size: int = 8
    normalize: bool = True
    fill: tuple[int, int, int] = (0, 0, 0)

    def _geometry(self, image: Image.Image) -> tuple[Image.Image, dict[str, Any]]:
        original_w, original_h = image.size
        scale = min(self.image_width / max(original_w, 1), self.image_height / max(original_h, 1))
        resized_w, resized_h = max(1, round(original_w * scale)), max(1, round(original_h * scale))
        resized = image.resize((resized_w, resized_h), Image.BILINEAR)
        pad_left = (self.image_width - resized_w) // 2
        pad_top = (self.image_height - resized_h) // 2
        boxed = ImageOps.expand(
            resized,
            border=(pad_left, pad_top, self.image_width - resized_w - pad_left, self.image_height - resized_h - pad_top),
            fill=self.fill,
        )
        return boxed, {
            "original_size": (original_w, original_h),
            "resized_size": (resized_w, resized_h),
            "padding": (pad_left, pad_top),
            "image_size": (self.image_width, self.image_height),
            "patch_grid": (self.image_height // self.patch_size, self.image_width // self.patch_size),
            "scale": scale,
        }

    def _transform_record(self, record: RAELGroundingRecord, meta: dict[str, Any], mirror: bool) -> RAELGroundingRecord:
        scale, (pad_left, pad_top) = float(meta["scale"]), meta["padding"]

        def project(x: float, y: float) -> tuple[float, float]:
            transformed_x, transformed_y = x * scale + pad_left, y * scale + pad_top
            return (self.image_width - transformed_x if mirror else transformed_x), transformed_y

        def transform_item(item: Mapping[str, Any], point_key: str | None) -> dict[str, Any]:
            transformed = _mirror_semantics(dict(item)) if mirror else dict(item)
            if "box" in transformed:
                x1, y1, x2, y2 = (float(value) for value in transformed["box"])
                px1, py1 = project(x1, y1)
                px2, py2 = project(x2, y2)
                transformed["box"] = [min(px1, px2), py1, max(px1, px2), py2]
            if point_key and point_key in transformed:
                transformed[point_key] = _points(transformed[point_key], project)
            return transformed

        detached = record.mutable_copy()
        return RAELGroundingRecord(
            detections=tuple(transform_item(item, None) for item in detached["detections"]),
            lanes=tuple(transform_item(item, "points") for item in detached["lanes"]),
            drivable=tuple(transform_item(item, "polygon") for item in detached["drivable"]),
            source_complete=detached["source_complete"],
        )

    def __call__(self, image: Image.Image, record: RAELGroundingRecord, *, mirror: bool = False) -> RAELGroundingTransformResult:
        if image.mode != "RGB":
            image = image.convert("RGB")
        boxed, meta = self._geometry(image)
        if mirror:
            boxed = ImageOps.mirror(boxed)
        tensor = _to_tensor(boxed)
        if self.normalize:
            tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
        meta = {**meta, "mirror": mirror}
        return RAELGroundingTransformResult(tensor, self._transform_record(record, meta, mirror), meta)
