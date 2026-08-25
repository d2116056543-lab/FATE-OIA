from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset

from fate_oia.transforms_video import _letterbox, _normalize


ROLE_NAMES = ("background", "vehicle", "vulnerable_road_user", "traffic_control", "lane_drivable")
ROLE_BY_CATEGORY = {
    "car": 1, "bus": 1, "truck": 1, "train": 1,
    "motor": 2, "bike": 2, "rider": 2, "person": 2,
    "traffic light": 3, "traffic sign": 3,
    "area/drivable": 4, "area/alternative": 4,
    "lane/road curb": 4, "lane/crosswalk": 4,
    "lane/single white": 4, "lane/single yellow": 4,
    "lane/double white": 4, "lane/double yellow": 4,
}


def _poly_points(value: Any) -> list[tuple[float, float]]:
    if isinstance(value, dict):
        if "value" in value:
            return [(float(value["value"][0]), float(value["value"][1]))]
        if "vertices" in value:
            return _poly_points(value["vertices"])
        return []
    if isinstance(value, (list, tuple)):
        if len(value) >= 2 and isinstance(value[0], (int, float)) and isinstance(value[1], (int, float)):
            return [(float(value[0]), float(value[1]))]
        points: list[tuple[float, float]] = []
        for item in value:
            points.extend(_poly_points(item))
        return points
    return []


def annotation_to_patch_roles(
    annotation: dict[str, Any],
    *,
    source_hw: tuple[int, int] = (720, 1280),
    grid_hw: tuple[int, int] = (45, 80),
) -> torch.Tensor:
    """Rasterize BDD100K object/lane supervision onto DINO patch roles."""
    source_h, source_w = source_hw
    grid_h, grid_w = grid_hw
    canvas = Image.new("L", (grid_w, grid_h), color=0)
    draw = ImageDraw.Draw(canvas)
    objects = annotation.get("frames", [{}])[-1].get("objects", [])
    # Broad lane/drivable regions are painted first; compact agents override.
    ordered = sorted(objects, key=lambda obj: 0 if ROLE_BY_CATEGORY.get(obj.get("category")) == 4 else 1)
    for obj in ordered:
        role = ROLE_BY_CATEGORY.get(str(obj.get("category", "")))
        if role is None:
            continue
        box = obj.get("box2d")
        if box is not None:
            x1 = max(0, min(grid_w - 1, int(float(box["x1"]) / source_w * grid_w)))
            y1 = max(0, min(grid_h - 1, int(float(box["y1"]) / source_h * grid_h)))
            x2 = max(x1, min(grid_w - 1, int(float(box["x2"]) / source_w * grid_w)))
            y2 = max(y1, min(grid_h - 1, int(float(box["y2"]) / source_h * grid_h)))
            draw.rectangle((x1, y1, x2, y2), fill=role)
        vertices = _poly_points(obj.get("poly2d", []))
        if len(vertices) >= 2:
            points = [(max(0, min(grid_w - 1, int(x / source_w * grid_w))),
                       max(0, min(grid_h - 1, int(y / source_h * grid_h)))) for x, y in vertices]
            if len(points) == 2:
                draw.line(points, fill=role, width=1)
            else:
                draw.polygon(points, fill=role)
    return torch.from_numpy(__import__("numpy").array(canvas, dtype="int64"))


class BDD100KObjectRoleDataset(Dataset):
    """Official BDD100K train images with patch-level weak role labels."""

    def __init__(
        self,
        image_root: str | Path,
        label_root: str | Path,
        *,
        file_names: list[str] | None = None,
        target_hw: tuple[int, int] = (360, 640),
        grid_hw: tuple[int, int] = (45, 80),
    ) -> None:
        self.image_root = Path(image_root)
        self.label_root = Path(label_root)
        self.target_hw = target_hw
        self.grid_hw = grid_hw
        if file_names is None:
            names = sorted(path.stem for path in self.label_root.glob("*.json"))
        else:
            names = [Path(name).stem for name in file_names]
        self.names = [
            name for name in names
            if (self.image_root / f"{name}.jpg").is_file() and (self.label_root / f"{name}.json").is_file()
        ]
        if not self.names:
            raise ValueError("BDD100K role dataset has no matched image/label pairs")

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> dict[str, Any]:
        name = self.names[index]
        image_path = self.image_root / f"{name}.jpg"
        label_path = self.label_root / f"{name}.json"
        with Image.open(image_path) as source:
            source = source.convert("RGB")
            source_hw = (source.height, source.width)
            boxed, _ = _letterbox(source, self.target_hw)
            image = _normalize(boxed)
        annotation = json.loads(label_path.read_text(encoding="utf-8"))
        roles = annotation_to_patch_roles(annotation, source_hw=source_hw, grid_hw=self.grid_hw)
        return {"image": image, "role_target": roles.flatten(), "file_name": f"{name}.jpg"}
