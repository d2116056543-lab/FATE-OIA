from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw

from fate_oia.datasets.bdd100k_grounding import BDD100KGroundingIndex, load_bdd100k_objects
from fate_oia.models.acpr_grounded_evidence_memory import ACPREvidenceSlotSpec, load_evidence_slot_specs


class _LRU:
    def __init__(self, limit: int = 256) -> None:
        self.limit = int(limit)
        self.data: OrderedDict[str, torch.Tensor] = OrderedDict()

    def get(self, key: str) -> torch.Tensor | None:
        if key not in self.data:
            return None
        val = self.data.pop(key)
        self.data[key] = val
        return val

    def put(self, key: str, val: torch.Tensor) -> None:
        self.data[key] = val
        while len(self.data) > self.limit:
            self.data.popitem(last=False)


def _box_center(box: dict[str, Any]) -> tuple[float, float]:
    x1 = float(box.get("x1", box.get("left", 0.0)))
    x2 = float(box.get("x2", box.get("right", x1)))
    y1 = float(box.get("y1", box.get("top", 0.0)))
    y2 = float(box.get("y2", box.get("bottom", y1)))
    return (x1 + x2) * 0.5 / 1280.0, (y1 + y2) * 0.5 / 720.0


def _slot_region_match(slot: ACPREvidenceSlotSpec, x: float, y: float) -> bool:
    if slot.region_prior == "global":
        return True
    if slot.region_prior == "upper_traffic_region":
        return y < 0.45
    if slot.region_prior == "left_corridor":
        return x < 0.50 and y > 0.30
    if slot.region_prior == "right_corridor":
        return x > 0.50 and y > 0.30
    if slot.region_prior == "bottom_drivable_region":
        return y > 0.50
    return 0.25 <= x <= 0.75


def _draw_box(mask: Image.Image, box: dict[str, Any], output_size: tuple[int, int]) -> None:
    out_h, out_w = output_size
    raw_x1 = float(box.get("x1", box.get("left", 0)))
    raw_x2 = float(box.get("x2", box.get("right", raw_x1)))
    raw_y1 = float(box.get("y1", box.get("top", 0)))
    raw_y2 = float(box.get("y2", box.get("bottom", raw_y1)))
    x1 = int(round(min(raw_x1, raw_x2) / 1280.0 * out_w))
    x2 = int(round(max(raw_x1, raw_x2) / 1280.0 * out_w))
    y1 = int(round(min(raw_y1, raw_y2) / 720.0 * out_h))
    y2 = int(round(max(raw_y1, raw_y2) / 720.0 * out_h))
    x1 = max(0, min(out_w - 1, x1))
    x2 = max(0, min(out_w - 1, x2))
    y1 = max(0, min(out_h - 1, y1))
    y2 = max(0, min(out_h - 1, y2))
    if x2 < x1 or y2 < y1:
        return
    ImageDraw.Draw(mask).rectangle([x1, y1, x2, y2], fill=1)


def _poly_vertices(poly: Any) -> list[tuple[float, float]]:
    if isinstance(poly, dict):
        vertices = poly.get("vertices") or poly.get("points") or poly.get("verts") or []
    else:
        vertices = poly
    out: list[tuple[float, float]] = []
    if not isinstance(vertices, list):
        return out
    for item in vertices:
        if isinstance(item, dict) and "x" in item and "y" in item:
            out.append((float(item["x"]), float(item["y"])))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            out.append((float(item[0]), float(item[1])))
    return out


def _draw_polyline(mask: Image.Image, poly: Any, output_size: tuple[int, int]) -> None:
    out_h, out_w = output_size
    if (
        isinstance(poly, list)
        and poly
        and all(isinstance(v, (list, tuple)) and len(v) >= 2 and isinstance(v[0], (int, float)) for v in poly)
    ):
        entries = [poly]
    else:
        entries = poly if isinstance(poly, list) else [poly]
    draw = ImageDraw.Draw(mask)
    for entry in entries:
        vertices = _poly_vertices(entry)
        if len(vertices) < 2:
            continue
        pts = [
            (
                max(0, min(out_w - 1, int(round(x / 1280.0 * out_w)))),
                max(0, min(out_h - 1, int(round(y / 720.0 * out_h)))),
            )
            for x, y in vertices
        ]
        draw.line(pts, fill=1, width=max(1, int(round(min(out_h, out_w) / 45))))


def _image_to_mask(path: str | Path, output_size: tuple[int, int]) -> torch.Tensor:
    out_h, out_w = output_size
    img = Image.open(path).convert("L").resize((out_w, out_h), Image.NEAREST)
    data = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes())).view(out_h, out_w).float()
    return (data > 0).float()


class ACPRGEMGroundingBuilder:
    def __init__(self, bdd100k_root: str | Path, slots_config: str | Path = "configs/acpr_gem_evidence_slots.yaml", output_size: tuple[int, int] = (45, 80), lru_size: int = 256) -> None:
        self.index = BDD100KGroundingIndex(bdd100k_root)
        self.slot_specs = load_evidence_slot_specs(slots_config)
        self.output_size = output_size
        self.drivable_lru = _LRU(lru_size)

    def _drivable_mask(self, path: str | None) -> torch.Tensor:
        if not path:
            return torch.zeros(self.output_size)
        cached = self.drivable_lru.get(path)
        if cached is not None:
            return cached
        mask = _image_to_mask(path, self.output_size)
        self.drivable_lru.put(path, mask)
        return mask

    def _objects_for(self, label_json: str | None) -> list[dict[str, Any]]:
        if not label_json:
            return []
        try:
            return load_bdd100k_objects(label_json)
        except Exception:
            try:
                data = json.loads(Path(label_json).read_text(encoding="utf-8", errors="ignore"))
                return list(data.get("labels", []))
            except Exception:
                return []

    def _slot_mask(self, slot: ACPREvidenceSlotSpec, objects: list[dict[str, Any]], drivable: torch.Tensor) -> torch.Tensor:
        out_h, out_w = self.output_size
        mask_img = Image.new("L", (out_w, out_h), 0)
        for obj in objects:
            cat = str(obj.get("category", "")).lower()
            box = obj.get("box2d") or obj.get("box")
            poly = obj.get("poly2d") or obj.get("polyline")
            if slot.use_object_boxes and isinstance(box, dict):
                x, y = _box_center(box)
                object_like = any(k in cat for k in ["car", "truck", "bus", "person", "pedestrian", "rider", "bike", "traffic light", "traffic sign", "obstacle", "crosswalk", "intersection"])
                if object_like and _slot_region_match(slot, x, y):
                    _draw_box(mask_img, box, self.output_size)
            if slot.use_lane_polylines and ("lane" in cat or poly):
                _draw_polyline(mask_img, poly, self.output_size)
        mask = torch.ByteTensor(torch.ByteStorage.from_buffer(mask_img.tobytes())).view(out_h, out_w).float()
        if slot.use_drivable_masks and drivable.numel():
            if slot.region_prior in {"left_corridor", "right_corridor", "bottom_drivable_region", "global"}:
                mask = torch.maximum(mask, drivable.float())
        return mask.clamp(0, 1)

    def build(self, file_names: list[str], device: torch.device | None = None) -> dict[str, Any]:
        b = len(file_names)
        m = len(self.slot_specs)
        n = self.output_size[0] * self.output_size[1]
        targets = torch.zeros(b, m, n, dtype=torch.float32)
        mask = torch.zeros(b, m, dtype=torch.float32)
        group_counts = {"object_slot_available": 0, "lane_slot_available": 0, "drivable_slot_available": 0}
        for i, fn in enumerate(file_names):
            paths = self.index.lookup(str(fn))
            objects = self._objects_for(paths.label_json)
            drivable = self._drivable_mask(paths.drivable_map)
            for j, slot in enumerate(self.slot_specs):
                sm = self._slot_mask(slot, objects, drivable).reshape(-1)
                if sm.sum() > 0 and slot.oracle_allowed:
                    targets[i, j] = sm
                    mask[i, j] = 1.0
                    if slot.group == "object":
                        group_counts["object_slot_available"] += 1
                    elif slot.group == "lane":
                        group_counts["lane_slot_available"] += 1
                    elif slot.group == "drivable":
                        group_counts["drivable_slot_available"] += 1
        if device is not None:
            targets = targets.to(device)
            mask = mask.to(device)
        return {
            "grounding_targets": targets,
            "grounding_mask": mask,
            "grounding_stats": {
                **group_counts,
                "available_rate": float(mask.mean().detach().cpu()) if mask.numel() else 0.0,
            },
        }
