from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch import Tensor

from fate_oia.datasets.bdd100k_grounding import BDD100KGroundingIndex


class CoEVGroundingTargetBuilder:
    """Conservative eight-map BDD100K supervision; absent sources remain unknown."""

    categories = {"car": 0, "truck": 0, "bus": 0, "pedestrian": 1, "person": 1,
                  "bicycle": 2, "bike": 2, "traffic sign": 5}

    def __init__(self, root: str | Path, grid_hw: tuple[int, int] = (16, 28)) -> None:
        self.index = BDD100KGroundingIndex(root); self.grid_hw = grid_hw

    def build(self, file_name: str, flipped: bool = False) -> tuple[Tensor, Tensor, Tensor]:
        h, w = self.grid_hw
        value = torch.zeros(8, h, w); known = torch.zeros_like(value, dtype=torch.bool)
        weight = torch.zeros_like(value)
        paths = self.index.lookup(file_name)
        if paths.label_json:
            payload = json.loads(Path(paths.label_json).read_text(encoding="utf-8", errors="ignore"))
            items = list(payload.get("labels", []))
            for frame in payload.get("frames", []) or []:
                items.extend(frame.get("labels", [])); items.extend(frame.get("objects", []))
            source_classes = {0, 1, 2, 5}
            for item in items:
                category = str(item.get("category", "")).lower()
                attrs = item.get("attributes") or item.get("attribute") or {}
                index = self.categories.get(category)
                if category == "traffic light":
                    color = str(attrs.get("trafficLightColor", attrs.get("color", ""))).lower()
                    index = 3 if color == "red" else 4 if color == "green" else None
                    if index is not None:
                        source_classes.update((3, 4))
                elif index is not None:
                    source_classes.add(index)
                if index is None or not isinstance(item.get("box2d") or item.get("box"), dict):
                    continue
                box = item.get("box2d") or item.get("box")
                x1 = float(box.get("x1", box.get("left", 0))) / 1280; x2 = float(box.get("x2", box.get("right", 0))) / 1280
                y1 = float(box.get("y1", box.get("top", 0))) / 720; y2 = float(box.get("y2", box.get("bottom", 0))) / 720
                xa, xb = int(max(0, x1) * w), max(int(max(0, x1) * w) + 1, int(min(1, x2) * w))
                ya, yb = int(max(0, y1) * h), max(int(max(0, y1) * h) + 1, int(min(1, y2) * h))
                value[index, ya:min(h, yb), xa:min(w, xb)] = 1
            for index in source_classes:
                known[index] = True; weight[index] = .8 if index in (0, 1, 2, 5) else 1.0
        if paths.drivable_map:
            image = Image.open(paths.drivable_map).convert("L").resize((w, h), Image.Resampling.NEAREST)
            value[6] = torch.as_tensor(list(image.getdata()), dtype=torch.float32).reshape(h, w).gt(0)
            known[6] = True; weight[6] = 1
        # Lane polylines have independent annotation semantics; lane absence is not inferred without the source.
        lane_sources=[path for path in (paths.label_json,paths.lane_json) if path]
        if lane_sources:
            canvas = Image.new("L", (w, h)); draw = ImageDraw.Draw(canvas); found = False
            items=[]
            for path in lane_sources:
                payload=json.loads(Path(path).read_text(encoding="utf-8",errors="ignore"))
                payloads=payload if isinstance(payload,list) else [payload]
                for entry in payloads:
                    items.extend(entry.get("labels",[]));items.extend(entry.get("laneLines",[]))
                    for frame in entry.get("frames",[]) or []:items.extend(frame.get("labels",[]));items.extend(frame.get("laneLines",[]))
            for item in items:
                if "lane" not in str(item.get("category",item.get("type","lane"))).lower(): continue
                raw = item.get("poly2d") or item.get("polyline") or []
                if raw and isinstance(raw[0], dict) and "vertices" in raw[0]: raw = raw[0]["vertices"]
                pts = [(float(p[0] if isinstance(p, (list, tuple)) else p.get("x", 0)) / 1280 * w,
                        float(p[1] if isinstance(p, (list, tuple)) else p.get("y", 0)) / 720 * h) for p in raw]
                if len(pts) > 1: draw.line(pts, fill=255, width=2); found = True
            if found:
                value[7] = torch.as_tensor(list(canvas.getdata()), dtype=torch.float32).reshape(h, w) / 255
                known[7] = True; weight[7] = .8
        if flipped:
            value = value.flip(-1); known = known.flip(-1); weight = weight.flip(-1)
        return value, known, weight
