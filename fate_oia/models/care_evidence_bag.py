from __future__ import annotations

from typing import Any

import torch


def _box_features(box: dict[str, float]) -> list[float]:
    x1 = float(box.get("x1", box.get("x", 0.0)))
    y1 = float(box.get("y1", box.get("y", 0.0)))
    x2 = float(box.get("x2", x1 + box.get("w", 1.0)))
    y2 = float(box.get("y2", y1 + box.get("h", 1.0)))
    w = max(x2 - x1, 1.0)
    h = max(y2 - y1, 1.0)
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h
    area = w * h
    return [cx / 1280.0, cy / 720.0, w / 1280.0, h / 720.0, area / (1280.0 * 720.0), w / h, cy / 720.0, 1.0]


def _lane_features(lane: dict[str, Any]) -> list[float]:
    polys = lane.get("poly2d") or []
    vertices = []
    for p in polys:
        vertices.extend(p.get("vertices") or [])
    if not vertices:
        return [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    xs = [float(v[0]) for v in vertices]
    ys = [float(v[1]) for v in vertices]
    return [sum(xs) / len(xs) / 1280.0, sum(ys) / len(ys) / 720.0, (max(xs) - min(xs)) / 1280.0, (max(ys) - min(ys)) / 720.0, len(vertices) / 100.0, 0.0, 0.0, 1.0]


class EvidenceBagBuilder:
    """Build weak privileged evidence bags from reason indices and BDD100K records.

    Reason display names are intentionally not used here; reason indices remain canonical.
    """

    source_types = ["object", "lane", "drivable", "traffic_control", "global_context"]

    def __init__(self, max_items_per_type: int = 16) -> None:
        self.max_items_per_type = max_items_per_type

    def build(self, structured: list[dict[str, Any] | None], reason_targets: torch.Tensor | None, device: torch.device | None = None) -> dict[str, Any]:
        device = device or (reason_targets.device if reason_targets is not None else torch.device("cpu"))
        b = len(structured)
        feats: dict[str, list[list[list[float]]]] = {k: [] for k in self.source_types}
        counts = {k: [] for k in self.source_types}
        for rec in structured:
            rec = rec or {}
            objects = list(rec.get("objects") or [])
            lanes = list(rec.get("lanes") or [])
            attrs = dict(rec.get("attributes") or {})
            obj_feats = [_box_features(o["box2d"]) for o in objects if isinstance(o.get("box2d"), dict)]
            traffic_feats = [_box_features(o["box2d"]) for o in objects if isinstance(o.get("box2d"), dict) and str(o.get("category", "")).lower() in {"traffic light", "traffic sign"}]
            lane_feats = [_lane_features(x) for x in lanes]
            drive_feats = [[0.5, 0.75, 1.0, 0.5, 0.5, 1.0, 0.75, 1.0]] if rec.get("drivable") else []
            global_feats = [[float(hash(str(attrs.get(k))) % 997) / 997.0 for k in ("weather", "timeofday", "scene")] + [0.0, 0.0, 0.0, 0.0, 1.0]]
            groups = {"object": obj_feats, "lane": lane_feats, "drivable": drive_feats, "traffic_control": traffic_feats, "global_context": global_feats}
            for key in self.source_types:
                vals = groups[key][: self.max_items_per_type]
                counts[key].append(len(vals))
                while len(vals) < self.max_items_per_type:
                    vals.append([0.0] * 8)
                feats[key].append(vals)
        tensors = {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in feats.items()}
        count_tensors = {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in counts.items()}
        coverage = torch.zeros(b, 21, device=device)
        if reason_targets is not None:
            coverage = (reason_targets > 0.5).float()
        return {"features": tensors, "counts": count_tensors, "positive_reason_mask": coverage}
