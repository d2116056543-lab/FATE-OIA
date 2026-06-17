from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml


SOURCE_IDS = {"object_box": 0, "lane_poly": 1, "drivable": 2, "weak_region": 3, "missing": -1}


class ACPRPredicatePatchTargetBuilder:
    def __init__(self, scene_config, bdd100k_root=None, grid_hw=(45, 80), image_size=(1280, 720)):
        self.scene_config = Path(scene_config)
        data = yaml.safe_load(self.scene_config.read_text(encoding="utf-8")) or {}
        self.predicates = list(data.get("predicates", []))
        self.names = [str(p.get("name", i)) for i, p in enumerate(self.predicates)]
        self.num_predicates = len(self.predicates)
        self.bdd100k_root = Path(bdd100k_root) if bdd100k_root else None
        self.grid_hw = tuple(grid_hw)
        self.image_size = tuple(image_size)

    def _empty(self, b: int, device=None):
        h, w = self.grid_hw
        n = h * w
        return {
            "predicate_patch_targets": torch.zeros(b, self.num_predicates, n, device=device),
            "predicate_patch_mask": torch.zeros(b, self.num_predicates, device=device),
            "predicate_patch_reliability": torch.zeros(b, self.num_predicates, device=device),
            "predicate_source": torch.full((b, self.num_predicates), -1, dtype=torch.long, device=device),
            "predicate_patch_coverage": {},
        }

    def _bbox_to_mask(self, box: dict[str, float], device=None):
        h, w = self.grid_hw
        img_w, img_h = self.image_size
        x1 = int(max(0, min(w - 1, float(box.get("x1", 0)) / img_w * w)))
        x2 = int(max(0, min(w - 1, float(box.get("x2", 0)) / img_w * w)))
        y1 = int(max(0, min(h - 1, float(box.get("y1", 0)) / img_h * h)))
        y2 = int(max(0, min(h - 1, float(box.get("y2", 0)) / img_h * h)))
        if x2 < x1: x1, x2 = x2, x1
        if y2 < y1: y1, y2 = y2, y1
        mask = torch.zeros(h, w, device=device)
        mask[y1 : y2 + 1, x1 : x2 + 1] = 1.0
        return mask.reshape(-1)

    def _region_mask(self, region: str, device=None):
        h, w = self.grid_hw
        yy = torch.linspace(0, 1, h, device=device).view(h, 1).expand(h, w)
        xx = torch.linspace(0, 1, w, device=device).view(1, w).expand(h, w)
        if "left" in region:
            mask = (xx < 0.45) & (yy > 0.35)
        elif "right" in region:
            mask = (xx > 0.55) & (yy > 0.35)
        elif "upper" in region or "traffic" in region:
            mask = yy < 0.45
        elif "front" in region:
            mask = (xx > 0.35) & (xx < 0.65) & (yy > 0.35)
        else:
            mask = yy > 0.55
        return mask.float().reshape(-1)

    def build(self, file_names: list[str], device=None, records: dict[str, Any] | None = None) -> dict:
        out = self._empty(len(file_names), device=device)
        if records is None:
            records = {}
        for bi, fn in enumerate(file_names):
            stem = Path(fn).name
            rec = records.get(stem) or records.get(fn)
            if rec is None:
                continue
            objects = rec.get("objects", []) or []
            lanes = rec.get("lanes", []) or []
            has_drivable = bool(rec.get("drivable"))
            for pi, pred in enumerate(self.predicates):
                name = str(pred.get("name", ""))
                region = str(pred.get("region", name))
                target = None
                source = SOURCE_IDS["missing"]
                rel = 0.0
                for obj in objects:
                    cat = str(obj.get("category", "")).lower()
                    if any(k in name.lower() for k in [cat, "vehicle", "car", "pedestrian", "traffic"]):
                        box = obj.get("box2d") or obj.get("box") or {}
                        target = self._bbox_to_mask(box, device=device)
                        source = SOURCE_IDS["object_box"]
                        rel = 0.85
                        break
                if target is None and lanes and ("lane" in name.lower() or "left" in name.lower() or "right" in name.lower()):
                    target = self._region_mask(region, device=device)
                    source = SOURCE_IDS["lane_poly"]
                    rel = 0.65
                if target is None and has_drivable and ("drivable" in name.lower() or "road" in name.lower()):
                    target = self._region_mask(region, device=device)
                    source = SOURCE_IDS["drivable"]
                    rel = 0.55
                if target is None and rec.get("allow_weak_region", False):
                    target = self._region_mask(region, device=device)
                    source = SOURCE_IDS["weak_region"]
                    rel = 0.25
                if target is not None and float(target.sum()) > 0:
                    out["predicate_patch_targets"][bi, pi] = target
                    out["predicate_patch_mask"][bi, pi] = 1.0
                    out["predicate_patch_reliability"][bi, pi] = rel
                    out["predicate_source"][bi, pi] = source
        valid = out["predicate_patch_mask"].float()
        out["predicate_patch_coverage"] = {
            "valid_predicate_mask_rate": float(valid.mean().detach().cpu()) if valid.numel() else 0.0,
            "object_box_count": int((out["predicate_source"] == SOURCE_IDS["object_box"]).sum().detach().cpu()),
            "lane_poly_count": int((out["predicate_source"] == SOURCE_IDS["lane_poly"]).sum().detach().cpu()),
            "drivable_count": int((out["predicate_source"] == SOURCE_IDS["drivable"]).sum().detach().cpu()),
            "weak_region_count": int((out["predicate_source"] == SOURCE_IDS["weak_region"]).sum().detach().cpu()),
        }
        return out
