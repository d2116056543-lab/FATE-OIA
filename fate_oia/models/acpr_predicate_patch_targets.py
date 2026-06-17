from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import yaml

try:
    from PIL import Image
except Exception:  # pragma: no cover - PIL is available in the training env, but keep import safe.
    Image = None


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
        self._label_paths: dict[str, Path] = {}
        self._drivable_paths: dict[str, Path] = {}
        self._record_cache: dict[str, dict[str, Any] | None] = {}
        self._index_bdd100k()

    @staticmethod
    def _key(name: str) -> str:
        stem = Path(str(name)).stem
        if stem.endswith("_drivable_color"):
            stem = stem[: -len("_drivable_color")]
        # BDD-OIA image names append a sampled frame suffix (e.g. ``_1``/``_3``)
        # to the BDD100K base stem. BDD100K labels/drivable maps are stored
        # under the base stem, so strip only a terminal numeric frame suffix.
        parts = stem.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            stem = parts[0]
        return stem

    def _index_bdd100k(self) -> None:
        if self.bdd100k_root is None or not self.bdd100k_root.exists():
            return
        label_root = self.bdd100k_root / "bdd100k_labels" / "bdd100k" / "labels" / "100k"
        if label_root.exists():
            for path in label_root.rglob("*.json"):
                self._label_paths[path.stem] = path
        drv_root = self.bdd100k_root / "bdd100k_drivable_maps" / "bdd100k" / "drivable_maps" / "color_labels"
        if drv_root.exists():
            for path in drv_root.rglob("*_drivable_color.png"):
                self._drivable_paths[self._key(path.name)] = path

    def _load_record(self, file_name: str) -> dict[str, Any] | None:
        key = self._key(file_name)
        if key in self._record_cache:
            return self._record_cache[key]
        path = self._label_paths.get(key)
        if path is None:
            self._record_cache[key] = None
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            self._record_cache[key] = None
            return None
        frame = (raw.get("frames") or [{}])[0]
        objects = list(frame.get("objects", []) or [])
        lanes = [o for o in objects if str(o.get("category", "")).lower().startswith("lane/")]
        drivable_objects = [o for o in objects if "drivable" in str(o.get("category", "")).lower()]
        rec = {
            "objects": objects,
            "lanes": lanes,
            "drivable_objects": drivable_objects,
            "drivable_path": self._drivable_paths.get(key),
            "drivable": bool(drivable_objects or key in self._drivable_paths),
            "allow_weak_region": True,
            "attributes": raw.get("attributes", {}),
        }
        self._record_cache[key] = rec
        return rec

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
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        mask = torch.zeros(h, w, device=device)
        mask[y1 : y2 + 1, x1 : x2 + 1] = 1.0
        return mask.reshape(-1)

    def _poly_to_mask(self, poly: list, device=None, dilation: int = 1):
        h, w = self.grid_hw
        img_w, img_h = self.image_size
        mask = torch.zeros(h, w, device=device)
        points: list[tuple[int, int]] = []
        for p in poly or []:
            if not isinstance(p, (list, tuple)) or len(p) < 2:
                continue
            x = int(max(0, min(w - 1, float(p[0]) / img_w * w)))
            y = int(max(0, min(h - 1, float(p[1]) / img_h * h)))
            points.append((x, y))
        if not points:
            return mask.reshape(-1)
        for (x1, y1), (x2, y2) in zip(points, points[1:] or points):
            steps = max(abs(x2 - x1), abs(y2 - y1), 1)
            for s in range(steps + 1):
                x = int(round(x1 + (x2 - x1) * s / steps))
                y = int(round(y1 + (y2 - y1) * s / steps))
                y0, y3 = max(0, y - dilation), min(h, y + dilation + 1)
                x0, x3 = max(0, x - dilation), min(w, x + dilation + 1)
                mask[y0:y3, x0:x3] = 1.0
        return mask.reshape(-1)

    def _drivable_image_to_mask(self, path: Path | None, device=None):
        if path is None or Image is None or not path.exists():
            return None
        try:
            img = Image.open(path).convert("RGB").resize((self.grid_hw[1], self.grid_hw[0]), Image.NEAREST)
            data = torch.as_tensor(list(img.getdata()), dtype=torch.float32, device=device).view(self.grid_hw[0], self.grid_hw[1], 3)
            mask = (data.sum(-1) > 20).float()
            return mask.reshape(-1)
        except Exception:
            return None

    def _region_mask(self, region: str, device=None):
        h, w = self.grid_hw
        yy = torch.linspace(0, 1, h, device=device).view(h, 1).expand(h, w)
        xx = torch.linspace(0, 1, w, device=device).view(1, w).expand(h, w)
        region_l = str(region).lower()
        if "left" in region_l:
            mask = (xx < 0.45) & (yy > 0.35)
        elif "right" in region_l:
            mask = (xx > 0.55) & (yy > 0.35)
        elif "upper" in region_l or "traffic" in region_l:
            mask = yy < 0.45
        elif "front" in region_l:
            mask = (xx > 0.35) & (xx < 0.65) & (yy > 0.35)
        else:
            mask = yy > 0.55
        return mask.float().reshape(-1)

    def _object_region_ok(self, obj: dict[str, Any], region: str) -> bool:
        box = obj.get("box2d") or obj.get("box") or {}
        if not box:
            return True
        img_w, img_h = self.image_size
        cx = (float(box.get("x1", 0)) + float(box.get("x2", img_w))) / (2.0 * img_w)
        cy = (float(box.get("y1", 0)) + float(box.get("y2", img_h))) / (2.0 * img_h)
        region_l = str(region).lower()
        if "left" in region_l:
            return cx < 0.55 and cy > 0.25
        if "right" in region_l:
            return cx > 0.45 and cy > 0.25
        if "upper" in region_l or "traffic" in region_l:
            return cy < 0.55
        if "front" in region_l:
            return 0.20 < cx < 0.80 and cy > 0.20
        return True

    def _object_matches(self, obj: dict[str, Any], pred: dict[str, Any]) -> bool:
        category = str(obj.get("category", "")).lower()
        name = str(pred.get("name", "")).lower()
        sources = [str(x).lower() for x in pred.get("bdd100k_sources", [])]
        if not sources:
            sources = [name]
        matched = any(src in category or category in src or src in name for src in sources if src not in {"lane", "drivable", "road", "global", "weather", "sky"})
        attrs = obj.get("attributes", {}) or {}
        color = str(attrs.get("trafficLightColor", "")).lower()
        if "traffic_light_red" in name:
            matched = matched and color == "red"
        if "traffic_light_green" in name:
            matched = matched and color == "green"
        return matched

    @staticmethod
    def _union(masks: list[torch.Tensor]) -> torch.Tensor | None:
        if not masks:
            return None
        out = masks[0].clone()
        for m in masks[1:]:
            out = torch.maximum(out, m)
        return out

    def build(self, file_names: list[str], device=None, records: dict[str, Any] | None = None) -> dict:
        out = self._empty(len(file_names), device=device)
        explicit_records = records or {}
        for bi, fn in enumerate(file_names):
            key = self._key(fn)
            rec = explicit_records.get(key) or explicit_records.get(Path(fn).name) or explicit_records.get(fn)
            if rec is None:
                rec = self._load_record(fn)
            if rec is None:
                continue
            objects = rec.get("objects", []) or []
            lanes = rec.get("lanes", []) or [o for o in objects if str(o.get("category", "")).lower().startswith("lane/")]
            drivable_objects = rec.get("drivable_objects", []) or [o for o in objects if "drivable" in str(o.get("category", "")).lower()]
            drivable_path = rec.get("drivable_path")
            has_drivable = bool(rec.get("drivable") or drivable_objects or drivable_path)
            for pi, pred in enumerate(self.predicates):
                name = str(pred.get("name", ""))
                name_l = name.lower()
                group = str(pred.get("group", "")).lower()
                region = str(pred.get("region", name))
                target = None
                source = SOURCE_IDS["missing"]
                rel = 0.0

                obj_masks = []
                for obj in objects:
                    if self._object_matches(obj, pred) and self._object_region_ok(obj, region):
                        if obj.get("box2d") or obj.get("box"):
                            obj_masks.append(self._bbox_to_mask(obj.get("box2d") or obj.get("box") or {}, device=device))
                        elif obj.get("poly2d"):
                            obj_masks.append(self._poly_to_mask(obj.get("poly2d") or [], device=device, dilation=1))
                target = self._union(obj_masks)
                if target is not None:
                    source = SOURCE_IDS["object_box"]
                    rel = 0.85

                if target is None and (group == "lane" or "lane" in name_l or "left" in name_l or "right" in name_l):
                    lane_masks = [self._poly_to_mask(l.get("poly2d") or [], device=device, dilation=1) for l in lanes]
                    target = self._union([m for m in lane_masks if float(m.sum()) > 0])
                    if target is not None:
                        source = SOURCE_IDS["lane_poly"]
                        rel = 0.65

                if target is None and has_drivable and (group in {"drivable", "road", "lane"} or "drivable" in name_l or "road" in name_l or "gap" in name_l):
                    target = self._drivable_image_to_mask(Path(drivable_path) if drivable_path else None, device=device)
                    if target is None and drivable_objects:
                        target = self._union([self._poly_to_mask(o.get("poly2d") or [], device=device, dilation=1) for o in drivable_objects])
                    if target is not None and float(target.sum()) > 0:
                        source = SOURCE_IDS["drivable"]
                        rel = 0.55

                if target is None and rec.get("allow_weak_region", False) and (group in {"global", "road"} or any(x in name_l for x in ["global", "visibility", "road", "region"])):
                    target = self._region_mask(region, device=device)
                    source = SOURCE_IDS["weak_region"]
                    rel = 0.20

                if target is not None and float(target.sum()) > 0:
                    out["predicate_patch_targets"][bi, pi] = target.clamp(0, 1)
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
