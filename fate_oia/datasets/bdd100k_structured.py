from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


def bdd_oia_base_stem(file_name: str) -> str:
    stem = Path(str(file_name).replace("\\", "/")).stem
    return re.sub(r"_[0-9]+$", "", stem)


def _safe_read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _strip_drivable_suffix(stem: str) -> str:
    for suffix in ("_drivable_color", "_drivable_id", "_color", "_id"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _norm_poly2d(poly2d: Any) -> list[dict[str, Any]]:
    """Normalize BDD100K poly2d variants without discarding raw metadata."""
    if not poly2d:
        return []
    polys = poly2d if isinstance(poly2d, list) else [poly2d]
    out: list[dict[str, Any]] = []
    for poly in polys:
        vertices: list[list[float]] = []
        closed = True
        types = []
        raw = poly
        if isinstance(poly, dict):
            closed = bool(poly.get("closed", True))
            types = list(poly.get("types") or [])
            raw_vertices = poly.get("vertices") or poly.get("points") or poly.get("poly2d") or []
        else:
            raw_vertices = poly
        if isinstance(raw_vertices, list):
            for v in raw_vertices:
                if isinstance(v, dict):
                    x = v.get("x")
                    y = v.get("y")
                    if x is not None and y is not None:
                        vertices.append([float(x), float(y)])
                elif isinstance(v, (list, tuple)) and len(v) >= 2:
                    try:
                        vertices.append([float(v[0]), float(v[1])])
                        if len(v) >= 3:
                            types.append(v[2])
                    except (TypeError, ValueError):
                        continue
        if vertices:
            out.append({"vertices": vertices, "closed": closed, "types": types, "raw": raw})
    return out


@dataclass
class StructuredObject:
    category: str
    box2d: dict[str, float] | None
    poly2d: list[dict[str, Any]]
    attributes: dict[str, Any]
    source_type: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BDD100KStructuredRecord:
    file_name: str
    base_stem: str
    source_split: str | None
    label_json_path: str | None
    drivable_map_path: str | None
    semantic_seg_path: str | None
    attributes: dict[str, Any]
    objects: list[dict[str, Any]]
    lanes: list[dict[str, Any]]
    drivable: dict[str, Any] | None
    counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BDD100KStructuredIndex:
    def __init__(self, bdd100k_root: str | Path, allow_cross_split_fallback: bool = True) -> None:
        self.root = Path(bdd100k_root)
        self.allow_cross_split_fallback = allow_cross_split_fallback
        self.label_map: dict[str, tuple[Path, str]] = {}
        self.drivable_map: dict[str, tuple[Path, str]] = {}
        self.semantic_map: dict[str, tuple[Path, str]] = {}
        self._index()

    def _infer_split(self, path: Path) -> str:
        parts = {p.lower() for p in path.parts}
        for split in ("train", "val", "test"):
            if split in parts:
                return split
        return "unknown"

    def _index(self) -> None:
        label_root = self.root / "bdd100k_labels"
        if label_root.exists():
            for p in label_root.rglob("*.json"):
                self.label_map.setdefault(p.stem, (p, self._infer_split(p)))
        drive_root = self.root / "bdd100k_drivable_maps"
        if drive_root.exists():
            for p in drive_root.rglob("*.png"):
                self.drivable_map.setdefault(_strip_drivable_suffix(p.stem), (p, self._infer_split(p)))
        seg_root = self.root / "bdd100k_seg"
        if seg_root.exists():
            for p in seg_root.rglob("*.png"):
                self.semantic_map.setdefault(_strip_drivable_suffix(p.stem), (p, self._infer_split(p)))

    def _lookup(self, mapping: dict[str, tuple[Path, str]], base: str, split: str) -> tuple[Path, str] | None:
        hit = mapping.get(base)
        if hit is None:
            return None
        if self.allow_cross_split_fallback or hit[1] in (split, "unknown"):
            return hit
        return None

    def resolve(self, file_name: str, split: str = "train") -> BDD100KStructuredRecord | None:
        base = bdd_oia_base_stem(file_name)
        label_hit = self._lookup(self.label_map, base, split)
        drive_hit = self._lookup(self.drivable_map, base, split)
        seg_hit = self._lookup(self.semantic_map, base, split)
        if not label_hit and not drive_hit:
            return None
        attributes: dict[str, Any] = {}
        objects: list[dict[str, Any]] = []
        lanes: list[dict[str, Any]] = []
        source_split = label_hit[1] if label_hit else (drive_hit[1] if drive_hit else None)
        if label_hit:
            data = _safe_read_json(label_hit[0])
            attributes = dict(data.get("attributes") or {})
            frames = data.get("frames") or []
            labels: list[dict[str, Any]] = []
            if frames:
                first = frames[0] or {}
                labels.extend(first.get("labels") or [])
                labels.extend(first.get("objects") or [])
            labels.extend(data.get("labels") or [])
            for lab in labels:
                cat = str(lab.get("category") or lab.get("type") or "unknown")
                poly = _norm_poly2d(lab.get("poly2d"))
                box = lab.get("box2d")
                attrs = dict(lab.get("attributes") or {})
                source_type = "lane" if cat.startswith("lane/") or "lane" in cat else "object"
                rec = StructuredObject(category=cat, box2d=box if isinstance(box, dict) else None, poly2d=poly, attributes=attrs, source_type=source_type).to_dict()
                if source_type == "lane" or poly:
                    lanes.append(rec)
                else:
                    objects.append(rec)
        drivable = None
        if drive_hit:
            drivable = {"path": str(drive_hit[0]), "source_split": drive_hit[1], "category": "drivable"}
        return BDD100KStructuredRecord(
            file_name=file_name,
            base_stem=base,
            source_split=source_split,
            label_json_path=str(label_hit[0]) if label_hit else None,
            drivable_map_path=str(drive_hit[0]) if drive_hit else None,
            semantic_seg_path=str(seg_hit[0]) if seg_hit else None,
            attributes=attributes,
            objects=objects,
            lanes=lanes,
            drivable=drivable,
            counts={
                "object_count": len(objects),
                "lane_count": len(lanes),
                "drivable_count": 1 if drive_hit else 0,
                "attribute_count": len(attributes),
                "semantic_count": 1 if seg_hit else 0,
            },
        )

    def audit_split(self, file_names: list[str], split: str = "train", sample_limit: int | None = None) -> dict[str, Any]:
        names = file_names[:sample_limit] if sample_limit else file_names
        matched = 0
        object_count = lane_count = drivable_count = attribute_count = semantic_count = 0
        source_splits: dict[str, int] = {}
        examples: list[dict[str, Any]] = []
        for fn in names:
            rec = self.resolve(fn, split)
            if rec is None:
                continue
            matched += 1
            c = rec.counts
            object_count += int(c.get("object_count", 0))
            lane_count += int(c.get("lane_count", 0))
            drivable_count += int(c.get("drivable_count", 0))
            attribute_count += int(c.get("attribute_count", 0))
            semantic_count += int(c.get("semantic_count", 0))
            source_splits[str(rec.source_split)] = source_splits.get(str(rec.source_split), 0) + 1
            if len(examples) < 5:
                examples.append({"file_name": fn, "base_stem": rec.base_stem, "counts": rec.counts, "source_split": rec.source_split})
        total = len(names)
        return {
            "split": split,
            "sample_count": total,
            "matched_count": matched,
            "missing_count": total - matched,
            "match_rate": matched / total if total else 0.0,
            "object_count": object_count,
            "lane_count": lane_count,
            "drivable_count": drivable_count,
            "attribute_count": attribute_count,
            "semantic_count": semantic_count,
            "source_splits": source_splits,
            "examples": examples,
        }
