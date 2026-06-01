from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


STEM_SUFFIX_RE = re.compile(r"_(?:1|2|3)(?=\.[^.]+$|$)")


def bdd_oia_to_bdd100k_stem(file_name: str) -> str:
    stem = Path(str(file_name)).stem
    return STEM_SUFFIX_RE.sub("", stem)


def _safe_load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {}
    return data


def _poly_vertices(poly: Any) -> list[tuple[float, float]]:
    if isinstance(poly, list):
        pts: list[tuple[float, float]] = []
        for item in poly:
            if isinstance(item, dict) and "x" in item and "y" in item:
                pts.append((float(item["x"]), float(item["y"])))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                pts.append((float(item[0]), float(item[1])))
        return pts
    if not isinstance(poly, dict):
        return []
    raw = poly.get("vertices") or poly.get("poly2d") or poly.get("points") or []
    pts: list[tuple[float, float]] = []
    for item in raw:
        if isinstance(item, dict) and "x" in item and "y" in item:
            pts.append((float(item["x"]), float(item["y"])))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            pts.append((float(item[0]), float(item[1])))
    return pts


@dataclass
class StructuredBDD100KRecord:
    file_name: str
    base_stem: str
    split: str
    label_path: str | None = None
    drivable_path: str | None = None
    semantic_path: str | None = None
    source_split: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    objects: list[dict[str, Any]] = field(default_factory=list)
    lanes: list[dict[str, Any]] = field(default_factory=list)
    drivable: dict[str, Any] = field(default_factory=dict)
    unknown_poly: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_name": self.file_name,
            "base_stem": self.base_stem,
            "split": self.split,
            "label_path": self.label_path,
            "drivable_path": self.drivable_path,
            "semantic_path": self.semantic_path,
            "source_split": self.source_split,
            "attributes": self.attributes,
            "objects": self.objects,
            "lanes": self.lanes,
            "drivable": self.drivable,
            "unknown_poly": self.unknown_poly,
            "warnings": self.warnings,
        }

    @property
    def box_count(self) -> int:
        return sum(1 for obj in self.objects if obj.get("box2d"))

    @property
    def poly_count(self) -> int:
        return len(self.lanes) + len(self.unknown_poly)

    @property
    def has_drivable(self) -> bool:
        return bool(self.drivable_path or self.drivable.get("has_map"))


class BDD100KStructuredIndex:
    """Lazy BDD100K structured evidence index.

    The index only stores path maps up front. Per-sample label JSON is parsed on
    demand, which keeps startup cheap and avoids feature-cache style artifacts.
    """

    def __init__(self, root: str | Path, split_aliases: dict[str, str] | None = None) -> None:
        self.root = Path(root)
        self.split_aliases = split_aliases or {"val": "val", "test": "val", "train": "train"}
        self.label_maps: dict[str, dict[str, Path]] = {}
        self.drivable_maps: dict[str, dict[str, Path]] = {}
        self.semantic_maps: dict[str, dict[str, Path]] = {}
        for split in {"train", "val", "test"}:
            bdd_split = self.split_aliases.get(split, split)
            self.label_maps[split] = self._index_files(
                [
                    self.root / "bdd100k_labels" / "bdd100k" / "labels" / "100k" / bdd_split,
                    self.root / "bdd100k_info" / "bdd100k" / "info" / "100k" / bdd_split,
                ],
                "*.json",
            )
            self.drivable_maps[split] = self._index_files(
                [self.root / "bdd100k_drivable_maps" / "bdd100k" / "drivable_maps" / "color_labels" / bdd_split],
                "*_drivable_color.png",
            )
            self.semantic_maps[split] = self._index_files(
                [self.root / "bdd100k_seg" / "bdd100k" / "seg" / "color_labels" / bdd_split],
                "*_train_color.png",
            )

    @staticmethod
    def _index_files(dirs: Iterable[Path], pattern: str) -> dict[str, Path]:
        out: dict[str, Path] = {}
        for directory in dirs:
            if not directory.exists():
                continue
            for path in directory.glob(pattern):
                stem = path.stem
                stem = stem.replace("_drivable_color", "").replace("_train_color", "")
                out.setdefault(stem, path)
        return out

    def lookup(self, file_name: str, split: str) -> StructuredBDD100KRecord:
        split = split if split in self.label_maps else "train"
        base = bdd_oia_to_bdd100k_stem(file_name)
        record = StructuredBDD100KRecord(file_name=file_name, base_stem=base, split=split)
        search_order = [split] + [s for s in ("train", "val", "test") if s != split]
        label_path = None
        drivable_path = None
        semantic_path = None
        source_split = None
        for candidate in search_order:
            if label_path is None and base in self.label_maps.get(candidate, {}):
                label_path = self.label_maps[candidate][base]
                source_split = candidate
            if drivable_path is None and base in self.drivable_maps.get(candidate, {}):
                drivable_path = self.drivable_maps[candidate][base]
            if semantic_path is None and base in self.semantic_maps.get(candidate, {}):
                semantic_path = self.semantic_maps[candidate][base]
        if label_path is not None:
            record.label_path = str(label_path)
            record.source_split = source_split
            self._fill_from_label_json(record, label_path)
        else:
            record.warnings.append("missing_label_json")
        if drivable_path is not None:
            record.drivable_path = str(drivable_path)
            record.drivable = {"has_map": True, "source": "bdd100k_drivable_map"}
        if semantic_path is not None:
            record.semantic_path = str(semantic_path)
        return record

    def _fill_from_label_json(self, record: StructuredBDD100KRecord, path: Path) -> None:
        data = _safe_load_json(path)
        record.attributes = data.get("attributes", {}) if isinstance(data.get("attributes"), dict) else {}
        labels: list[dict[str, Any]] = []
        if isinstance(data.get("frames"), list) and data["frames"]:
            frame0 = data["frames"][0]
            if isinstance(frame0, dict):
                labels = frame0.get("objects") or frame0.get("labels") or []
        if not labels:
            labels = data.get("labels") or data.get("objects") or []
        if not isinstance(labels, list):
            labels = []
        for raw in labels:
            if not isinstance(raw, dict):
                continue
            category = str(raw.get("category", "")).lower()
            box = raw.get("box2d")
            poly_raw = raw.get("poly2d")
            if box and isinstance(box, dict):
                record.objects.append(
                    {
                        "category": category,
                        "box2d": {
                            "x1": float(box.get("x1", box.get("x", 0.0))),
                            "y1": float(box.get("y1", box.get("y", 0.0))),
                            "x2": float(box.get("x2", box.get("x", 0.0))),
                            "y2": float(box.get("y2", box.get("y", 0.0))),
                        },
                    }
                )
            if isinstance(poly_raw, list):
                poly_items = poly_raw
                if poly_raw and isinstance(poly_raw[0], (list, tuple)) and len(poly_raw[0]) >= 2:
                    poly_items = [poly_raw]
                for poly in poly_items:
                    verts = _poly_vertices(poly)
                    if not verts:
                        record.warnings.append(f"unparsed_poly2d:{category or 'unknown'}")
                        continue
                    item = {"category": category, "vertices": verts, "closed": bool(poly.get("closed", True)) if isinstance(poly, dict) else True}
                    if any(k in category for k in ["lane", "crosswalk", "curb", "road", "area", "drivable"]):
                        record.lanes.append(item)
                    else:
                        record.unknown_poly.append(item)
        if any("drivable" in item.get("category", "") for item in record.lanes):
            record.drivable.setdefault("has_poly", True)

    def audit_samples(self, file_names: list[str], split: str, max_examples: int = 8) -> dict[str, Any]:
        total = len(file_names)
        matched = 0
        box_count = 0
        poly_count = 0
        lane_count = 0
        drivable_count = 0
        examples: list[dict[str, Any]] = []
        for fn in file_names:
            rec = self.lookup(fn, split)
            if rec.label_path:
                matched += 1
            box_count += rec.box_count
            poly_count += rec.poly_count
            lane_count += len(rec.lanes)
            drivable_count += int(rec.has_drivable)
            if len(examples) < max_examples:
                examples.append(
                    {
                        "file_name": fn,
                        "base_stem": rec.base_stem,
                        "has_label": bool(rec.label_path),
                        "box_count": rec.box_count,
                        "lane_count": len(rec.lanes),
                        "drivable": rec.has_drivable,
                        "warnings": rec.warnings[:3],
                    }
                )
        return {
            "split": split,
            "sample_count": total,
            "matched_count": matched,
            "missing_count": total - matched,
            "match_rate": matched / max(total, 1),
            "box2d_count": box_count,
            "poly2d_count": poly_count,
            "lane_count": lane_count,
            "drivable_count": drivable_count,
            "examples": examples,
        }
