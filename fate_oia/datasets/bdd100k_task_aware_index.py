"""In-memory, task-aware BDD100K metadata for RAEL train-time grounding.

The index only keeps annotation metadata.  It deliberately has no image, DINO,
feature-cache, or test-forward API.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Iterable, Mapping
from uuid import uuid4

from fate_oia.datasets.bdd100k_grounding import bdd_oia_base_stem


_SOURCE_NAMES = ("detections", "lanes", "drivable")
_GROUNDING_COVERAGE_FIELDS = (
    "matched_entity_count",
    "unmatched_positive_count",
    "reliable_negative_count",
    "unknown_count",
    "traffic_state_valid_count",
    "drivable_valid_count",
    "boundary_valid_count",
)
_SOURCE_ALIAS_POLICY = "union_all_exact_aliases_by_reduced_stem"


@dataclass(frozen=True)
class FrozenDict(Mapping[Any, Any]):
    """Pickle-safe immutable mapping used for train-worker grounding metadata."""

    _items: tuple[tuple[Any, Any], ...]

    def __init__(self, values: Mapping[Any, Any]) -> None:
        object.__setattr__(self, "_items", tuple(values.items()))

    def __getitem__(self, key: Any) -> Any:
        for item_key, item_value in self._items:
            if item_key == key:
                return item_value
        raise KeyError(key)

    def __iter__(self):
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Mapping) and dict(self._items) == dict(other.items())

    def __reduce__(self):
        return type(self), (dict(self._items),)


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, FrozenDict):
        return value
    if isinstance(value, Mapping):
        return FrozenDict({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class RAELGroundingRecord:
    """All train-time grounding metadata associated with one image stem."""

    detections: tuple[Mapping[str, Any], ...]
    lanes: tuple[Mapping[str, Any], ...]
    drivable: tuple[Mapping[str, Any], ...]
    source_complete: Mapping[str, bool]

    def __post_init__(self) -> None:
        object.__setattr__(self, "detections", tuple(_deep_freeze(item) for item in self.detections))
        object.__setattr__(self, "lanes", tuple(_deep_freeze(item) for item in self.lanes))
        object.__setattr__(self, "drivable", tuple(_deep_freeze(item) for item in self.drivable))
        object.__setattr__(self, "source_complete", _deep_freeze(dict(self.source_complete)))

    def mutable_copy(self) -> dict[str, Any]:
        """Return detached mutable metadata for transforms without exposing index state."""

        return {
            "detections": [_deep_thaw(item) for item in self.detections],
            "lanes": [_deep_thaw(item) for item in self.lanes],
            "drivable": [_deep_thaw(item) for item in self.drivable],
            "source_complete": _deep_thaw(self.source_complete),
        }


def _normalise_category(value: Any) -> str:
    return str(value or "unknown").strip().lower().replace(" ", "_").replace("-", "_")


def _candidate_stems(file_name: str) -> tuple[str, ...]:
    exact = Path(str(file_name).replace("\\", "/")).stem
    reduced = bdd_oia_base_stem(file_name)
    # Keep both positions explicit even when the values coincide: the source
    # alias contract distinguishes the exact lookup key from its reduced form.
    return exact, reduced


def _frame_name(frame: Mapping[str, Any]) -> str | None:
    for key in ("name", "file_name", "filename", "image"):
        value = frame.get(key)
        if value:
            return str(value)
    return None


def _rows(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, Mapping)]
    if not isinstance(payload, Mapping):
        return []
    for key in ("frames", "images", "records", "annotations"):
        value = payload.get(key)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, Mapping)]
    return []


def _box_from(value: Mapping[str, Any]) -> list[float] | None:
    raw = value.get("box") or value.get("bbox") or value.get("box2d")
    if isinstance(raw, Mapping):
        keys = ("x1", "y1", "x2", "y2")
        if all(key in raw for key in keys):
            box = [float(raw[key]) for key in keys]
            return box if _valid_box(box) else None
        if all(key in raw for key in ("x", "y", "w", "h")):
            x, y, w, h = (float(raw[key]) for key in ("x", "y", "w", "h"))
            box = [x, y, x + w, y + h]
            return box if _valid_box(box) else None
    if isinstance(raw, (list, tuple)) and len(raw) >= 4:
        box = [float(item) for item in raw[:4]]
        return box if _valid_box(box) else None
    return None


def _valid_box(box: Iterable[float]) -> bool:
    x1, y1, x2, y2 = tuple(box)
    return all(math.isfinite(value) for value in (x1, y1, x2, y2)) and x1 < x2 and y1 < y2


def _items(frame: Mapping[str, Any], source: str) -> Iterable[Mapping[str, Any]]:
    keys = {
        "detections": ("labels", "objects", "detections"),
        "lanes": ("lanes", "labels", "objects"),
        "drivable": ("drivable", "areas", "labels"),
    }[source]
    for key in keys:
        values = frame.get(key)
        if isinstance(values, list):
            if not values:
                continue
            for value in values:
                if isinstance(value, Mapping):
                    yield value
            return


def _normalise_detection(item: Mapping[str, Any]) -> dict[str, Any] | None:
    box = _box_from(item)
    if box is None:
        return None
    attributes = item.get("attributes")
    return {
        "category": _normalise_category(item.get("category") or item.get("type")),
        "box": box,
        "sector": _normalise_category(item.get("sector") or (attributes or {}).get("sector")),
        "attributes": dict(attributes) if isinstance(attributes, Mapping) else {},
    }


def _normalise_lane(item: Mapping[str, Any]) -> dict[str, Any] | None:
    points = item.get("points") or item.get("poly2d") or item.get("polygon")
    points = _normalise_poly2d_points(points)
    if points is None:
        return None
    attributes = item.get("attributes")
    return {
        "category": _normalise_category(item.get("category") or "lane"),
        "side": _normalise_category(item.get("side") or (attributes or {}).get("side")),
        "points": points,
        "attributes": dict(attributes) if isinstance(attributes, Mapping) else {},
    }


def _normalise_drivable(item: Mapping[str, Any]) -> dict[str, Any] | None:
    polygon = item.get("polygon") or item.get("points") or item.get("poly2d")
    polygon = _normalise_poly2d_points(polygon)
    if polygon is None:
        return None
    return {
        "category": _normalise_category(item.get("category") or "drivable"),
        "side": _normalise_category(item.get("side")),
        "polygon": polygon,
    }


def _normalise_poly2d_points(value: Any) -> list[list[float]] | None:
    """Convert BDD100K ``[x, y, type]`` poly2d values to finite XY points."""

    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    points: list[list[float]] = []
    for raw in value:
        if isinstance(raw, Mapping):
            x, y = raw.get("x"), raw.get("y")
        elif isinstance(raw, (list, tuple)) and len(raw) >= 2:
            x, y = raw[0], raw[1]
        else:
            return None
        try:
            point = [float(x), float(y)]
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(item) for item in point):
            return None
        points.append(point)
    return points


def _frame_objects_from_label_file(payload: Any, *, source_path: Path) -> Iterable[tuple[str, list[Mapping[str, Any]]]]:
    """Yield explicit BDD100K per-frame object lists from one label JSON.

    RAEL accepts only the documented ``{name, frames:[{objects:...}]}``
    layout here.  It deliberately does not recursively search a BDD100K root
    or infer alternative metadata formats.
    """

    if not isinstance(payload, Mapping):
        raise ValueError(f"BDD100K label file must be a mapping: {source_path}")
    name = payload.get("name")
    frames = payload.get("frames")
    if not isinstance(name, str) or not name or not isinstance(frames, list):
        raise ValueError(f"BDD100K label file must contain name and frames: {source_path}")
    for frame in frames:
        if not isinstance(frame, Mapping):
            continue
        objects = frame.get("objects")
        if not isinstance(objects, list):
            continue
        yield name, [item for item in objects if isinstance(item, Mapping)]


def _source_for_label_object(item: Mapping[str, Any]) -> str | None:
    category = _normalise_category(item.get("category") or item.get("type"))
    if category in {"area", "drivable"} or category.startswith("drivable") or category.startswith("area/drivable"):
        return "drivable"
    if category.startswith("lane") or category in {"road_boundary", "lane_marking"}:
        return "lanes"
    return "detections" if _box_from(item) is not None else None


class RAELTaskAwareBDD100KIndex:
    """Preload detection, lane, and drivable JSON once and merge by stem."""

    def __init__(
        self,
        detections: str | Path | None = None,
        lanes: str | Path | None = None,
        drivable: str | Path | None = None,
        label_directories: Mapping[str, str | Path] | None = None,
        include_file_names: Iterable[str] | None = None,
    ) -> None:
        self._by_source: dict[str, dict[str, list[dict[str, Any]]]] = {
            source: defaultdict(list) for source in _SOURCE_NAMES
        }
        self._present_stems: dict[str, set[str]] = {source: set() for source in _SOURCE_NAMES}
        self._reduced_aliases: dict[str, set[str]] = defaultdict(set)
        self._source_hashes: dict[str, str | None] = {source: None for source in _SOURCE_NAMES}
        self._parse_calls: dict[str, int] = {source: 0 for source in _SOURCE_NAMES}
        self._include_stems = (
            None
            if include_file_names is None
            else {_candidate_stems(str(name))[1] for name in include_file_names}
        )
        self._filtered_file_count = 0
        paths = {"detections": detections, "lanes": lanes, "drivable": drivable}
        for source, path in paths.items():
            if path is not None:
                self._load_source(source, Path(path))
        if label_directories is not None:
            self._load_label_directories(label_directories)
        self._records = self._merge_records()

    def _register_items(self, source: str, name: str, items: Iterable[Mapping[str, Any]]) -> None:
        exact_stem, reduced_stem = _candidate_stems(name)
        self._present_stems[source].add(reduced_stem)
        self._reduced_aliases[reduced_stem].add(exact_stem)
        normalise = {
            "detections": _normalise_detection,
            "lanes": _normalise_lane,
            "drivable": _normalise_drivable,
        }[source]
        for item in items:
            record = normalise(item)
            if record is not None:
                self._by_source[source][reduced_stem].append(record)

    def _load_label_directories(self, label_directories: Mapping[str, str | Path]) -> None:
        if set(label_directories).difference({"train", "val"}):
            raise ValueError("label_directories may contain only explicit train/val entries")
        if not label_directories:
            raise ValueError("label_directories must not be empty")
        hashes: list[bytes] = []
        for split, raw_path in sorted(label_directories.items()):
            directory = Path(raw_path)
            if not directory.is_dir():
                raise FileNotFoundError(f"BDD100K {split} label directory does not exist: {directory}")
            files = tuple(sorted(directory.glob("*.json")))
            if not files:
                raise ValueError(f"BDD100K {split} label directory has no direct JSON files: {directory}")
            for path in files:
                if self._include_stems is not None and _candidate_stems(path.name)[1] not in self._include_stems:
                    self._filtered_file_count += 1
                    continue
                payload_bytes = path.read_bytes()
                hashes.append(path.name.encode("utf-8") + b"\0" + hashlib.sha256(payload_bytes).digest())
                payload = json.loads(payload_bytes.decode("utf-8", errors="strict"))
                for name, objects in _frame_objects_from_label_file(payload, source_path=path):
                    grouped: dict[str, list[Mapping[str, Any]]] = {source: [] for source in _SOURCE_NAMES}
                    for item in objects:
                        source = _source_for_label_object(item)
                        if source is not None:
                            grouped[source].append(item)
                    # All three views originate in this explicit, complete
                    # object list; empty lists are known absence, not an
                    # unknown external annotation source.
                    for source in _SOURCE_NAMES:
                        self._register_items(source, name, grouped[source])
        directory_hash = hashlib.sha256(b"".join(hashes)).hexdigest()
        for source in _SOURCE_NAMES:
            previous = self._source_hashes[source]
            if previous is not None:
                raise ValueError("cannot mix aggregate BDD100K sources with label_directories")
            self._source_hashes[source] = directory_hash
            self._parse_calls[source] += 1

    def _load_source(self, source: str, path: Path) -> None:
        payload_bytes = path.read_bytes()
        self._source_hashes[source] = hashlib.sha256(payload_bytes).hexdigest()
        payload = json.loads(payload_bytes.decode("utf-8", errors="ignore"))
        self._parse_calls[source] += 1
        normalise = {
            "detections": _normalise_detection,
            "lanes": _normalise_lane,
            "drivable": _normalise_drivable,
        }[source]
        for frame in _rows(payload):
            name = _frame_name(frame)
            if not name:
                continue
            self._register_items(source, name, _items(frame, source))

    def _merge_records(self) -> dict[str, RAELGroundingRecord]:
        stems: set[str] = set()
        for values in self._present_stems.values():
            stems.update(values)
        return {
            stem: RAELGroundingRecord(
                detections=tuple(self._by_source["detections"].get(stem, ())),
                lanes=tuple(self._by_source["lanes"].get(stem, ())),
                drivable=tuple(self._by_source["drivable"].get(stem, ())),
                source_complete={source: stem in self._present_stems[source] for source in _SOURCE_NAMES},
            )
            for stem in sorted(stems)
        }

    def lookup(self, file_name: str) -> RAELGroundingRecord:
        _, reduced_stem = _candidate_stems(file_name)
        return self._records.get(
            reduced_stem,
            RAELGroundingRecord((), (), (), {source: False for source in _SOURCE_NAMES}),
        )

    def source_stem_aliases(self, file_name: str) -> tuple[str, ...]:
        """Return the explicit exact/reduced lookup contract for one source name."""

        return _candidate_stems(file_name)

    @staticmethod
    def _grounding_coverage_payload(
        grounding_coverage: Mapping[str, int] | None,
    ) -> dict[str, int | bool]:
        """Expose audited values only; absence is not evidence of zero targets."""

        if grounding_coverage is None:
            return {"grounding_available": False}
        missing = [field for field in _GROUNDING_COVERAGE_FIELDS if field not in grounding_coverage]
        if missing:
            raise ValueError(f"grounding coverage is incomplete: {missing}")
        invalid = [
            field for field in _GROUNDING_COVERAGE_FIELDS
            if isinstance(grounding_coverage[field], bool) or not isinstance(grounding_coverage[field], Integral)
        ]
        if invalid:
            raise TypeError(f"grounding coverage must use integer counts: {invalid}")
        values = {field: int(grounding_coverage[field]) for field in _GROUNDING_COVERAGE_FIELDS}
        if any(value < 0 for value in values.values()):
            raise ValueError("grounding coverage values must be non-negative")
        return {"grounding_available": True, **values}

    def manifest(self, *, grounding_coverage: Mapping[str, int] | None = None) -> dict[str, Any]:
        source_coverage = {
            source: len(stems) for source, stems in self._present_stems.items()
        }
        aliases = {
            reduced: sorted(stems)
            for reduced, stems in sorted(self._reduced_aliases.items())
        }
        ambiguous_aliases = {
            reduced: sorted(stems)
            for reduced, stems in sorted(self._reduced_aliases.items())
            if len(stems) > 1
        }
        payload = {
            "metadata_only": True,
            "feature_cache": False,
            "filtered_file_count": self._filtered_file_count,
            "parse_calls": dict(self._parse_calls),
            "source_hashes": dict(self._source_hashes),
            "source_alias_policy": _SOURCE_ALIAS_POLICY,
            "reduced_stem_policy": _SOURCE_ALIAS_POLICY,
            "reduced_stem_aliases": aliases,
            "ambiguous_reduced_stem_aliases": ambiguous_aliases,
            "coverage": {
                "stems": len(self._records),
                **source_coverage,
                **self._grounding_coverage_payload(grounding_coverage),
            },
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return {**payload, "manifest_hash": hashlib.sha256(canonical).hexdigest()}

    def write_manifest(
        self,
        path: str | Path,
        *,
        grounding_coverage: Mapping[str, int] | None = None,
    ) -> dict[str, Any]:
        """Write the stable metadata-only manifest used by P1 audit evidence."""

        manifest = self.manifest(grounding_coverage=grounding_coverage)
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        return manifest
