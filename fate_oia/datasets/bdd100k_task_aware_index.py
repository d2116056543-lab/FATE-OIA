from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TaskAwareGroundingRecord:
    detection_jsons: tuple[str, ...]
    lane_jsons: tuple[str, ...]
    drivable_maps: tuple[str, ...]
    semantic_maps: tuple[str, ...]
    source_complete: dict[str, bool]


def _stem(name: str) -> str:
    stem = Path(name).stem
    # BDD100K color drivable maps use <image>_drivable_color.png.
    return stem.removesuffix("_drivable_color").removesuffix("_drivable_id")


def _read_json(path: Path) -> Any | None:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None


class BDD100KTaskAwareIndex:
    """Preparsed BDD100K metadata index that never writes visual feature caches."""

    def __init__(self, root: str | Path, manifest_path: str | Path | None = None, emit_manifest: bool = True) -> None:
        self.root = Path(root)
        self._records: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        self._metadata: dict[str, dict[str, list[Any]]] = defaultdict(lambda: defaultdict(list))
        self._paths: dict[str, list[str]] = defaultdict(list)
        self.invalid_json_paths: list[str] = []
        self._scan()
        self._freeze()
        if emit_manifest or manifest_path is not None:
            target = Path(manifest_path) if manifest_path is not None else self.root / "grounding_source_manifest.json"
            self.write_manifest(target)

    def _source_kind(self, path: Path, payload: Any | None = None) -> str:
        lower = str(path).lower().replace("\\", "/")
        if "drivable" in lower:
            return "drivable"
        if "semantic" in lower:
            return "semantic"
        if "lane" in lower or "poly2d" in lower:
            return "lane"
        if isinstance(payload, dict) and ("lanes" in payload or "poly2d" in payload):
            return "lane"
        return "detection"

    def _file_stems(self, path: Path, payload: Any | None) -> list[str]:
        if isinstance(payload, dict):
            candidates = [payload.get("name"), payload.get("file_name"), payload.get("filename")]
            frames = payload.get("frames")
            if isinstance(frames, list):
                candidates.extend(frame.get("name") or frame.get("file_name") for frame in frames if isinstance(frame, dict))
            found = [_stem(str(item)) for item in candidates if item]
            if found:
                return sorted(set(found))
        if isinstance(payload, list):
            found = [_stem(str(item.get("name") or item.get("file_name") or item.get("filename"))) for item in payload if isinstance(item, dict) and (item.get("name") or item.get("file_name") or item.get("filename"))]
            if found:
                return sorted(set(found))
        return [_stem(path.name)]

    def _scan(self) -> None:
        if not self.root.exists():
            return
        # Do not walk RGB image trees: they are not annotation sources and
        # make startup scale with every BDD100K image rather than its labels.
        known_roots = [self.root / name for name in ("bdd100k_labels", "bdd100k_drivable_maps", "bdd100k_seg") if (self.root / name).exists()]
        scan_roots = known_roots or [self.root]
        for scan_root in scan_roots:
            for path in scan_root.rglob("*"):
                if not path.is_file():
                    continue
                suffix = path.suffix.lower()
                if suffix not in {".json", ".png", ".jpg", ".jpeg"}:
                    continue
                payload: Any | None = _read_json(path) if suffix == ".json" else None
                if suffix == ".json" and payload is None:
                    self.invalid_json_paths.append(str(path))
                    continue
                kind = self._source_kind(path, payload)
                if suffix != ".json" and kind not in {"drivable", "semantic"}:
                    continue
                kinds = [kind]
                # The public 100k JSON contains detection boxes and lane/poly2d
                # records together.  Keep the two semantic sources separately
                # while retaining the original file in both source lists.
                if suffix == ".json" and self._contains_poly2d(payload):
                    kinds.append("lane")
                self._paths[kind].append(str(path))
                for stem in self._file_stems(path, payload):
                    for source in set(kinds):
                        self._records[stem][source].append(str(path))
                        if payload is not None:
                            self._metadata[stem][source].append(payload)

    @staticmethod
    def _contains_poly2d(payload: Any) -> bool:
        if not isinstance(payload, dict):
            return False
        frames = payload.get("frames", [])
        return any(
            isinstance(item, dict) and item.get("poly2d")
            for frame in frames if isinstance(frame, dict)
            for item in frame.get("objects", frame.get("labels", []))
        )

    def _freeze(self) -> None:
        for stem, sources in self._records.items():
            for kind in list(sources):
                sources[kind] = sorted(set(sources[kind]))

    def get(self, file_name: str) -> TaskAwareGroundingRecord:
        sources = self._records.get(_stem(file_name), {})
        return TaskAwareGroundingRecord(
            tuple(sources.get("detection", ())),
            tuple(sources.get("lane", ())),
            tuple(sources.get("drivable", ())),
            tuple(sources.get("semantic", ())),
            {kind: bool(sources.get(kind)) for kind in ("detection", "lane", "drivable", "semantic")},
        )

    def metadata_for(self, file_name: str) -> dict[str, list[Any]]:
        return self._metadata.get(_stem(file_name), {})

    def write_manifest(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Full hashing every dense PNG serializes startup behind many GB of IO.
        # A content hash remains exact for JSON annotation sources; maps use a
        # deterministic metadata fingerprint and are reopened only when a
        # sample's target is built.
        hashes = {}
        for values in self._paths.values():
            for item in values:
                source = Path(item)
                stat = source.stat()
                if source.suffix.lower() == ".json":
                    hashes[item] = hashlib.sha256(source.read_bytes()).hexdigest()
                else:
                    hashes[item] = hashlib.sha256(f"{source}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8")).hexdigest()
        payload = {
            "root": str(self.root),
            "source_counts": {kind: len(values) for kind, values in self._paths.items()},
            "paths": {kind: sorted(values) for kind, values in self._paths.items()},
            "duplicates": {stem: {kind: values for kind, values in sources.items() if len(values) > 1} for stem, sources in self._records.items()},
            "missing": {kind: sum(not bool(sources.get(kind)) for sources in self._records.values()) for kind in ("detection", "lane", "drivable", "semantic")},
            "hashes": hashes,
            "invalid_json_paths": sorted(self.invalid_json_paths),
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
