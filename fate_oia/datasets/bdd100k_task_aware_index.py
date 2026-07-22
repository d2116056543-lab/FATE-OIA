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
    return Path(name).stem


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return json.load(handle)


class BDD100KTaskAwareIndex:
    """Preparsed BDD100K metadata index that never writes visual feature caches."""

    def __init__(self, root: str | Path, manifest_path: str | Path | None = None, emit_manifest: bool = True) -> None:
        self.root = Path(root)
        self._records: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        self._metadata: dict[str, dict[str, list[Any]]] = defaultdict(lambda: defaultdict(list))
        self._paths: dict[str, list[str]] = defaultdict(list)
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
        for path in self.root.rglob("*"):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix not in {".json", ".png", ".jpg", ".jpeg"}:
                continue
            payload: Any | None = _read_json(path) if suffix == ".json" else None
            kind = self._source_kind(path, payload)
            if suffix != ".json" and kind not in {"drivable", "semantic"}:
                continue
            self._paths[kind].append(str(path))
            for stem in self._file_stems(path, payload):
                self._records[stem][kind].append(str(path))
                if payload is not None:
                    self._metadata[stem][kind].append(payload)

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
        hashes = {item: hashlib.sha256(Path(item).read_bytes()).hexdigest() for values in self._paths.values() for item in values}
        payload = {
            "root": str(self.root),
            "source_counts": {kind: len(values) for kind, values in self._paths.items()},
            "paths": {kind: sorted(values) for kind, values in self._paths.items()},
            "duplicates": {stem: {kind: values for kind, values in sources.items() if len(values) > 1} for stem, sources in self._records.items()},
            "missing": {kind: sum(not bool(sources.get(kind)) for sources in self._records.values()) for kind in ("detection", "lane", "drivable", "semantic")},
            "hashes": hashes,
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
