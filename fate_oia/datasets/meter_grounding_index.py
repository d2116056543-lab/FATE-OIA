from __future__ import annotations

import json
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

from .bdd100k_grounding import BDD100KGroundingIndex, bdd_oia_base_stem
from .meter_typed_targets import METERTypedTargetBuilder


class METERGroundingIndex:
    """LRU metadata reader producing conservative train-only typed targets."""

    def __init__(
        self,
        bdd100k_root: str | Path,
        *,
        schema_path: str | Path,
        cache_size: int = 512,
    ) -> None:
        self.base = BDD100KGroundingIndex(bdd100k_root)
        self.builder = METERTypedTargetBuilder(schema_path)
        self.cache_size = int(cache_size)
        self._records: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.json_parse_count = 0
        self.json_parse_seconds = 0.0
        self.target_build_seconds = 0.0
        self.target_build_count = 0
        self.anchor_counts = [0] * len(self.builder.factors)
        self.state_counts = [0] * len(self.builder.factors)

    def _record(self, file_name: str) -> dict[str, Any]:
        stem = bdd_oia_base_stem(file_name)
        if stem in self._records:
            record = self._records.pop(stem)
            self._records[stem] = record
            return record
        paths = self.base.lookup(file_name)
        record: dict[str, Any] = {
            "source_complete": bool(paths.label_json),
            "objects": [],
            "lanes": [],
            "polylines": [],
            "lane_complete": False,
            "drivable_map_path": paths.drivable_map,
        }
        if paths.label_json:
            start = time.perf_counter()
            raw = json.loads(
                Path(paths.label_json).read_text(encoding="utf-8", errors="ignore")
            )
            self.json_parse_seconds += time.perf_counter() - start
            for frame in raw.get("frames", []) if isinstance(raw, dict) else []:
                record["objects"].extend(frame.get("objects", []) or [])
                record["objects"].extend(frame.get("labels", []) or [])
                record["lanes"].extend(frame.get("lanes", frame.get("lane", [])) or [])
                record["polylines"].extend(frame.get("polylines", []) or [])
                record["lane_complete"] = bool(
                    record["lane_complete"] or frame.get("lane_complete", False)
                )
            self.json_parse_count += 1
        self._records[stem] = record
        while len(self._records) > self.cache_size:
            self._records.popitem(last=False)
        return record

    def typed_target(self, file_name: str, *, split: str) -> dict[str, Any] | None:
        if split != "train":
            return None
        start = time.perf_counter()
        target = self.builder.build(self._record(file_name))
        self.target_build_seconds += time.perf_counter() - start
        self.target_build_count += 1
        self.anchor_counts = [
            left + int(right)
            for left, right in zip(
                self.anchor_counts, target["factor_anchor_valid"].tolist()
            )
        ]
        self.state_counts = [
            left + int(right)
            for left, right in zip(
                self.state_counts, target["factor_state_valid"].tolist()
            )
        ]
        return target

    # Backward-compatible name for dataset/regression callers.
    def signed_target(self, file_name: str, *, split: str) -> dict[str, Any] | None:
        return self.typed_target(file_name, split=split)

    def coverage(self, file_names: list[str]) -> dict[str, Any]:
        return {
            **self.base.audit_file_names(file_names),
            "json_parse_count": self.json_parse_count,
            "json_parse_seconds": self.json_parse_seconds,
            "target_build_seconds": self.target_build_seconds,
            "target_build_count": self.target_build_count,
            "lru_size": len(self._records),
            "anchor_counts": self.anchor_counts,
            "state_counts": self.state_counts,
        }
