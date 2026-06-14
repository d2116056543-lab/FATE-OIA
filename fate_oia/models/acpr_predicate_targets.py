from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import yaml

from fate_oia.datasets.bdd100k_grounding import BDD100KGroundingIndex


class WeakPredicateTargetBuilder:
    def __init__(self, scene_config: str | Path, bdd100k_root: str | Path | None = None) -> None:
        data = yaml.safe_load(Path(scene_config).read_text(encoding="utf-8")) or {}
        self.predicates = list(data.get("predicates", []))
        if len(self.predicates) < 32:
            raise ValueError("ACPR requires at least 32 scene predicates")
        self.names = [str(p["name"]) for p in self.predicates]
        self.index = BDD100KGroundingIndex(bdd100k_root) if bdd100k_root else None

    def _labels_for_file(self, file_name: str) -> set[str]:
        if self.index is None:
            return set()
        paths = self.index.lookup(file_name)
        if not paths.label_json:
            return set()
        try:
            data = json.loads(Path(paths.label_json).read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            return set()
        labels = set()
        for row in data.get("frames", data.get("labels", [])):
            entries = row.get("labels", []) if isinstance(row, dict) else []
            for item in entries:
                cat = str(item.get("category", "")).lower()
                if cat:
                    labels.add(cat)
        for item in data.get("labels", []):
            cat = str(item.get("category", "")).lower()
            if cat:
                labels.add(cat)
        return labels

    def build(self, file_names: list[str], device: torch.device | None = None) -> dict[str, torch.Tensor | dict[str, int]]:
        b = len(file_names)
        m = len(self.predicates)
        target = torch.zeros(b, m, dtype=torch.float32, device=device)
        mask = torch.zeros(b, m, dtype=torch.float32, device=device)
        source_counts = {"label_json": 0, "heuristic": 0, "missing": 0}
        for i, fn in enumerate(file_names):
            labels = self._labels_for_file(fn)
            if labels:
                source_counts["label_json"] += 1
            else:
                source_counts["missing"] += 1
            for j, pred in enumerate(self.predicates):
                sources = [str(x).lower() for x in pred.get("bdd100k_sources", [])]
                if labels and any(src in labels for src in sources):
                    target[i, j] = 1.0
                    mask[i, j] = 1.0
                elif labels:
                    mask[i, j] = 1.0
        return {"predicate_targets": target, "predicate_mask": mask, "source_counts": source_counts}
