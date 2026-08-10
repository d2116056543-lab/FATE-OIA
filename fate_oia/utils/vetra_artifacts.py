from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


REQUIRED_PROBE_ARTIFACTS = (
    "VETRA_SOURCE_REPLAY.json", "VETRA_SPLIT_OVERLAP_AUDIT.json",
    "DICE_CF_SEMANTICS_REPAIR.json", "VETRA_IMPLEMENTATION_AUDIT.json",
    "VETRA_RUNTIME_PROFILE.json", "VETRA_MECHANISM_SCREEN.json",
    "VETRA_METRICS_SUMMARY.jsonl", "VETRA_PER_ACTION_METRICS.jsonl",
    "VETRA_ROUTE_STATS.jsonl", "VETRA_ABLATION_METRICS.json",
    "VETRA_COUNTERFACTUAL_AUDIT.json", "VETRA_PAIRED_BOOTSTRAP.json",
)


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: str | Path, value: Any) -> None:
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_probe_artifacts(root: str | Path) -> list[str]:
    root = Path(root)
    return [name for name in REQUIRED_PROBE_ARTIFACTS if not (root / name).is_file()]
