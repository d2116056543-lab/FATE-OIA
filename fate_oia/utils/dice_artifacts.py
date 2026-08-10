from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


REQUIRED_GATE_FILES = (
    "DICE_BASE_REPLAY.json", "DICE_IMPLEMENTATION_REVIEW.json", "DICE_ORACLE_POTENTIAL.json",
    "DICE_PROBE_METRICS.json", "DICE_PAIRED_BOOTSTRAP.json", "DICE_MECHANISM_GATES.json",
)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(path: str | Path, value: Any) -> None:
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def validate_gate_artifacts(root: str | Path) -> list[str]:
    root = Path(root)
    return [name for name in REQUIRED_GATE_FILES if not (root / name).is_file()]
