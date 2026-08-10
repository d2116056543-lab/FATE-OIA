from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path


def stable_split_ids(ids: list[str], seed: int = 20260806, calib_fraction: float = 0.10, audit_count: int = 1024) -> dict[str, list[str]]:
    ordered = list(ids)
    random.Random(seed).shuffle(ordered)
    calib_count = max(1, int(round(len(ordered) * calib_fraction))) if ordered else 0
    audit_start = calib_count
    audit_end = min(audit_start + audit_count, len(ordered))
    return {
        "train_calib": ordered[:calib_count],
        "train_audit": ordered[audit_start:audit_end],
        "train_main": ordered[audit_end:],
    }


def ids_sha256(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def write_split_manifest(path: str | Path, train_ids: list[str], seed: int, calib_fraction: float, audit_count: int) -> dict:
    splits = stable_split_ids(train_ids, seed, calib_fraction, audit_count)
    payload = {
        "seed": seed,
        "train_count": len(train_ids),
        "train_sha256": ids_sha256(train_ids),
        "train_calib_count": len(splits["train_calib"]),
        "train_calib_sha256": ids_sha256(splits["train_calib"]),
        "train_audit_count": len(splits["train_audit"]),
        "train_audit_sha256": ids_sha256(splits["train_audit"]),
        "train_main_count": len(splits["train_main"]),
        "train_main_sha256": ids_sha256(splits["train_main"]),
        "main_calib_overlap": len(set(splits["train_main"]) & set(splits["train_calib"])),
        "main_audit_overlap": len(set(splits["train_main"]) & set(splits["train_audit"])),
        "calib_audit_overlap": len(set(splits["train_calib"]) & set(splits["train_audit"])),
        "internal_engineering_protocol": True,
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
