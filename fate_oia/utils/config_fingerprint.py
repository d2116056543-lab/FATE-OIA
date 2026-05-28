from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _to_plain(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_plain(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    return value


def canonical_config_json(config: dict[str, Any]) -> str:
    return json.dumps(_to_plain(config), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def config_fingerprint(config: dict[str, Any]) -> dict[str, Any]:
    canonical = canonical_config_json(config)
    return {
        "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "canonical_json": canonical,
    }


def _flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, child in sorted(value.items(), key=lambda item: str(item[0])):
            child_key = f"{prefix}.{key}" if prefix else str(key)
            _flatten(child_key, child, out)
    else:
        out[prefix] = _to_plain(value)


def diff_configs(left: dict[str, Any], right: dict[str, Any]) -> dict[str, dict[str, Any]]:
    left_flat: dict[str, Any] = {}
    right_flat: dict[str, Any] = {}
    _flatten("", left, left_flat)
    _flatten("", right, right_flat)
    keys = sorted(set(left_flat) | set(right_flat))
    added = {k: right_flat[k] for k in keys if k not in left_flat}
    removed = {k: left_flat[k] for k in keys if k not in right_flat}
    changed = {
        k: {"left": left_flat[k], "right": right_flat[k]}
        for k in keys
        if k in left_flat and k in right_flat and left_flat[k] != right_flat[k]
    }
    return {"added": added, "removed": removed, "changed": changed}


def write_fingerprint(path: str | Path, config: dict[str, Any], *, diff_against: dict[str, Any] | None = None) -> dict[str, Any]:
    record = {"fingerprint": config_fingerprint(config), "config": _to_plain(config)}
    if diff_against is not None:
        record["diff"] = diff_configs(diff_against, config)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record
