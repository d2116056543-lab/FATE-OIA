from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

import yaml


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _flatten(value: Any, prefix: str) -> dict[str, Any]:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            result.update(_flatten(item, f"{prefix}.{key}" if prefix else str(key)))
        return result
    if isinstance(value, list):
        result = {}
        for index, item in enumerate(value):
            result.update(_flatten(item, f"{prefix}[{index}]"))
        return result or {prefix: []}
    return {prefix: value}


def resolve_icdor_config_tree(paths: Iterable[str | Path]) -> dict[str, Any]:
    sources: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        name = path.name
        if name in sources:
            raise ValueError(f"duplicate IC-DOR config source {name}")
        sources[name] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        hashes[name] = _sha256(path)
    if len(sources) != 5:
        raise ValueError("IC-DOR resolved config tree requires exactly five YAML sources")
    return {"sources": sources, "source_sha256": hashes}


class ConfigUsageTracker:
    def __init__(self, resolved_tree: dict[str, Any]) -> None:
        self.resolved_tree = resolved_tree
        self._leaves: dict[str, Any] = {}
        for source, value in resolved_tree.get("sources", {}).items():
            self._leaves.update(_flatten(value, source))
        self.rows = {
            path: {
                "path": path,
                "status": "unused",
                "consumer_file": None,
                "consumer_symbol": None,
            }
            for path in self._leaves
        }

    @property
    def leaf_paths(self) -> tuple[str, ...]:
        return tuple(self._leaves)

    def consume(self, path: str, *, consumer_file: str, consumer_symbol: str) -> Any:
        if path not in self._leaves:
            raise KeyError(f"unknown resolved config path {path}")
        if not consumer_file or not consumer_symbol:
            raise ValueError("config consumers require file and symbol")
        self.rows[path] = {
            "path": path,
            "status": "consumed",
            "consumer_file": consumer_file,
            "consumer_symbol": consumer_symbol,
        }
        return self._leaves[path]

    def diagnostic_only(self, path: str, *, consumer_file: str, consumer_symbol: str) -> Any:
        value = self.consume(path, consumer_file=consumer_file, consumer_symbol=consumer_symbol)
        self.rows[path]["status"] = "diagnostic_only"
        return value

    def consume_source(self, source_name: str, *, consumer_file: str, consumer_symbol: str) -> None:
        prefix = source_name + "."
        matches = [path for path in self._leaves if path == source_name or path.startswith(prefix)]
        if not matches:
            raise KeyError(f"unknown resolved config source {source_name}")
        for path in matches:
            self.consume(path, consumer_file=consumer_file, consumer_symbol=consumer_symbol)

    def finalize(self, *, require_all_consumed: bool) -> dict[str, Any]:
        unused = sorted(path for path, row in self.rows.items() if row["status"] == "unused")
        payload = {
            "source_sha256": dict(self.resolved_tree.get("source_sha256", {})),
            "rows": [self.rows[path] for path in sorted(self.rows)],
            "unused_config_keys": unused,
        }
        if require_all_consumed and unused:
            raise ValueError(f"unused resolved config keys: {unused}")
        return payload
