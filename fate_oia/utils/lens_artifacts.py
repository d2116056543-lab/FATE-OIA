from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def json_safe(value: Any) -> Any:
    if torch.is_tensor(value): return value.detach().cpu().tolist()
    if isinstance(value, dict): return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [json_safe(v) for v in value]
    return value


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle: handle.write(json.dumps(json_safe(row), ensure_ascii=False) + "\n")


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")
