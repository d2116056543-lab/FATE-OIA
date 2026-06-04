from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def json_safe(obj: Any) -> Any:
    try:
        import torch

        if torch.is_tensor(obj):
            if obj.ndim == 0:
                return obj.detach().cpu().item()
            return obj.detach().cpu().tolist()
    except Exception:
        pass
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(obj), indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: str | Path, rows: list[Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(json_safe(row), ensure_ascii=False) + "\n")
