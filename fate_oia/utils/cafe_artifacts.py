from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from typing import Any

import torch


def json_safe(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return float(obj.detach().cpu())
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, set):
        return sorted(json_safe(x) for x in obj)
    if isinstance(obj, (list, tuple)):
        return [json_safe(x) for x in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Namespace):
        return json_safe(vars(obj))
    return obj


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(json_safe(obj), indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(json_safe(obj), ensure_ascii=False) + "\n")
