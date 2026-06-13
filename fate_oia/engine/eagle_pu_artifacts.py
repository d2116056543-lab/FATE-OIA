from __future__ import annotations

import json
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
    if isinstance(obj, (list, tuple)):
        return [json_safe(x) for x in obj]
    try:
        import numpy as np
        if isinstance(obj, np.generic):
            return obj.item()
    except Exception:
        pass
    return obj


def write_json(path: str | Path, data: Any) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(json_safe(data), indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(path: str | Path, row: Any) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(json_safe(row), ensure_ascii=False) + "\n")


def save_tensor(path: str | Path, tensor: torch.Tensor) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor.detach().cpu(), p)
