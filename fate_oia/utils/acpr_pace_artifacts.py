from __future__ import annotations

import json
from pathlib import Path

import torch


def write_pace_json(path: str | Path, payload: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def save_compact_contrib(path: str | Path, tensor: torch.Tensor, max_cases: int | None = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    t = tensor.detach().cpu()
    if max_cases is not None:
        t = t[: int(max_cases)]
    torch.save(t.to(torch.float16), p)
