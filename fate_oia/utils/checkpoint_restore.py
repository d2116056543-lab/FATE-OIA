from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_training_checkpoint(path: str | Path, **payload: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_training_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=device)
