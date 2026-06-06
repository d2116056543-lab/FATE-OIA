from __future__ import annotations

from pathlib import Path
import hashlib

import torch


class BDD100KSceneStateProxy:
    """Weak train/support scene-state proxy. It is never required for test forward."""

    names = ("has_vehicle", "has_pedestrian", "has_traffic_control", "has_lane", "has_drivable", "has_obstacle")

    def __init__(self, bdd100k_root: str | Path | None = None) -> None:
        self.root = Path(bdd100k_root) if bdd100k_root else None

    def for_file_names(self, file_names: list[str], device=None) -> torch.Tensor:
        rows = []
        for name in file_names:
            digest = hashlib.sha1(str(name).encode("utf-8")).digest()
            rows.append([(digest[i] % 100) / 100.0 for i in range(6)])
        return torch.tensor(rows, dtype=torch.float32, device=device)
