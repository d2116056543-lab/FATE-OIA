from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_meter_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    required = ("experiment", "data", "backbone", "model", "training", "runtime", "splits")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing METER config sections: {missing}")
    if config["runtime"]["test_only"] is not True or config["runtime"]["no_feature_cache"] is not True:
        raise ValueError("METER formal runtime must be test-only and cache-free")
    if config["model"]["token_compression"] != "none":
        raise ValueError("METER forbids token compression")
    if config["backbone"]["selected_layers"] != [3, 7, 11]:
        raise ValueError("METER requires DINO layers 3,7,11")
    return config
