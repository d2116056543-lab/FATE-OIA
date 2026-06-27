from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_REQUIRED_TOP_LEVEL = {
    "paths",
    "data",
    "model",
    "loss",
    "optimization",
    "evaluation",
    "traffic_influence",
    "profile",
    "visualization",
    "supervisor",
}


def load_interactflow_config(path: str | Path) -> dict[str, Any]:
    cfg_path = Path(path)
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if "experiment" not in data and "experiment_name" not in data:
        raise ValueError("InteractFlow config missing experiment/experiment_name")
    missing = sorted(DEFAULT_REQUIRED_TOP_LEVEL.difference(data))
    if missing:
        raise ValueError(f"InteractFlow config missing top-level keys: {missing}")
    cfg = deepcopy(data)
    cfg["_config_path"] = str(cfg_path)
    validate_interactflow_config(cfg)
    return cfg


def validate_interactflow_config(cfg: dict[str, Any]) -> None:
    data = cfg.get("data", {})
    runtime = cfg.get("runtime", {})
    model = cfg.get("model", {})
    if int(data.get("observed_frames", 0)) != 15:
        raise ValueError("formal PSI protocol requires observed_frames=15")
    if bool(data.get("formal_input_uses_target_frame", True)):
        raise ValueError("formal input must not use target_frame image")
    if bool(runtime.get("feature_cache_enabled", False)) or bool(model.get("feature_cache_enabled", False)):
        raise ValueError("feature cache is forbidden for ACPR-InteractFlow++")
    if str(model.get("token_compression", "none")) != "none":
        raise ValueError("token compression must be none")
    evaluation = cfg.get("evaluation", {})
    splits = evaluation.get("eval_splits", ["test"])
    if splits != ["test"] and splits != "test":
        raise ValueError(f"test-only evaluation required, got {splits}")
    if not bool(evaluation.get("validation_loader_forbidden_formal", False)):
        raise ValueError("formal PSI protocol must forbid validation loader")


def resolve_path(value: str | Path, root: str | Path | None = None) -> Path:
    path = Path(value)
    if path.is_absolute() or root is None:
        return path
    return Path(root) / path
