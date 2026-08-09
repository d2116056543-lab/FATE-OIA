from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch


def json_safe(value):
    if torch.is_tensor(value):
        return value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(path: str | Path, payload: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(target)


def append_jsonl(path: str | Path, payload: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_safe(payload), ensure_ascii=False, sort_keys=True) + "\n")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


EPOCH_ARTIFACTS = (
    "metrics_summary.json", "branch_metrics.json", "per_label_metrics.json",
    "role_gradient_stats.json", "pareto_license_stats.json", "predicate_agreement_stats.json",
    "action_rank_stats.json", "reason_rank_coverage.json", "counterfactual_summary.json",
    "paired_bootstrap.json", "test_outputs.pt",
)


def validate_epoch_artifacts(directory: str | Path) -> list[str]:
    root = Path(directory)
    return [name for name in EPOCH_ARTIFACTS if not (root / name).exists()]
