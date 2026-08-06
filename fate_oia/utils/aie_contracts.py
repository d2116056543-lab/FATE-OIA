from __future__ import annotations

from pathlib import Path

import torch


FORBIDDEN_PATTERNS = (
    "ACPRPairMemory", "ACPRActionComboAux", "ACPRCalibrationHead", "ACPRThresholdHead",
    "feature_cache_enabled: true", "token_compression: keep_merge", "best_selection_split: val",
    "Start-Process", "Start-Job", "nohup", "FrozenRunC", "cached_logits",
)


def scan_forbidden(paths: list[str | Path]) -> dict[str, list[str]]:
    findings: dict[str, list[str]] = {}
    for value in paths:
        path = Path(value)
        text = path.read_text(encoding="utf-8", errors="ignore")
        hits = [pattern for pattern in FORBIDDEN_PATTERNS if pattern in text]
        if hits:
            findings[str(path)] = hits
    return findings


def gradient_norm(parameters) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().square().sum())
    return total ** 0.5


def assert_finite_output(output: dict, keys: tuple[str, ...]) -> None:
    for key in keys:
        value = output[key]
        if not torch.is_tensor(value) or not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"AIE output {key} is missing or non-finite")


