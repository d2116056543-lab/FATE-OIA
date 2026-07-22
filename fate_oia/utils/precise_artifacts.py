from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import yaml


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: str | Path, value: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")


def append_jsonl(path: str | Path, value: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, default=str) + "\n")


def write_resolved_config(path: str | Path, config: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def save_epoch_tensors(epoch_dir: str | Path, output: dict[str, Any], action: torch.Tensor, reason: torch.Tensor) -> None:
    target = Path(epoch_dir)
    target.mkdir(parents=True, exist_ok=True)
    for name in ("action_logits_direct", "action_logits_final_raw", "action_logits_deploy", "reason_logits_direct", "reason_logits_semantic", "reason_logits_observed", "reason_logits_deploy"):
        torch.save(output[name].detach().cpu(), target / f"logits_{name}.pt")
    torch.save(action.detach().cpu(), target / "labels_action.pt")
    torch.save(reason.detach().cpu(), target / "labels_reason.pt")
