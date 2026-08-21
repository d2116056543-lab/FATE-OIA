from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from contextlib import contextmanager
from collections import OrderedDict


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().item() if value.ndim == 0 else value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def atomic_write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, target)


def append_jsonl(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(json_safe(payload), ensure_ascii=False) + "\n")


def seed_tida_run(seed: int) -> None:
    """Seed every RNG used before constructing a fresh TIDA model."""
    value = int(seed)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda"):
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


COMMON_BINDING_FIELDS = (
    "pass", "git_head", "git_tree", "base_source_head", "base_source_tree", "config_sha256",
    "skill_sha256", "plan_sha256", "clip_manifest_sha256", "image_checkpoint_sha256", "commands", "gates", "created_at",
)


def validate_completion_artifact(payload: dict[str, Any], *, phase: str) -> list[str]:
    failures = [field for field in COMMON_BINDING_FIELDS if field not in payload]
    if phase == "full_train_ready":
        for field in (
            "design_review", "implementation_review", "mechanism_review", "memory_review",
            "golden_oracle_sha256",
        ):
            if field not in payload:
                failures.append(field)
    if payload.get("pass") is not True:
        failures.append("pass")
    return sorted(set(failures))


def save_checkpoint_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, target)


class TIDATrainableEMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.998) -> None:
        self.decay = float(decay)
        self.state = OrderedDict(
            (name, parameter.detach().clone()) for name, parameter in model.named_parameters() if parameter.requires_grad
        )

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        current = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
        if current.keys() != self.state.keys():
            raise RuntimeError("TIDA EMA owner keys changed")
        for name, parameter in current.items():
            self.state[name].mul_(self.decay).add_(parameter.detach(), alpha=1.0 - self.decay)

    def state_dict(self) -> OrderedDict[str, torch.Tensor]:
        return OrderedDict((name, value.detach().clone()) for name, value in self.state.items())

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        if state.keys() != self.state.keys():
            raise RuntimeError("TIDA EMA checkpoint keys changed")
        for name, value in state.items():
            self.state[name].copy_(value)

    @contextmanager
    @torch.no_grad()
    def average_parameters(self, model: torch.nn.Module):
        current = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
        backup = {name: parameter.detach().clone() for name, parameter in current.items()}
        for name, value in self.state.items():
            current[name].copy_(value)
        try:
            yield model
        finally:
            for name, value in backup.items():
                current[name].copy_(value)
