from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn


@dataclass
class RunCCache:
    action_fused_logits: torch.Tensor
    action_visual_logits: torch.Tensor
    action_reason_logits: torch.Tensor
    reason_logits: torch.Tensor
    labels_action: torch.Tensor
    labels_reason: torch.Tensor
    file_names: list[str]
    run_dir: Path
    suffix: str

    @property
    def labels(self) -> torch.Tensor:
        return torch.cat([self.labels_action.float(), self.labels_reason.float()], dim=1)

    @property
    def logits(self) -> torch.Tensor:
        return torch.cat([self.action_fused_logits.float(), self.reason_logits.float()], dim=1)

    def subset(self, indices: torch.Tensor) -> "RunCCache":
        idx = indices.detach().cpu().long()
        return RunCCache(
            action_fused_logits=self.action_fused_logits[idx],
            action_visual_logits=self.action_visual_logits[idx],
            action_reason_logits=self.action_reason_logits[idx],
            reason_logits=self.reason_logits[idx],
            labels_action=self.labels_action[idx],
            labels_reason=self.labels_reason[idx],
            file_names=[self.file_names[int(i)] for i in idx.tolist()],
            run_dir=self.run_dir,
            suffix=self.suffix,
        )


def _load_tensor(path: Path) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(str(path))
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_names(path: Path, expected: int) -> list[str]:
    if path.exists():
        names = json.loads(path.read_text(encoding="utf-8"))
        if len(names) != expected:
            raise ValueError(f"{path} has {len(names)} names, expected {expected}")
        return [str(x) for x in names]
    return [f"sample_{i:06d}" for i in range(expected)]


def load_run_c_cache(run_dir: str | Path, suffix: str = "best_test") -> RunCCache:
    root = Path(run_dir)
    cache = RunCCache(
        action_fused_logits=_load_tensor(root / f"logits_action_fused_{suffix}.pt").float(),
        action_visual_logits=_load_tensor(root / f"logits_action_visual_{suffix}.pt").float(),
        action_reason_logits=_load_tensor(root / f"logits_action_reason_{suffix}.pt").float(),
        reason_logits=_load_tensor(root / f"logits_reason_{suffix}.pt").float(),
        labels_action=_load_tensor(root / f"labels_action_{suffix}.pt").float(),
        labels_reason=_load_tensor(root / f"labels_reason_{suffix}.pt").float(),
        file_names=[],
        run_dir=root,
        suffix=suffix,
    )
    n = int(cache.reason_logits.shape[0])
    cache.file_names = _load_names(root / f"file_names_{suffix}.json", n)
    for name, tensor in {
        "action_fused_logits": cache.action_fused_logits,
        "action_visual_logits": cache.action_visual_logits,
        "action_reason_logits": cache.action_reason_logits,
        "reason_logits": cache.reason_logits,
        "labels_action": cache.labels_action,
        "labels_reason": cache.labels_reason,
    }.items():
        if tensor.shape[0] != n:
            raise ValueError(f"{name} has {tensor.shape[0]} rows, expected {n}")
    return cache


class FrozenRunCPredictor(nn.Module):
    """Frozen wrapper around cached Run C logits.

    This intentionally does not re-run the image model. The tail-adapter track
    is a residual/calibration diagnostic on the preserved Run C outputs.
    """

    def __init__(self, cache: RunCCache) -> None:
        super().__init__()
        self.register_buffer("action_fused_logits", cache.action_fused_logits.float(), persistent=False)
        self.register_buffer("action_visual_logits", cache.action_visual_logits.float(), persistent=False)
        self.register_buffer("action_reason_logits", cache.action_reason_logits.float(), persistent=False)
        self.register_buffer("reason_logits", cache.reason_logits.float(), persistent=False)

    def forward(self, indices: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if indices is None:
            return {
                "action_logits": self.action_fused_logits,
                "action_visual_logits": self.action_visual_logits,
                "action_reason_logits": self.action_reason_logits,
                "reason_logits": self.reason_logits,
            }
        idx = indices.to(self.action_fused_logits.device).long()
        return {
            "action_logits": self.action_fused_logits[idx],
            "action_visual_logits": self.action_visual_logits[idx],
            "action_reason_logits": self.action_reason_logits[idx],
            "reason_logits": self.reason_logits[idx],
        }
