from __future__ import annotations

import torch
from torch import nn


def thresholds_to_bias(thresholds: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    thresholds = thresholds.float().clamp(eps, 1.0 - eps)
    return torch.log(thresholds / (1.0 - thresholds))


class PerLabelBiasCalibrator(nn.Module):
    """Small calibration head: calibrated_logit = base_logit - bias.

    A zero bias is an exact identity. Setting ``bias=logit(threshold)`` makes
    fixed 0.5 decisions equivalent to applying the supplied probability
    threshold on uncalibrated logits.
    """

    def __init__(
        self,
        num_labels: int,
        *,
        tail_indices: list[int] | None = None,
        init_bias: torch.Tensor | None = None,
        train_tail_only: bool = False,
    ) -> None:
        super().__init__()
        self.num_labels = int(num_labels)
        bias = torch.zeros(self.num_labels, dtype=torch.float32) if init_bias is None else init_bias.float().clone()
        if bias.numel() != self.num_labels:
            raise ValueError(f"init_bias must have {self.num_labels} values, got {bias.numel()}")
        self.bias = nn.Parameter(bias.view(self.num_labels))
        mask = torch.ones(self.num_labels, dtype=torch.float32)
        if train_tail_only:
            mask.zero_()
            if not tail_indices:
                raise ValueError("tail_indices are required when train_tail_only=True")
            mask[torch.tensor(tail_indices, dtype=torch.long)] = 1.0
        self.register_buffer("train_mask", mask, persistent=True)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.shape[-1] != self.num_labels:
            raise ValueError(f"Expected {self.num_labels} labels, got {logits.shape[-1]}")
        effective_bias = self.bias * self.train_mask.to(dtype=logits.dtype)
        return logits - effective_bias.view(1, -1)

    def clamp_non_tail_(self) -> None:
        with torch.no_grad():
            self.bias.mul_(self.train_mask.to(dtype=self.bias.dtype))
