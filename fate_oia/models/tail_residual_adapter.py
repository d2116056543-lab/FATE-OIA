from __future__ import annotations

import torch
from torch import nn


class TailResidualAdapter(nn.Module):
    """Residual adapter that can only modify selected weak reason labels."""

    def __init__(
        self,
        *,
        action_dim: int = 4,
        reason_dim: int = 21,
        tail_indices: list[int] | tuple[int, ...],
        hidden_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not tail_indices:
            raise ValueError("tail_indices must be non-empty")
        self.action_dim = int(action_dim)
        self.reason_dim = int(reason_dim)
        self.tail_indices = [int(i) for i in tail_indices]
        for idx in self.tail_indices:
            if idx < 0 or idx >= self.reason_dim:
                raise ValueError(f"tail index {idx} is outside reason_dim={self.reason_dim}")
        input_dim = self.action_dim + self.reason_dim
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(self.tail_indices)),
        )
        self.delta_scale = nn.Parameter(torch.zeros(len(self.tail_indices), dtype=torch.float32))
        mask = torch.zeros(self.reason_dim, dtype=torch.float32)
        mask[torch.tensor(self.tail_indices, dtype=torch.long)] = 1.0
        self.register_buffer("tail_mask", mask, persistent=True)

    def forward(self, action_logits: torch.Tensor, reason_logits: torch.Tensor) -> dict[str, torch.Tensor]:
        if action_logits.shape[-1] != self.action_dim:
            raise ValueError(f"Expected action logits dim {self.action_dim}, got {action_logits.shape[-1]}")
        if reason_logits.shape[-1] != self.reason_dim:
            raise ValueError(f"Expected reason logits dim {self.reason_dim}, got {reason_logits.shape[-1]}")
        features = torch.cat([action_logits.float(), reason_logits.float()], dim=-1)
        tail_delta = torch.tanh(self.net(features)) * self.delta_scale.to(features.dtype).view(1, -1)
        delta = reason_logits.new_zeros(reason_logits.shape)
        idx = torch.tensor(self.tail_indices, device=reason_logits.device, dtype=torch.long)
        delta.index_copy_(1, idx, tail_delta.to(reason_logits.dtype))
        return {
            "reason_logits": reason_logits + delta,
            "delta_reason_logits": delta,
            "tail_delta": tail_delta,
        }
