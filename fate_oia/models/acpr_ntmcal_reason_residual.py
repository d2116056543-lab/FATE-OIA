from __future__ import annotations

import torch
from torch import nn


class NativeTextReasonResidual(nn.Module):
    def __init__(self, dim: int = 384, reason_dim: int = 21, cap_max: float = 0.18) -> None:
        super().__init__()
        self.reason_dim = reason_dim
        self.cap_max = float(cap_max)
        self.net = nn.Sequential(nn.LayerNorm(dim + 4), nn.Linear(dim + 4, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1))

    def cap_for_epoch(self, epoch: int) -> float:
        if epoch < 3:
            return 0.0
        if epoch < 7:
            return min(self.cap_max, 0.05 + 0.07 * ((epoch - 3) / 3.0))
        return self.cap_max

    def forward(self, base_reason_logits: torch.Tensor, reason_nodes: torch.Tensor, support_score: torch.Tensor, contra_score: torch.Tensor, reason_rho: torch.Tensor, theta_reason_global: torch.Tensor | None = None, epoch: int = 0) -> dict[str, torch.Tensor | dict]:
        cap = self.cap_for_epoch(epoch)
        feat = torch.cat([reason_nodes, support_score.detach().unsqueeze(-1), contra_score.detach().unsqueeze(-1), reason_rho.detach().unsqueeze(-1), base_reason_logits.detach().unsqueeze(-1)], dim=-1)
        delta = torch.tanh(self.net(feat).squeeze(-1)) * cap
        return {"reason_delta": delta, "reason_delta_stats": {"reason_delta_abs_mean": float(delta.abs().mean().detach().cpu()), "reason_delta_cap": cap}}
