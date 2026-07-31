from __future__ import annotations

import torch
from torch import Tensor, nn


class _ZeroInitLowRankResidual(nn.Module):
    def __init__(self, dim: int, rank: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.up.weight)

    def forward(self, value: Tensor) -> Tensor:
        return self.up(torch.nn.functional.gelu(self.down(self.norm(value))))


class HECASharedPrivateAdapters(nn.Module):
    """Zero-effect shared/action/reason adapters after label self-attention."""

    def __init__(
        self,
        dim: int = 384,
        action_dim: int = 4,
        reason_dim: int = 21,
        rank: int = 16,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.reason_dim = int(reason_dim)
        self.shared_adapter = _ZeroInitLowRankResidual(dim, rank)
        self.action_private_adapter = _ZeroInitLowRankResidual(dim, rank)
        self.reason_private_adapter = _ZeroInitLowRankResidual(dim, rank)

    def forward(self, label_nodes: Tensor) -> dict[str, Tensor]:
        if label_nodes.shape[1] != self.action_dim + self.reason_dim:
            raise ValueError("HECA adapters require action+reason label nodes")
        shared = label_nodes + self.shared_adapter(label_nodes)
        action = shared[:, : self.action_dim]
        reason = shared[:, self.action_dim :]
        return {
            "shared_nodes": shared,
            "action_nodes": action + self.action_private_adapter(action),
            "reason_nodes": reason + self.reason_private_adapter(reason),
        }


class METERFactorMetaAdapters(nn.Module):
    """Per-factor low-rank adapters; only these expose a reason gradient bridge."""

    def __init__(self, factor_dim: int = 21, dim: int = 384, rank: int = 16) -> None:
        super().__init__()
        self.factor_dim = int(factor_dim)
        self.dim = int(dim)
        self.rank = int(rank)
        self.down = nn.Parameter(torch.empty(factor_dim, rank, dim))
        self.up = nn.Parameter(torch.zeros(factor_dim, dim, rank))
        nn.init.xavier_uniform_(self.down)

    def forward(self, core_tokens: Tensor, parameter_override: dict[str, Tensor] | None = None) -> Tensor:
        if core_tokens.ndim != 3 or core_tokens.shape[1:] != (self.factor_dim, self.dim):
            raise ValueError("Meta adapters require [B,factor_dim,dim] core tokens")
        parameter_override = parameter_override or {}
        down = parameter_override.get("down", self.down)
        up = parameter_override.get("up", self.up)
        if down.shape != self.down.shape or up.shape != self.up.shape:
            raise ValueError("Functional meta adapter override has an invalid shape")
        hidden = torch.einsum("bfd,frd->bfr", core_tokens.detach(), down)
        return torch.einsum("bfr,fdr->bfd", torch.nn.functional.gelu(hidden), up)

    def parameter_grad_norm(self) -> float:
        total = torch.zeros((), device=self.down.device)
        for parameter in self.parameters():
            if parameter.grad is not None:
                total = total + parameter.grad.detach().float().square().sum()
        return float(total.sqrt().cpu())
