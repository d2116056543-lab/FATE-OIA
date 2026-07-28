from __future__ import annotations

import torch
from torch import Tensor, nn


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
