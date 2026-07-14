from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F


_PYRAMID_KEYS = ("F_hi", "F_mid", "F_ctx")


class _LowRankLocalResidual(nn.Module):
    def __init__(self, dim: int, rank: int, dropout: float) -> None:
        super().__init__()
        self.down = nn.Conv2d(dim, rank, kernel_size=1, bias=False)
        self.up = nn.Conv2d(rank, dim, kernel_size=1, bias=True)
        self.depthwise = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.dropout = nn.Dropout2d(dropout)
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        # Zero-output initialization preserves the source model exactly while
        # leaving up/depthwise gradients live on the first backward pass.
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        nn.init.zeros_(self.depthwise.weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        low_rank = self.up(self.dropout(F.gelu(self.down(value))))
        return low_rank + self.depthwise(value)


class MOSAICLowRankReZeroPyramidAdapter(nn.Module):
    """A zero-output, bounded low-rank local residual adapter for one visual lane."""

    def __init__(
        self,
        *,
        dim: int,
        rank: int = 48,
        dropout: float = 0.05,
        rezero_init: float = 0.0,
        rezero_max: float = 0.30,
        base_scale_fraction: float = 0.20,
        modulation_fraction: float = 0.10,
    ) -> None:
        super().__init__()
        if type(dim) is not int or dim <= 0 or type(rank) is not int or not 0 < rank <= dim:
            raise ValueError("dim and rank must be positive integers with rank <= dim")
        if not 0.0 <= dropout < 1.0 or not 0.0 <= rezero_init <= 1.0 or not 0.0 < rezero_max <= 1.0:
            raise ValueError("adapter dropout/ReZero values are outside their bounded contract")
        if not 0.0 < base_scale_fraction <= 1.0 or not 0.0 <= modulation_fraction < base_scale_fraction:
            raise ValueError("adapter scale fractions are invalid")
        self.dim = dim
        self.rank = rank
        self.rezero_max = float(rezero_max)
        self.base_scale_fraction = float(base_scale_fraction)
        self.modulation_fraction = float(modulation_fraction)
        self.blocks = nn.ModuleDict({name: _LowRankLocalResidual(dim, rank, dropout) for name in _PYRAMID_KEYS})
        self.rezero_raw = nn.Parameter(torch.full((_PYRAMID_KEYS.__len__(),), float(rezero_init)))

    @property
    def effective_scale(self) -> torch.Tensor:
        raw = torch.tanh(self.rezero_raw)
        fraction = self.base_scale_fraction + self.modulation_fraction * raw
        return self.rezero_max * fraction.clamp(min=0.0, max=1.0)

    def forward(self, pyramid: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if set(pyramid) != set(_PYRAMID_KEYS):
            raise ValueError(f"adapter requires exactly {_PYRAMID_KEYS}")
        output: dict[str, torch.Tensor] = {}
        for index, name in enumerate(_PYRAMID_KEYS):
            value = pyramid[name]
            if value.ndim != 4 or value.shape[1] != self.dim:
                raise ValueError(f"{name} must be [B,{self.dim},H,W]")
            output[name] = value + self.effective_scale[index] * self.blocks[name](value)
        return output
