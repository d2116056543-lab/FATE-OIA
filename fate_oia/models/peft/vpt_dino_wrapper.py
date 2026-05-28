from __future__ import annotations

import torch
from torch import nn


class ShallowVPT(nn.Module):
    """Prepend trainable prompt tokens to patch tokens before a head/encoder."""

    def __init__(self, dim: int = 384, prompt_len: int = 8) -> None:
        super().__init__()
        self.prompt = nn.Parameter(torch.empty(int(prompt_len), int(dim)))
        nn.init.trunc_normal_(self.prompt, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        prompt = self.prompt.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        return torch.cat([prompt.to(device=tokens.device, dtype=tokens.dtype), tokens], dim=1)
