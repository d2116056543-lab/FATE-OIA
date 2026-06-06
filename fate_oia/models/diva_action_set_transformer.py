from __future__ import annotations

import torch
from torch import nn


class ActionSetPrototypeBank(nn.Module):
    def __init__(self, dim: int, residual_scale: float = 0.05) -> None:
        super().__init__()
        proto = torch.tensor([
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 1],
            [1, 0, 1, 0],
            [1, 0, 1, 1],
            [0, 1, 0, 1],
            [0, 1, 1, 0],
            [1, 1, 0, 0],
        ], dtype=torch.float32)
        self.register_buffer("prototype_vectors", proto)
        self.proj = nn.Linear(4, dim)
        self.residual = nn.Parameter(torch.zeros(proto.shape[0], dim))
        self.residual_scale = float(residual_scale)

    def forward(self) -> torch.Tensor:
        base = self.proj(self.prototype_vectors)
        residual = torch.clamp(self.residual, -self.residual_scale, self.residual_scale)
        return base + residual


class ActionSetRelationTransformer(nn.Module):
    def __init__(self, dim: int = 384, action_dim: int = 4, depth: int = 2, num_heads: int = 4) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.prototype_bank = ActionSetPrototypeBank(dim)
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, dim_feedforward=dim * 2, dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.action_score = nn.Linear(dim, 1)

    def forward(self, action_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        b = action_tokens.shape[0]
        proto = self.prototype_bank().unsqueeze(0).expand(b, -1, -1)
        seq = torch.cat([action_tokens, proto], dim=1)
        encoded = self.encoder(seq)
        action_out = encoded[:, : self.action_dim]
        z_eva = self.action_score(action_out).squeeze(-1)
        return {"action_tokens": action_out, "prototype_tokens": encoded[:, self.action_dim :], "z_eva": z_eva}
