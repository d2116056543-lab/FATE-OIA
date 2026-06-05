from __future__ import annotations

import torch
from torch import nn

from fate_oia.models.egcaf_factor_types import FactorBatch


class DinoObjectLikeFactorGenerator(nn.Module):
    def __init__(self, hidden_dim: int = 256, num_factors: int = 8, num_types: int = 11) -> None:
        super().__init__()
        self.num_factors = num_factors
        self.type_head = nn.Linear(hidden_dim, num_types)

    def forward(self, p1: torch.Tensor) -> FactorBatch:
        b, c, h, w = p1.shape
        tokens = p1.flatten(2).transpose(1, 2)
        norm = tokens.norm(dim=-1)
        vals, idx = torch.topk(norm, k=min(self.num_factors, tokens.shape[1]), dim=1)
        emb = torch.gather(tokens, 1, idx.unsqueeze(-1).expand(-1, -1, c))
        yy = (idx // w).float() / max(h - 1, 1)
        xx = (idx % w).float() / max(w - 1, 1)
        grid_y = torch.linspace(0, 1, h, device=p1.device).view(1, 1, h, 1)
        grid_x = torch.linspace(0, 1, w, device=p1.device).view(1, 1, 1, w)
        masks = torch.exp(-((grid_x - xx.unsqueeze(-1).unsqueeze(-1)) ** 2 + (grid_y - yy.unsqueeze(-1).unsqueeze(-1)) ** 2) / 0.01)
        boxes = torch.stack([(xx - 0.12).clamp(0, 1), (yy - 0.12).clamp(0, 1), (xx + 0.12).clamp(0, 1), (yy + 0.12).clamp(0, 1)], -1)
        src = torch.ones(b, emb.shape[1], dtype=torch.long, device=p1.device)
        typ = self.type_head(emb)
        typ[..., 6] += 1.0
        rel = torch.sigmoid(vals / (vals.detach().mean(dim=1, keepdim=True) + 1e-6))
        valid = torch.ones(b, emb.shape[1], dtype=torch.bool, device=p1.device)
        return FactorBatch(emb, masks, boxes, src, typ, rel, valid, {"generator": "dino_object_like", "uses_bdd100k_gt": False})
