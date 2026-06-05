from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from fate_oia.models.egcaf_factor_types import FactorBatch


class DinoObjectLikeFactorGenerator(nn.Module):
    """DINO affinity-region object-like factors, not top-norm factors.

    Seeds are selected from salient tokens, then each factor expands into an
    affinity region by cosine similarity to the seed embedding.
    """

    def __init__(self, hidden_dim: int = 256, num_factors: int = 8, num_types: int = 11) -> None:
        super().__init__()
        self.num_factors = int(num_factors)
        self.type_head = nn.Linear(hidden_dim, num_types)

    def forward(self, p1: torch.Tensor) -> FactorBatch:
        b, c, h, w = p1.shape
        tokens = p1.flatten(2).transpose(1, 2)
        token_norm = tokens.norm(dim=-1)
        k = min(self.num_factors, tokens.shape[1])
        _, seed_idx = torch.topk(token_norm, k=k, dim=1)
        seed_tokens = torch.gather(tokens, 1, seed_idx.unsqueeze(-1).expand(-1, -1, c))
        norm_tokens = F.normalize(tokens, dim=-1)
        norm_seed = F.normalize(seed_tokens, dim=-1)
        affinity = torch.bmm(norm_seed, norm_tokens.transpose(1, 2))
        # Region expansion: keep high-affinity neighborhood, then soft-pool.
        threshold = affinity.mean(-1, keepdim=True) + 0.35 * affinity.std(-1, keepdim=True)
        region = torch.relu(affinity - threshold)
        weights = region / (region.sum(-1, keepdim=True) + 1e-6)
        emb = torch.bmm(weights, tokens)
        masks = weights.reshape(b, k, h, w)
        yy_grid = torch.linspace(0, 1, h, device=p1.device).view(1, 1, h, 1)
        xx_grid = torch.linspace(0, 1, w, device=p1.device).view(1, 1, 1, w)
        mass = masks.sum((-1, -2), keepdim=True) + 1e-6
        cx = (masks * xx_grid).sum((-1, -2), keepdim=True) / mass
        cy = (masks * yy_grid).sum((-1, -2), keepdim=True) / mass
        sx = torch.sqrt(((xx_grid - cx) ** 2 * masks).sum((-1, -2), keepdim=True) / mass).squeeze(-1).squeeze(-1)
        sy = torch.sqrt(((yy_grid - cy) ** 2 * masks).sum((-1, -2), keepdim=True) / mass).squeeze(-1).squeeze(-1)
        cx = cx.squeeze(-1).squeeze(-1)
        cy = cy.squeeze(-1).squeeze(-1)
        boxes = torch.stack([(cx - 2 * sx).clamp(0, 1), (cy - 2 * sy).clamp(0, 1), (cx + 2 * sx).clamp(0, 1), (cy + 2 * sy).clamp(0, 1)], -1)
        src = torch.ones(b, k, dtype=torch.long, device=p1.device)
        typ = self.type_head(emb)
        typ[..., 6] += 1.0
        rel = torch.sigmoid(region.sum(-1) / (region.sum(-1).detach().mean(dim=1, keepdim=True) + 1e-6))
        valid = masks.sum((-1, -2)) > 1e-6
        return FactorBatch(
            emb,
            masks,
            boxes,
            src,
            typ,
            rel,
            valid,
            {
                "generator": "dino_affinity_region",
                "region_method": "seed_affinity_expansion",
                "not_top_norm_only": True,
                "uses_bdd100k_gt": False,
            },
            torch.full((b, k), -1, dtype=torch.long, device=p1.device),
        )

