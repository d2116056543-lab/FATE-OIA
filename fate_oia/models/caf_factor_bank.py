from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class CAFFactorBank(nn.Module):
    def __init__(self, dim: int = 384, action_dim: int = 4, factors_per_action: int = 3, object_factors: int = 8, scene_state_dim: int = 6) -> None:
        super().__init__()
        self.dim = int(dim)
        self.action_dim = int(action_dim)
        self.object_factors = int(object_factors)
        self.scene_proj = nn.Linear(scene_state_dim, dim)
        self.type_head = nn.Linear(dim, 7)

    def _affinity_region_factors(self, dino_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, c, h, w = dino_map.shape
        tokens = dino_map.flatten(2).transpose(1, 2)
        saliency = tokens.std(dim=-1) + 0.05 * tokens.mean(dim=-1).abs()
        k = min(self.object_factors, tokens.shape[1])
        seed_idx = torch.topk(saliency, k=k, dim=-1).indices
        factors = []
        regions = []
        grid_y = seed_idx // w
        grid_x = seed_idx % w
        for i in range(k):
            y = grid_y[:, i]
            x = grid_x[:, i]
            pooled = []
            boxes = []
            for bi in range(b):
                y0 = int(max(0, y[bi].item() - 1)); y1 = int(min(h, y[bi].item() + 2))
                x0 = int(max(0, x[bi].item() - 1)); x1 = int(min(w, x[bi].item() + 2))
                patch = dino_map[bi, :, y0:y1, x0:x1].flatten(1).transpose(0, 1)
                center = dino_map[bi, :, y[bi], x[bi]].unsqueeze(0)
                affinity = F.cosine_similarity(patch, center, dim=-1).softmax(dim=0)
                pooled.append((patch * affinity.unsqueeze(-1)).sum(0))
                boxes.append(torch.tensor([x0 / w, y0 / h, x1 / w, y1 / h], dtype=dino_map.dtype, device=dino_map.device))
            factors.append(torch.stack(pooled, dim=0))
            regions.append(torch.stack(boxes, dim=0))
        return torch.stack(factors, dim=1), torch.stack(regions, dim=1)

    def forward(self, actor_evidence_tokens: torch.Tensor, dino_map: torch.Tensor, scene_state_proxy: torch.Tensor | None = None, train_mode: bool = False) -> dict[str, torch.Tensor]:
        b, a, m, d = actor_evidence_tokens.shape
        actor_factors = actor_evidence_tokens.reshape(b, a * m, d)
        actor_groups = torch.arange(m, device=actor_factors.device).repeat(a).view(1, -1).expand(b, -1)
        actor_regions = torch.zeros(b, a * m, 4, dtype=dino_map.dtype, device=dino_map.device)
        actor_origin = torch.zeros(b, a * m, self.action_dim, dtype=dino_map.dtype, device=dino_map.device)
        for ai in range(a):
            actor_origin[:, ai * m:(ai + 1) * m, ai] = 1.0

        object_factors, object_regions = self._affinity_region_factors(dino_map)
        object_groups = torch.full((b, object_factors.shape[1]), 5, dtype=torch.long, device=dino_map.device)
        object_origin = torch.zeros(b, object_factors.shape[1], self.action_dim, dtype=dino_map.dtype, device=dino_map.device)

        factors = [actor_factors, object_factors]
        groups = [actor_groups, object_groups]
        regions = [actor_regions, object_regions]
        origins = [actor_origin, object_origin]
        source = [torch.zeros_like(actor_groups), torch.ones_like(object_groups)]
        if scene_state_proxy is not None and train_mode:
            scene = self.scene_proj(scene_state_proxy.float()).unsqueeze(1)
            factors.append(scene)
            groups.append(torch.full((b, 1), 6, dtype=torch.long, device=dino_map.device))
            regions.append(torch.zeros(b, 1, 4, dtype=dino_map.dtype, device=dino_map.device))
            origins.append(torch.zeros(b, 1, self.action_dim, dtype=dino_map.dtype, device=dino_map.device))
            source.append(torch.full((b, 1), 2, dtype=torch.long, device=dino_map.device))
        factor_tokens = torch.cat(factors, dim=1)
        factor_group_ids = torch.cat(groups, dim=1)
        factor_region = torch.cat(regions, dim=1)
        factor_action_origin = torch.cat(origins, dim=1)
        factor_source_id = torch.cat(source, dim=1)
        return {
            "factor_tokens": factor_tokens,
            "factor_group_ids": factor_group_ids,
            "factor_groups": factor_group_ids,
            "factor_region": factor_region,
            "factor_type_logits": self.type_head(factor_tokens),
            "factor_source_id": factor_source_id,
            "factor_available_mask": torch.ones(factor_tokens.shape[:2], dtype=torch.bool, device=factor_tokens.device),
            "factor_action_origin": factor_action_origin,
        }
