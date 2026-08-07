from __future__ import annotations

import torch
from torch import Tensor, nn


def spatial_soft_iou(left: Tensor, right: Tensor) -> Tensor:
    left = left / left.amax(-1, keepdim=True).clamp_min(1e-8)
    right = right / right.amax(-1, keepdim=True).clamp_min(1e-8)
    return torch.minimum(left, right).sum(-1) / torch.maximum(left, right).sum(-1).clamp_min(1e-8)


class AIECertNaming(nn.Module):
    def __init__(self, dim=384, key_dim=64, confidence_threshold=0.45, margin_threshold=0.08):
        super().__init__()
        self.projection = nn.Linear(dim, key_dim)
        self.readout_scale = nn.Parameter(torch.ones(()))
        self.confidence_threshold = float(confidence_threshold)
        self.margin_threshold = float(margin_threshold)

    def forward(self, atom_token: Tensor, atom_map: Tensor, shared_keys: Tensor,
                predicate_attention: Tensor, predicate_probs: Tensor, certificate: Tensor | None = None,
                reliability: Tensor | None = None) -> dict[str, Tensor]:
        token, amap, keys = atom_token.detach(), atom_map.detach(), shared_keys.detach()
        pattn, pprob = predicate_attention.detach(), predicate_probs.detach()
        overlap = spatial_soft_iou(amap[..., None, :], pattn[:, None, None])
        compatibility = torch.sigmoid(self.readout_scale * torch.einsum("bakd,pd->bakp", self.projection(token), keys))
        quality = overlap * compatibility * pprob[:, None, None]
        if certificate is not None:
            effect = torch.sigmoid(certificate.detach())
            if reliability is not None:
                effect = effect * reliability.detach()
            quality = quality * effect[..., None]
        top = quality.topk(2, -1)
        confidence, index = top.values[..., 0], top.indices[..., 0]
        margin = top.values[..., 0] - top.values[..., 1]
        valid = (confidence >= self.confidence_threshold) & (margin >= self.margin_threshold)
        name_id = torch.where(valid, index, torch.full_like(index, -1))
        return {"name_id": name_id, "name_confidence": confidence, "name_margin": margin,
                "name_quality": quality, "named_coverage": valid.float().mean()}
