from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class AIEPredicateNaming(nn.Module):
    """Names evidence atoms without introducing a competing null class."""

    def __init__(
        self,
        dim: int = 384,
        num_predicates: int = 32,
        confidence_threshold: float = 0.45,
        margin_threshold: float = 0.08,
        presence_threshold: float = 0.30,
    ) -> None:
        super().__init__()
        self.predicate_keys = nn.Parameter(torch.randn(num_predicates, 64) * 0.02)
        self.evidence_projection = nn.Linear(dim, 64)
        self.confidence_threshold = confidence_threshold
        self.margin_threshold = margin_threshold
        self.presence_threshold = presence_threshold

    def forward(
        self,
        evidence_token: Tensor,
        evidence_map: Tensor,
        predicate_attention: Tensor,
        predicate_probs: Tensor,
        cf_effect: Tensor | None = None,
    ) -> dict[str, Tensor]:
        predicate_map = predicate_attention.detach().clamp_min(0)
        predicate_presence = predicate_probs.detach().clamp(0, 1)
        intersection = torch.einsum("bakn,bpn->bakp", evidence_map, predicate_map)
        union = evidence_map.sum(-1, keepdim=True) + predicate_map.sum(-1)[:, None, None, :] - intersection
        soft_iou = intersection / union.clamp_min(1e-8)
        compatibility = torch.sigmoid(
            torch.einsum("bakd,pd->bakp", self.evidence_projection(evidence_token), self.predicate_keys)
            / math.sqrt(self.predicate_keys.shape[-1])
        )
        quality = soft_iou * compatibility * predicate_presence[:, None, None, :]
        if cf_effect is not None:
            quality = quality * torch.sigmoid(cf_effect.detach())[..., None]
        top = quality.topk(k=2, dim=-1)
        confidence = top.values[..., 0]
        margin = top.values[..., 0] - top.values[..., 1]
        top_id = top.indices[..., 0]
        top_presence = predicate_presence.gather(1, top_id.reshape(top_id.shape[0], -1)).reshape_as(top_id)
        valid = (
            (confidence >= self.confidence_threshold)
            & (margin >= self.margin_threshold)
            & (top_presence >= self.presence_threshold)
        )
        name_id = torch.where(valid, top_id, torch.full_like(top_id, -1))
        return {
            "name_id": name_id,
            "name_confidence": confidence,
            "name_margin": margin,
            "name_quality": quality,
            "name_spatial_soft_iou": soft_iou,
            "name_compatibility": compatibility,
            "named_coverage": valid.float().mean(),
        }


