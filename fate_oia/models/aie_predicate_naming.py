from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def spatial_soft_iou(left: Tensor, right: Tensor, eps: float = 1e-8) -> Tensor:
    """Soft IoU over above-background saliency rather than probability mass."""
    left_support = torch.relu(left - left.mean(dim=-1, keepdim=True))
    right_support = torch.relu(right - right.mean(dim=-1, keepdim=True))
    left_peak = left_support.amax(dim=-1, keepdim=True)
    right_peak = right_support.amax(dim=-1, keepdim=True)
    left_support = torch.where(left_peak > eps, left_support / left_peak.clamp_min(eps), torch.zeros_like(left_support))
    right_support = torch.where(right_peak > eps, right_support / right_peak.clamp_min(eps), torch.zeros_like(right_support))
    intersection = torch.minimum(left_support, right_support).sum(dim=-1)
    union = torch.maximum(left_support, right_support).sum(dim=-1)
    return torch.where(union > eps, intersection / union.clamp_min(eps), torch.zeros_like(union))


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
        soft_iou = spatial_soft_iou(evidence_map[..., None, :], predicate_map[:, None, None, :, :])
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

