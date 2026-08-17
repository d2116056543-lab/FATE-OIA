from __future__ import annotations

from typing import Sequence, Union

import torch
from torch import Tensor, nn


ActionScale = Union[float, Tensor, Sequence[float]]


def direction_preserving_l2_cap(logits: Tensor, cap: float = 20.0) -> Tensor:
    norm = logits.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = torch.clamp(torch.as_tensor(cap, device=logits.device, dtype=logits.dtype) / norm, max=1.0)
    return logits * scale


class AIEContributionHead(nn.Module):
    def __init__(self, dim: int = 384, action_dim: int = 4, probes_per_action: int = 4, kappa: float = 3.0, logit_norm_cap: float = 20.0) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.probes_per_action = probes_per_action
        self.kappa = float(kappa)
        self.logit_norm_cap = float(logit_norm_cap)
        self.norm = nn.LayerNorm(dim)
        self.weight = nn.Parameter(torch.empty(action_dim, dim))
        self.bias = nn.Parameter(torch.zeros(action_dim, probes_per_action))
        nn.init.normal_(self.weight, std=1e-3)

    def _scale_row(self, action_scale: ActionScale, reference: Tensor) -> Tensor:
        scale = torch.as_tensor(action_scale, device=reference.device, dtype=reference.dtype)
        if scale.ndim == 0:
            scale = scale.expand(self.action_dim)
        if scale.shape != (self.action_dim,):
            raise ValueError(
                "action_scale must be a scalar or contain exactly four action values "
                f"(expected {self.action_dim}, got shape {tuple(scale.shape)})"
            )
        if not bool(torch.isfinite(scale).all()):
            raise ValueError("action_scale values must be finite")
        return scale.view(1, self.action_dim)

    def forward(
        self,
        evidence_token: Tensor,
        action_logits_primary: Tensor,
        *,
        action_scale: ActionScale = 1.0,
    ) -> dict[str, Tensor]:
        raw = torch.einsum("bakd,ad->bak", self.norm(evidence_token), self.weight) + self.bias[None]
        summed = raw.sum(-1)
        scale_row = self._scale_row(action_scale, summed)
        bounded_delta = scale_row * self.kappa * torch.tanh(summed / self.kappa)
        uncapped_final = action_logits_primary + bounded_delta
        final = direction_preserving_l2_cap(uncapped_final, self.logit_norm_cap)
        exact_delta = final - action_logits_primary
        ratio = torch.where(
            summed.abs() > 1e-7,
            exact_delta / summed,
            scale_row.expand_as(summed),
        )
        bounded_contribution = torch.where(
            (summed.abs() > 1e-7)[..., None],
            raw * ratio[..., None],
            exact_delta[..., None] / float(self.probes_per_action),
        )
        final_train = direction_preserving_l2_cap(action_logits_primary.detach() + bounded_delta, self.logit_norm_cap)
        reconstruction = (final - action_logits_primary - bounded_contribution.sum(-1)).abs().max()
        return {
            "action_logits_primary": action_logits_primary,
            "action_logits_final": final,
            "action_logits_final_train": final_train,
            "raw_contribution": raw,
            "bounded_contribution": bounded_contribution,
            "action_delta": exact_delta,
            "action_logits_final_uncapped": uncapped_final,
            "contribution_reconstruction_error": reconstruction,
        }
