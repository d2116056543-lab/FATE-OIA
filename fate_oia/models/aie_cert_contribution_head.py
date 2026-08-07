from __future__ import annotations

import torch
from torch import Tensor, nn


def direction_preserving_l2_cap(delta: Tensor, cap: float = 20.0) -> Tensor:
    norm = delta.norm(dim=-1, keepdim=True)
    return delta * (cap / norm.clamp_min(cap))


def stable_signed_decomposition(raw: Tensor, delta: Tensor) -> Tensor:
    """Conserve delta exactly without dividing by a cancelling signed sum."""
    centered = raw - raw.mean(dim=-1, keepdim=True)
    contrast = centered / centered.abs().sum(dim=-1, keepdim=True).clamp_min(1e-6)
    base = delta[..., None] / raw.shape[-1]
    return base + 0.5 * delta.abs()[..., None] * contrast


class AIECertContributionHead(nn.Module):
    def __init__(self, dim: int = 384, action_dim: int = 4, kappa: float = 3.0, norm_cap: float = 20.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.weight = nn.Parameter(torch.empty(action_dim, dim))
        nn.init.normal_(self.weight, std=0.02)
        self.kappa = float(kappa)
        self.norm_cap = float(norm_cap)

    def forward(self, centered_atom_token: Tensor, primary_logits: Tensor, action_scale: float) -> dict[str, Tensor]:
        raw = torch.einsum("bakd,ad->bak", self.norm(centered_atom_token), self.weight)
        raw_delta = action_scale * self.kappa * torch.tanh(raw.sum(-1) / self.kappa)
        delta = direction_preserving_l2_cap(raw_delta, self.norm_cap)
        bounded = stable_signed_decomposition(raw, delta)
        final = primary_logits + bounded.sum(-1)
        error = (final - primary_logits - bounded.sum(-1)).abs().amax()
        return {
            "raw_contribution": raw,
            "bounded_contribution": bounded,
            "action_delta": delta,
            "action_logits_final": final,
            "action_logits_final_train": primary_logits.detach() + bounded.sum(-1),
            "contribution_reconstruction_error": error,
        }
