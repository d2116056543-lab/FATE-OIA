from __future__ import annotations

import torch
from torch import Tensor, nn

from .acpr_sparse_ops import entmax15_bisect


class FactorSpecificActionTransport(nn.Module):
    """Exact factor-index-specific additive action evidence transport."""

    def __init__(
        self,
        dim: int = 384,
        action_dim: int = 4,
        factor_dim: int = 21,
        rank: int = 16,
        rms_momentum: float = 0.95,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.action_dim = int(action_dim)
        self.factor_dim = int(factor_dim)
        self.rank = int(rank)
        self.rms_momentum = float(rms_momentum)
        self.action_query = nn.Linear(dim, dim)
        self.factor_key = nn.Parameter(torch.randn(factor_dim, dim, dim) * 0.01)
        self.type_bias = nn.Parameter(torch.zeros(action_dim, factor_dim))
        self.null_bias = nn.Parameter(torch.zeros(action_dim))
        self.factor_down = nn.Parameter(torch.randn(factor_dim, rank, dim) * 0.02)
        self.factor_up = nn.Parameter(torch.randn(factor_dim, dim, rank) * 0.02)
        self.register_buffer("running_visual_rms", torch.ones(action_dim), persistent=True)
        self.register_buffer("running_updates", torch.zeros((), dtype=torch.long), persistent=True)

    @staticmethod
    def _ramp(progress: float) -> float:
        return float(min(max(progress / 0.10, 0.0), 1.0))

    def forward(
        self,
        action_logits_visual: Tensor,
        action_nodes: Tensor,
        factor_typed_token: Tensor,
        factor_reliability: Tensor,
        factor_action_ownership: Tensor,
        *,
        progress: float = 1.0,
        update_running_stats: bool = False,
    ) -> dict[str, Tensor]:
        query = self.action_query(action_nodes)
        factor_key = torch.einsum(
            "brd,rde->bre", factor_typed_token, self.factor_key
        )
        score = torch.einsum("bad,brd->bar", query, factor_key) + self.type_bias
        owner = factor_action_ownership.to(score).view(1, 1, -1)
        allowed = owner > 0
        masked_score = score.masked_fill(~allowed, -1e9)
        full_score = torch.cat(
            [
                masked_score,
                self.null_bias.view(1, -1, 1).expand(score.shape[0], -1, -1),
            ],
            dim=-1,
        )
        dense = torch.softmax(full_score, dim=-1)
        sparse = entmax15_bisect(full_score, dim=-1)
        ramp = self._ramp(progress)
        full_weight = dense * (1.0 - ramp) + sparse * ramp
        factor_weight = full_weight[..., :-1] * allowed.to(score.dtype)
        null_weight = full_weight[..., -1]
        low = torch.einsum("brd,rkd->brk", factor_typed_token, self.factor_down)
        projected = torch.einsum("brk,rdk->brd", low, self.factor_up)
        factor_value = torch.einsum("bad,brd->bar", query, projected)
        contributions = (
            owner
            * factor_reliability.unsqueeze(1)
            * factor_weight
            * factor_value
        )
        if self.training and update_running_stats:
            with torch.no_grad():
                current = action_logits_visual.detach().float().square().mean(0).sqrt()
                if int(self.running_updates) == 0:
                    self.running_visual_rms.copy_(current)
                else:
                    self.running_visual_rms.mul_(self.rms_momentum).add_(
                        current * (1.0 - self.rms_momentum)
                    )
                self.running_updates.add_(1)
        kappa = (0.20 * self.running_visual_rms).clamp_min(1e-4).to(
            action_logits_visual.dtype
        )
        raw_sum = contributions.sum(-1)
        evidence_delta = kappa.view(1, -1) * torch.tanh(
            raw_sum / kappa.view(1, -1)
        )
        final = action_logits_visual + ramp * evidence_delta
        return {
            "action_logits_visual": action_logits_visual,
            "action_evidence_delta": evidence_delta,
            "action_logits_final": final,
            "action_factor_weights": factor_weight,
            "action_null_factor_weight": null_weight,
            "action_factor_values": factor_value,
            "action_factor_contributions": contributions,
            "action_correction_kappa": kappa,
            "action_correction_rms_ratio": evidence_delta.detach().float().square().mean(0).sqrt()
            / action_logits_visual.detach().float().square().mean(0).sqrt().clamp_min(1e-6),
        }


METERSemanticActionPeer = FactorSpecificActionTransport
