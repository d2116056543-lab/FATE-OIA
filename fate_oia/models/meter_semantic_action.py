from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .acpr_sparse_ops import entmax15_bisect


class METERSemanticActionPeer(nn.Module):
    """Directly supervised signed-factor action expert and action-specific peer selector."""

    def __init__(self, dim: int = 384, action_dim: int = 4, factor_dim: int = 21) -> None:
        super().__init__()
        self.dim = int(dim)
        self.action_dim = int(action_dim)
        self.factor_dim = int(factor_dim)
        self.action_query = nn.Linear(dim, dim)
        self.factor_key = nn.Linear(dim, dim)
        self.factor_value = nn.Parameter(torch.empty(action_dim, dim))
        self.semantic_bias = nn.Parameter(torch.zeros(action_dim))
        self.null_key = nn.Parameter(torch.randn(dim) * 0.02)
        self.null_logit_offset = nn.Parameter(
            torch.full((action_dim,), math.log(0.10 / 0.90))
        )
        self.selector = nn.Sequential(
            nn.Linear(dim * 2 + 4, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )
        nn.init.xavier_uniform_(self.factor_value)
        nn.init.constant_(self.selector[-1].bias, 2.2)

    @staticmethod
    def _ramp(progress: float) -> float:
        return float(min(max(progress / 0.10, 0.0), 1.0))

    def forward(
        self,
        action_logits_visual: Tensor,
        action_nodes: Tensor,
        factor_action_tokens: Tensor,
        factor_reliability: Tensor,
        *,
        progress: float = 1.0,
    ) -> dict[str, Tensor]:
        if action_nodes.shape[1:] != (self.action_dim, self.dim):
            raise ValueError("Action nodes have an unexpected shape")
        if factor_action_tokens.shape[1:] != (self.factor_dim, self.dim):
            raise ValueError("Factor action tokens have an unexpected shape")
        query = self.action_query(action_nodes)
        key = self.factor_key(factor_action_tokens)
        score = torch.einsum("bad,brd->bar", query, key) / math.sqrt(self.dim)
        null_score = torch.einsum("bad,d->ba", query, self.null_key)
        null_mass = torch.sigmoid(
            null_score
            - score.mean(dim=-1)
            + self.null_logit_offset.view(1, -1)
        )
        dense = torch.softmax(score, dim=-1)
        sparse = entmax15_bisect(score, dim=-1)
        factor_distribution = (
            (1.0 - self._ramp(progress)) * dense
            + self._ramp(progress) * sparse
        )
        factor_weights = (1.0 - null_mass).unsqueeze(-1) * factor_distribution
        factor_values = torch.einsum("brd,ad->bar", factor_action_tokens, self.factor_value)
        contributions = factor_weights * factor_reliability.unsqueeze(1) * factor_values
        semantic_logits = self.semantic_bias.view(1, -1) + contributions.sum(dim=-1)
        summary = torch.einsum("bar,brd->bad", factor_weights, factor_action_tokens)
        feature = torch.cat(
            [
                action_nodes,
                summary,
                (semantic_logits - action_logits_visual).abs().unsqueeze(-1),
                factor_reliability.mean(dim=-1, keepdim=True).unsqueeze(1).expand(-1, self.action_dim, -1),
                null_mass.unsqueeze(-1),
                contributions.abs().mean(dim=-1, keepdim=True),
            ],
            dim=-1,
        )
        selector = torch.sigmoid(self.selector(feature).squeeze(-1))
        peer_logits = selector * action_logits_visual + (1.0 - selector) * semantic_logits
        final_logits = action_logits_visual + self._ramp(progress) * (peer_logits - action_logits_visual)
        return {
            "action_logits_visual": action_logits_visual,
            "action_logits_semantic": semantic_logits,
            "action_logits_peer": peer_logits,
            "action_logits_final": final_logits,
            "action_factor_weights": factor_weights,
            "action_factor_values": factor_values,
            "action_factor_contributions": contributions,
            "semantic_bias": self.semantic_bias.view(1, -1).expand_as(semantic_logits),
            "action_null_mass": null_mass,
            "action_selector": selector,
        }
