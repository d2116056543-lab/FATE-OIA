from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .acpr_sparse_ops import entmax15_bisect


class METERSemanticActionPeer(nn.Module):
    """Directly supervised signed-factor action expert and action-specific peer selector."""

    def __init__(
        self,
        dim: int = 384,
        action_dim: int = 4,
        factor_dim: int = 21,
        *,
        semantic_transport_target_ratio: float = 0.15,
        semantic_transport_rms_momentum: float = 0.95,
        semantic_transport_per_action: bool = True,
    ) -> None:
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
        if not semantic_transport_per_action:
            raise ValueError("METER requires per-action semantic transport")
        self.semantic_rms_ratio = float(semantic_transport_target_ratio)
        self.rms_momentum = float(semantic_transport_rms_momentum)
        self.register_buffer(
            "running_visual_rms", torch.ones(action_dim), persistent=True
        )
        self.register_buffer(
            "running_semantic_delta_rms", torch.ones(action_dim), persistent=True
        )
        self.register_buffer(
            "running_rms_updates", torch.zeros((), dtype=torch.long), persistent=True
        )

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
        update_running_stats: bool = False,
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
        contributions = (
            factor_weights * factor_reliability.unsqueeze(1) * factor_values
        )
        semantic_bias = self.semantic_bias.view(1, -1).expand_as(action_logits_visual)
        semantic_logits = semantic_bias + contributions.sum(dim=-1)
        semantic_delta = semantic_logits - action_logits_visual
        if self.training and update_running_stats:
            with torch.no_grad():
                batch_visual_rms = (
                    action_logits_visual.detach().float().square().mean(dim=0).sqrt()
                )
                batch_delta_rms = (
                    semantic_delta.detach().float().square().mean(dim=0).sqrt()
                )
                if int(self.running_rms_updates.item()) == 0:
                    self.running_visual_rms.copy_(batch_visual_rms)
                    self.running_semantic_delta_rms.copy_(batch_delta_rms)
                else:
                    self.running_visual_rms.mul_(self.rms_momentum).add_(
                        batch_visual_rms * (1.0 - self.rms_momentum)
                    )
                    self.running_semantic_delta_rms.mul_(self.rms_momentum).add_(
                        batch_delta_rms * (1.0 - self.rms_momentum)
                    )
                self.running_rms_updates.add_(1)
        transport_scale = (
            self.semantic_rms_ratio
            * self.running_visual_rms.clamp_min(1e-4)
            / self.running_semantic_delta_rms.clamp_min(1e-4)
        ).clamp(min=1e-3, max=100.0).to(dtype=semantic_delta.dtype)
        transport_delta = transport_scale.view(1, -1) * semantic_delta
        semantic_transport_logits = action_logits_visual + transport_delta
        actual_ratio = (
            transport_delta.detach().float().square().mean(dim=0).sqrt()
            / action_logits_visual.detach().float().square().mean(dim=0).sqrt().clamp_min(1e-6)
        )
        summary = torch.einsum("bar,brd->bad", factor_weights, factor_action_tokens)
        feature = torch.cat(
            [
                action_nodes,
                summary,
                transport_delta.abs().unsqueeze(-1),
                factor_reliability.mean(dim=-1, keepdim=True).unsqueeze(1).expand(-1, self.action_dim, -1),
                null_mass.unsqueeze(-1),
                contributions.abs().mean(dim=-1, keepdim=True),
            ],
            dim=-1,
        )
        selector = torch.sigmoid(self.selector(feature).squeeze(-1))
        peer_logits = (
            selector * action_logits_visual
            + (1.0 - selector) * semantic_transport_logits
        )
        selector_regret = torch.sigmoid(
            self.selector(feature.detach()).squeeze(-1)
        )
        peer_logits_regret = (
            selector_regret * action_logits_visual.detach()
            + (1.0 - selector_regret) * semantic_transport_logits.detach()
        )
        final_logits = action_logits_visual + self._ramp(progress) * (peer_logits - action_logits_visual)
        return {
            "action_logits_visual": action_logits_visual,
            "action_logits_semantic": semantic_logits,
            "action_logits_semantic_transport": semantic_transport_logits,
            "action_semantic_transport_delta": transport_delta,
            "action_logits_peer": peer_logits,
            "action_logits_peer_regret": peer_logits_regret,
            "action_logits_final": final_logits,
            "action_factor_weights": factor_weights,
            "action_factor_values": factor_values,
            "action_factor_contributions": contributions,
            "semantic_bias": semantic_bias,
            "semantic_transport_scale": transport_scale,
            "semantic_transport_actual_ratio": actual_ratio,
            "semantic_transport_saturation_rate": (
                (transport_scale <= 1e-3 + 1e-8)
                | (transport_scale >= 100.0 - 1e-6)
            ).float(),
            "action_null_mass": null_mass,
            "action_selector": selector,
            "action_selector_regret": selector_regret,
        }
