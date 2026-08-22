from __future__ import annotations

import math

import torch
from torch import nn

from .acpr_sparse_ops import entmax15_bisect


class TIDAActionReader(nn.Module):
    def __init__(
        self,
        dim: int = 384,
        num_actions: int = 4,
        num_predicates: int = 32,
        kappa: float = 0.15,
        eps: float = 1e-7,
        evidence_trust_cap: float = 0.25,
    ) -> None:
        super().__init__()
        self.num_actions = int(num_actions)
        self.num_predicates = int(num_predicates)
        self.kappa = float(kappa)
        self.eps = float(eps)
        self.evidence_trust_cap = float(evidence_trust_cap)
        if not 0.0 < self.evidence_trust_cap <= 1.0:
            raise ValueError("evidence_trust_cap must be in (0, 1]")
        self.action_query = nn.Linear(dim, dim)
        self.factor_key = nn.Linear(dim, dim)
        self.visual_value_projection = nn.Linear(dim, dim)
        self.action_output_weight = nn.Parameter(torch.zeros(num_actions, dim))
        self.null_key = nn.Parameter(torch.zeros(dim))
        self.flow_query = nn.Linear(dim, dim)
        self.flow_key = nn.Linear(dim, dim)
        self.flow_value = nn.Linear(dim, dim)
        self.flow_output_weight = nn.Parameter(torch.zeros(num_actions, dim))
        self.flow_mix_cap = 0.35

    def _reconcile(self, contribution: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        residual = delta - contribution.sum(-1)
        largest = contribution.abs().argmax(-1, keepdim=True)
        return contribution.scatter_add(-1, largest, residual.unsqueeze(-1))

    def forward(
        self,
        action_nodes: torch.Tensor,
        predicate_state: torch.Tensor,
        action_innovation: torch.Tensor,
        reliability: torch.Tensor,
        *,
        temporal_scale: float | torch.Tensor,
        predicate_key_state: torch.Tensor | None = None,
        transition_state: torch.Tensor | None = None,
        transition_reliability: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch = action_nodes.shape[0]
        factors = torch.cat([predicate_state, action_innovation], dim=1)
        key_factors = torch.cat([
            predicate_state if predicate_key_state is None else predicate_key_state,
            action_innovation,
        ], dim=1)
        if factors.shape[1] != self.num_predicates + self.num_actions:
            raise ValueError("factor bank must contain predicate and action innovation factors")
        if reliability.shape != factors.shape[:2]:
            raise ValueError("reliability shape mismatch")
        query = self.action_query(action_nodes)
        keys = self.factor_key(key_factors)
        nonnull_score = torch.einsum("bad,bfd->baf", query, keys) / math.sqrt(keys.shape[-1])
        null_score = torch.einsum("bad,d->ba", query, self.null_key)[:, :, None] / math.sqrt(keys.shape[-1])
        nonnull_reliability = reliability.detach().clamp(0, 1)
        null_reliability = (1.0 - nonnull_reliability.max(-1, keepdim=True).values).clamp_min(self.eps)
        base_reliability = torch.cat([nonnull_reliability, null_reliability], dim=-1)
        route_score = torch.cat([nonnull_score, null_score], dim=-1)
        base_route = entmax15_bisect(route_score + base_reliability.clamp_min(self.eps).log()[:, None], dim=-1)
        visual_value = self.visual_value_projection(factors)
        null_value = torch.zeros(batch, 1, visual_value.shape[-1], device=visual_value.device, dtype=visual_value.dtype)
        base_value = torch.cat([visual_value, null_value], dim=1)[:, None].expand(-1, self.num_actions, -1, -1)
        base_score = torch.einsum("bafd,ad->baf", base_value, self.action_output_weight)

        if transition_state is None:
            route = base_route
            factor_value = base_value
            factor_score = base_score
            factor_reliability = base_reliability
            all_keys = torch.cat([keys, self.null_key.view(1, 1, -1).expand(batch, -1, -1)], dim=1)
            flow_route_mass = route[..., :0].sum(-1)
        else:
            if transition_reliability is None:
                raise ValueError("transition reliability is required with transition state")
            flow_reliability = transition_reliability.detach().clamp(0, 1)
            flow_keys = self.flow_key(transition_state)
            flow_query = self.flow_query(action_nodes)
            flow_score_nonnull = torch.einsum("bad,bpd->bap", flow_query, flow_keys) / math.sqrt(flow_keys.shape[-1])
            flow_distribution = entmax15_bisect(
                flow_score_nonnull + flow_reliability.clamp_min(self.eps).log()[:, None],
                dim=-1,
            )
            flow_strength = flow_reliability.max(-1, keepdim=True).values
            flow_values = self.flow_value(transition_state)
            flow_value = flow_values[:, None].expand(-1, self.num_actions, -1, -1)
            flow_score = torch.einsum("bapd,ad->bap", flow_value, self.flow_output_weight)
            # Reliability decides how much temporal evidence exists. The target
            # query only distributes that fixed budget and cannot learn to route
            # all motion into a null token to evade counterfactual supervision.
            flow_mix = self.flow_mix_cap * flow_strength.expand(-1, self.num_actions)
            base_mix = 1.0 - flow_mix
            route = torch.cat(
                [
                    base_mix[..., None] * base_route[..., :-1],
                    flow_mix[..., None] * flow_distribution,
                    base_mix[..., None] * base_route[..., -1:],
                ],
                dim=-1,
            )
            factor_value = torch.cat([base_value[..., :-1, :], flow_value, null_value[:, None].expand(-1, self.num_actions, -1, -1)], dim=2)
            factor_score = torch.cat([base_score[..., :-1], flow_score, base_score[..., -1:]], dim=-1)
            factor_reliability = torch.cat(
                [nonnull_reliability, flow_reliability, (1.0 - torch.cat([nonnull_reliability, flow_reliability], dim=1).max(-1, keepdim=True).values).clamp_min(self.eps)],
                dim=-1,
            )
            all_keys = torch.cat(
                [keys, flow_keys, self.null_key.view(1, 1, -1).expand(batch, -1, -1)], dim=1
            )
            flow_start = self.num_predicates + self.num_actions
            flow_route_mass = route[..., flow_start:-1].sum(-1)
        raw_contribution = route * factor_score
        raw_sum = raw_contribution.sum(-1)
        scale = torch.as_tensor(temporal_scale, device=raw_sum.device, dtype=raw_sum.dtype)
        evidence_confidence = (route[..., :-1] * factor_reliability[:, None, :-1]).sum(-1)
        effective_trust = self.evidence_trust_cap * evidence_confidence
        delta = scale * effective_trust * self.kappa * torch.tanh(raw_sum / self.kappa)
        ratio = torch.where(raw_sum.abs() > self.eps, delta / raw_sum, torch.ones_like(raw_sum))
        bounded = raw_contribution * ratio[..., None]
        bounded = torch.where((raw_sum.abs() > self.eps)[..., None], bounded, torch.zeros_like(bounded))
        bounded = self._reconcile(bounded, delta)
        return {
            "action_route": route,
            "action_factor_keys": all_keys,
            "action_factor_value": factor_value,
            "action_factor_reliability": factor_reliability,
            "action_raw_factor_contribution": raw_contribution,
            "action_factor_contribution": bounded,
            "action_temporal_delta": delta,
            "action_evidence_confidence": evidence_confidence,
            "action_effective_trust": effective_trust,
            "action_nonnull_mass": route[..., :-1].sum(-1),
            "action_route_entropy": -(route * route.clamp_min(self.eps).log()).sum(-1),
            "action_flow_route_mass": flow_route_mass,
            "selected_action_temporal_evidence": torch.einsum("baf,bafd->bad", route, factor_value),
        }
