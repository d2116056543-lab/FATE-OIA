from __future__ import annotations

import math

import torch
from torch import nn


class TIDAReasonReader(nn.Module):
    """Private reason correction with an explicit detach firewall."""

    def __init__(
        self,
        dim: int = 384,
        num_reasons: int = 21,
        kappa: float = 0.12,
        evidence_trust_cap: float = 0.25,
    ) -> None:
        super().__init__()
        self.num_reasons = int(num_reasons)
        self.kappa = float(kappa)
        self.evidence_trust_cap = float(evidence_trust_cap)
        if not 0.0 < self.evidence_trust_cap <= 1.0:
            raise ValueError("evidence_trust_cap must be in (0, 1]")
        self.reason_query = nn.Linear(dim, dim)
        self.factor_key = nn.Linear(dim, dim)
        self.factor_value = nn.Linear(dim, dim)
        self.null_key = nn.Parameter(torch.zeros(dim))
        self.delta_query = nn.Linear(dim, dim, bias=False)
        self.delta_value = nn.Linear(dim, dim, bias=False)
        self.flow_query = nn.Linear(dim, dim)
        self.flow_key = nn.Linear(dim, dim)
        self.flow_value = nn.Linear(dim, dim)
        nn.init.zeros_(self.flow_value.weight)
        nn.init.zeros_(self.flow_value.bias)
        self.flow_mix_cap = 0.35

    def forward(
        self,
        reason_nodes: torch.Tensor,
        predicate_state: torch.Tensor,
        action_innovation: torch.Tensor,
        reliability: torch.Tensor,
        *,
        temporal_scale: float | torch.Tensor,
        transition_state: torch.Tensor | None = None,
        transition_reliability: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        factors = torch.cat([predicate_state.detach(), action_innovation.detach()], dim=1)
        weights = reliability.detach().clamp(0, 1)
        query = self.reason_query(reason_nodes)
        keys = self.factor_key(factors)
        values = self.factor_value(factors)
        null_key = self.null_key.view(1, 1, -1).expand(factors.shape[0], -1, -1)
        null_value = torch.zeros_like(null_key)
        keys = torch.cat([keys, null_key], dim=1)
        values = torch.cat([values, null_value], dim=1)
        evidence_strength = weights.max(-1, keepdim=True).values
        relative_reliability = weights / weights.sum(-1, keepdim=True).clamp_min(1e-7)
        nonnull_prior = relative_reliability * evidence_strength
        null_reliability = (1.0 - evidence_strength).clamp_min(1e-7)
        base_factor_reliability = torch.cat([nonnull_prior, null_reliability], dim=-1)
        score = torch.einsum("brd,bfd->brf", query, keys) / math.sqrt(keys.shape[-1])
        base_attention = torch.softmax(score + base_factor_reliability.clamp_min(1e-7).log()[:, None], dim=-1)
        no_reliable_factor = ~weights.gt(0).any(dim=-1)
        null_route = torch.zeros_like(base_attention)
        null_route[..., -1] = 1.0
        base_attention = torch.where(no_reliable_factor[:, None, None], null_route, base_attention)

        if transition_state is None:
            attention = base_attention
            factor_reliability = base_factor_reliability
            nonnull_weights = weights
            private = torch.einsum("brf,bfd->brd", attention, values)
            flow_route_mass = attention[..., :0].sum(-1)
        else:
            if transition_reliability is None:
                raise ValueError("transition reliability is required with transition state")
            flow_state = transition_state.detach()
            flow_weights = transition_reliability.detach().clamp(0, 1)
            flow_strength = flow_weights.max(-1, keepdim=True).values
            flow_keys = self.flow_key(flow_state)
            flow_values = self.flow_value(flow_state)
            flow_score = torch.einsum("brd,bpd->brp", self.flow_query(reason_nodes), flow_keys) / math.sqrt(flow_keys.shape[-1])
            flow_distribution = torch.softmax(
                flow_score + flow_weights.clamp_min(1e-7).log()[:, None], dim=-1
            )
            no_flow = ~flow_weights.gt(0).any(dim=-1)
            no_reliable_factor = no_reliable_factor & no_flow
            # As in the action reader, measured motion owns the route budget;
            # reason queries can select a transition but cannot suppress all
            # temporal evidence through a learned null competition.
            flow_mix = self.flow_mix_cap * flow_strength.expand(-1, self.num_reasons)
            base_mix = 1.0 - flow_mix
            attention = torch.cat(
                [
                    base_mix[..., None] * base_attention[..., :-1],
                    flow_mix[..., None] * flow_distribution,
                    base_mix[..., None] * base_attention[..., -1:],
                ],
                dim=-1,
            )
            base_private = torch.einsum("brf,bfd->brd", base_attention, values)
            flow_private = torch.einsum("brp,bpd->brd", flow_distribution, flow_values)
            private = base_mix[..., None] * base_private + flow_mix[..., None] * flow_private
            factor_reliability = torch.cat(
                [
                    weights,
                    flow_weights,
                    (1.0 - torch.cat([weights, flow_weights], dim=1).max(-1, keepdim=True).values).clamp_min(1e-7),
                ],
                dim=-1,
            )
            nonnull_weights = torch.cat([weights, flow_weights], dim=1)
            flow_start = predicate_state.shape[1] + action_innovation.shape[1]
            flow_route_mass = attention[..., flow_start:-1].sum(-1)
        # Keep each reason's correction private. A query-evidence interaction
        # preserves label-specific signs while guaranteeing zero evidence gives
        # an exact zero correction without a static-query or bias shortcut.
        delta_query = self.delta_query(reason_nodes)
        delta_value = self.delta_value(private)
        raw_delta = torch.einsum("brd,brd->br", delta_query, delta_value) / math.sqrt(delta_query.shape[-1])
        scale = torch.as_tensor(temporal_scale, device=raw_delta.device, dtype=raw_delta.dtype)
        nonnull_mass = 1.0 - attention[..., -1]
        # Non-null attention alone only proves that a factor was selected. It
        # does not prove that the selected factor is reliable. Bound the side
        # residual by the reliability actually transported through the route.
        evidence_confidence = (attention[..., :-1] * nonnull_weights[:, None, :]).sum(-1)
        effective_trust = self.evidence_trust_cap * evidence_confidence
        delta = scale * effective_trust * self.kappa * torch.tanh(raw_delta / self.kappa)
        delta = torch.where(no_reliable_factor[:, None], torch.zeros_like(delta), delta)
        return {
            "reason_temporal_route": attention,
            "reason_temporal_evidence": private,
            "reason_factor_reliability": factor_reliability,
            "reason_nonnull_mass": nonnull_mass,
            "reason_evidence_confidence": evidence_confidence,
            "reason_effective_trust": effective_trust,
            "reason_temporal_attention": attention,
            "reason_private_token": private,
            "reason_raw_temporal_delta": raw_delta,
            "reason_temporal_delta": delta,
            "reason_flow_route_mass": flow_route_mass,
        }
