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
        factor_parts = [predicate_state.detach(), action_innovation.detach()]
        if transition_state is not None:
            if transition_reliability is None:
                raise ValueError("transition reliability is required with transition state")
            factor_parts.append(transition_state.detach())
            reliability = torch.cat([reliability, transition_reliability.detach()], dim=1)
        factors = torch.cat(factor_parts, dim=1)
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
        factor_reliability = torch.cat([nonnull_prior, null_reliability], dim=-1)
        score = torch.einsum("brd,bfd->brf", query, keys) / math.sqrt(keys.shape[-1])
        attention = torch.softmax(score + factor_reliability.clamp_min(1e-7).log()[:, None], dim=-1)
        no_reliable_factor = ~weights.gt(0).any(dim=-1)
        null_route = torch.zeros_like(attention)
        null_route[..., -1] = 1.0
        attention = torch.where(no_reliable_factor[:, None, None], null_route, attention)
        private = torch.einsum("brf,bfd->brd", attention, values)
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
        evidence_confidence = (attention[..., :-1] * weights[:, None, :]).sum(-1)
        effective_trust = self.evidence_trust_cap * evidence_confidence
        delta = scale * effective_trust * self.kappa * torch.tanh(raw_delta / self.kappa)
        delta = torch.where(no_reliable_factor[:, None], torch.zeros_like(delta), delta)
        flow_start = predicate_state.shape[1] + action_innovation.shape[1]
        flow_route_mass = attention[..., flow_start:-1].sum(-1) if transition_state is not None else attention[..., :0].sum(-1)
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
