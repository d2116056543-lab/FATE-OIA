from __future__ import annotations

import math
from typing import Any, Sequence

import torch
from torch import nn

from .acpr_sparse_ops import entmax15_bisect


_TIER_TO_ID = {"abstained": 0, "reason_only": 1, "certified": 2}


class MOSAICTargetSparseRouter(nn.Module):
    """Target-owned sparse factor router with finite logits and a dustbin."""

    def __init__(self, ontology: dict[str, Any], *, dim: int = 384) -> None:
        super().__init__()
        self.factor_names = tuple(factor["name"] for factor in ontology["factors"])
        self.action_names = tuple(ontology["action_names"])
        self.factor_count = len(self.factor_names)
        self.action_count = len(self.action_names)
        if self.action_count != 4 or self.factor_count == 0:
            raise ValueError("IC-DOR target router requires the complete factor/action ontology")
        factor_index = ontology["factor_index"]
        action_index = ontology["action_index"]
        candidate_polarity = torch.zeros(2, 2, self.factor_count, self.action_count, dtype=torch.bool)
        polarity_index = {"present": 0, "absent": 1}
        for direction_index, direction in enumerate(("support", "veto")):
            for action_name, directions in ontology["action_routes"].items():
                for edge in directions[direction]:
                    if edge["polarity"] not in {"present", "absent"}:
                        raise ValueError("IC-DOR router received invalid route polarity")
                    candidate_polarity[
                        direction_index,
                        polarity_index[edge["polarity"]],
                        factor_index[edge["factor"]],
                        action_index[action_name],
                    ] = True
        if (candidate_polarity.sum(dim=1) > 1).any():
            raise ValueError("IC-DOR permits only one polarity per factor-target-direction edge")
        candidate = candidate_polarity.any(dim=1)
        self.register_buffer("candidate_polarity_mask", candidate_polarity, persistent=True)
        self.register_buffer("candidate_edge_mask", candidate, persistent=True)
        self.register_buffer("certificate_tier", torch.zeros(self.factor_count, dtype=torch.long), persistent=True)
        self.register_buffer("edge_admission_mask", torch.zeros_like(candidate), persistent=True)
        self.factor_proj = nn.Linear(dim, dim, bias=False)
        self.target_proj = nn.Linear(dim, dim, bias=False)
        self.direction_bias = nn.Parameter(torch.zeros(2, self.action_count))
        self.dustbin_logit = nn.Parameter(torch.zeros(2, self.action_count))

    @torch.no_grad()
    def set_certificate_tiers(self, tiers: Sequence[str]) -> None:
        if len(tiers) != self.factor_count or any(tier not in _TIER_TO_ID for tier in tiers):
            raise ValueError("IC-DOR certificate tier vector is invalid")
        self.certificate_tier.copy_(torch.tensor([_TIER_TO_ID[tier] for tier in tiers], device=self.certificate_tier.device))

    @torch.no_grad()
    def set_edge_admission(self, edge_admission_mask: torch.Tensor) -> None:
        if edge_admission_mask.shape != self.candidate_edge_mask.shape or edge_admission_mask.dtype != torch.bool:
            raise ValueError("IC-DOR edge admission mask has an invalid shape or dtype")
        self.edge_admission_mask.copy_(edge_admission_mask & self.candidate_edge_mask)

    def _active_edge_mask(self, route_mode: str) -> torch.Tensor:
        certified = (self.certificate_tier == _TIER_TO_ID["certified"]).view(1, self.factor_count, 1)
        candidate = self.candidate_edge_mask & certified
        if route_mode == "off":
            return torch.zeros_like(candidate)
        if route_mode == "shadow":
            return candidate
        if route_mode == "admitted":
            return candidate & self.edge_admission_mask
        raise ValueError("IC-DOR route_mode must be off, shadow, or admitted")

    def _route_one_direction(
        self,
        factor_scores: torch.Tensor,
        evidence: torch.Tensor,
        direction_index: int,
        active_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        evidence_by_edge = evidence.unsqueeze(-1) if evidence.ndim == 2 else evidence
        if evidence_by_edge.shape != factor_scores.shape:
            raise ValueError("IC-DOR route evidence must be [B,F] or [B,F,A]")
        finite_logits = factor_scores + evidence_by_edge.clamp_min(1e-8).log()
        finite_logits = finite_logits + self.direction_bias[direction_index].view(1, 1, -1)
        dustbin = self.dustbin_logit[direction_index].view(1, 1, -1).expand(finite_logits.shape[0], -1, -1)
        # Exclude forbidden factors before entmax so their semantics cannot
        # alter normalization for an allowed edge or the dustbin. A relative
        # finite floor keeps the bisection interval stable for sparse entmax.
        allowed_logits = finite_logits.masked_fill(~active_mask.unsqueeze(0), float("-inf"))
        floor = torch.maximum(allowed_logits.amax(dim=1, keepdim=True), dustbin) - 100.0
        masked_logits = torch.where(active_mask.unsqueeze(0), finite_logits, floor)
        unconstrained = entmax15_bisect(torch.cat((masked_logits, dustbin), dim=1), dim=1)
        factor_weights = torch.where(
            active_mask.unsqueeze(0),
            unconstrained[:, : self.factor_count],
            torch.zeros_like(unconstrained[:, : self.factor_count]),
        )
        # Any unused mass, including the hard-masked paths, belongs to dustbin.
        dustbin_weight = 1.0 - factor_weights.sum(dim=1)
        return factor_weights, dustbin_weight, masked_logits

    def forward(
        self,
        factor_features: torch.Tensor,
        factor_positive_evidence: torch.Tensor,
        factor_negative_evidence: torch.Tensor,
        action_queries: torch.Tensor,
        *,
        route_mode: str,
    ) -> dict[str, torch.Tensor]:
        if factor_features.ndim != 3 or factor_features.shape[1] != self.factor_count:
            raise ValueError("IC-DOR router factor_features must be [B,F,D]")
        if factor_positive_evidence.shape != factor_features.shape[:2] or factor_negative_evidence.shape != factor_features.shape[:2]:
            raise ValueError("IC-DOR router evidence must be [B,F]")
        if action_queries.shape != (factor_features.shape[0], self.action_count, factor_features.shape[-1]):
            raise ValueError("IC-DOR router action_queries must be [B,4,D]")
        factors = self.factor_proj(factor_features.detach())
        targets = self.target_proj(action_queries.detach())
        factor_scores = torch.einsum("bfd,bad->bfa", factors, targets) / math.sqrt(factors.shape[-1])
        active = self._active_edge_mask(route_mode)
        polarity = self.candidate_polarity_mask & active.unsqueeze(1)
        support_evidence = (
            polarity[0, 0].unsqueeze(0) * factor_positive_evidence.detach().unsqueeze(-1)
            + polarity[0, 1].unsqueeze(0) * factor_negative_evidence.detach().unsqueeze(-1)
        )
        veto_evidence = (
            polarity[1, 0].unsqueeze(0) * factor_positive_evidence.detach().unsqueeze(-1)
            + polarity[1, 1].unsqueeze(0) * factor_negative_evidence.detach().unsqueeze(-1)
        )
        support, support_dustbin, support_logits = self._route_one_direction(
            factor_scores, support_evidence, 0, active[0]
        )
        veto, veto_dustbin, veto_logits = self._route_one_direction(
            factor_scores, veto_evidence, 1, active[1]
        )
        return {
            "support_weights": support,
            "veto_weights": veto,
            "support_dustbin": support_dustbin,
            "veto_dustbin": veto_dustbin,
            "support_route_logits": support_logits,
            "veto_route_logits": veto_logits,
            "active_edge_mask": active,
            "active_edge_polarity_mask": polarity,
            "support_route_evidence": support_evidence,
            "veto_route_evidence": veto_evidence,
            "route_mode_code": factor_scores.new_tensor({"off": 0, "shadow": 1, "admitted": 2}[route_mode]),
        }
