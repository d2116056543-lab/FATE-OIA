from __future__ import annotations

import math
from typing import Any, Sequence

import torch
from torch import nn

from .acpr_sparse_ops import entmax15_bisect


_TIER_TO_ID = {"abstained": 0, "reason_only": 1, "certified": 2}


class MOSAICTargetSparseRouter(nn.Module):
    """Target-owned sparse factor router with finite logits and a dustbin."""

    def __init__(
        self,
        ontology: dict[str, Any],
        *,
        dim: int = 384,
        dustbin_init: float = -4.0,
    ) -> None:
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
        # Shadow learning starts before an audit has accumulated useful cV.
        # A conservative low dustbin prior prevents sparse entmax from
        # assigning all initial mass to dustbin and starving the route owner.
        self.dustbin_logit = nn.Parameter(torch.full((2, self.action_count), float(dustbin_init)))

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
        # Certificate/admission is a deployment gate, not a learning gate.
        # Shadow routing must expose the factor path before certification;
        # otherwise the all-abstained state creates a zero-gradient deadlock.
        candidate = self.candidate_edge_mask
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
        credibility: torch.Tensor | None,
        target_utility: torch.Tensor | None,
        direction_index: int,
        active_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        evidence_by_edge = evidence.unsqueeze(-1) if evidence.ndim == 2 else evidence
        if evidence_by_edge.shape != factor_scores.shape:
            raise ValueError("IC-DOR route evidence must be [B,F] or [B,F,A]")
        finite_logits = factor_scores + evidence_by_edge.clamp_min(1e-8).log()
        if credibility is not None:
            finite_logits = finite_logits + credibility.detach().clamp_min(1e-8).log().unsqueeze(-1)
        if target_utility is not None:
            # ``u`` is audit-derived target utility. Keep a small shadow
            # learning floor so an inconclusive early audit cannot recreate
            # the cold-start deadlock; final use remains edge-admission gated.
            finite_logits = finite_logits + target_utility.detach().clamp_min(0.10).log()
        finite_logits = finite_logits + self.direction_bias[direction_index].view(1, 1, -1)
        dustbin = self.dustbin_logit[direction_index].view(1, 1, -1).expand(finite_logits.shape[0], -1, -1)
        # Exclude forbidden factors before entmax so their semantics cannot
        # alter normalization for an allowed edge or the dustbin. A relative
        # finite floor keeps the bisection interval stable for sparse entmax.
        allowed_logits = finite_logits.masked_fill(~active_mask.unsqueeze(0), float("-inf"))
        floor = torch.maximum(allowed_logits.amax(dim=1, keepdim=True), dustbin) - 100.0
        masked_logits = torch.where(active_mask.unsqueeze(0), finite_logits, floor)
        unconstrained = entmax15_bisect(torch.cat((masked_logits, dustbin), dim=1), dim=1)
        relative = torch.where(
            active_mask.unsqueeze(0),
            unconstrained[:, : self.factor_count],
            torch.zeros_like(unconstrained[:, : self.factor_count]),
        )
        # CREDO-MAP keeps two different objects: ``pi`` describes which factor
        # owns an already available route, while ``m`` is the absolute visual
        # evidence mass.  The old implementation exposed only ``relative``;
        # normalising it in a rereader let zero-credibility routes create a
        # generic target head through the query and centre sample.
        raw_mass = relative.sum(dim=1)
        has_visual_evidence = (
            (evidence_by_edge.clamp_min(0.0) * active_mask.unsqueeze(0).to(evidence_by_edge.dtype))
            .sum(dim=1)
            > 1e-8
        )
        route_mass = raw_mass * has_visual_evidence.to(raw_mass.dtype)
        route_distribution = relative / raw_mass.unsqueeze(1).clamp_min(1e-8)
        route_distribution = route_distribution * has_visual_evidence.unsqueeze(1).to(route_distribution.dtype)
        factor_weights = route_distribution * route_mass.unsqueeze(1)
        # Preserve the exact invariant needed by downstream action/reason
        # transport: no factor evidence means no correction, not a centre
        # sample with a learned query bias.
        zero_mass = ~has_visual_evidence
        topk = min(2, self.factor_count)
        _, topk_ids = route_distribution.transpose(1, 2).topk(topk, dim=-1)
        topk_ids = torch.where(
            zero_mass.unsqueeze(-1),
            torch.full_like(topk_ids, -1),
            topk_ids,
        )
        return {
            "weights": factor_weights,
            "distribution": route_distribution,
            "mass": route_mass,
            "dustbin": 1.0 - route_mass,
            "masked_logits": masked_logits,
            "topk_factor_ids": topk_ids,
            "zero_mass_mask": zero_mass,
        }

    def forward(
        self,
        factor_features: torch.Tensor,
        factor_positive_evidence: torch.Tensor,
        factor_negative_evidence: torch.Tensor,
        action_queries: torch.Tensor,
        *,
        route_mode: str,
        factor_credibility: torch.Tensor | None = None,
        factor_target_utility: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if factor_features.ndim != 3 or factor_features.shape[1] != self.factor_count:
            raise ValueError("IC-DOR router factor_features must be [B,F,D]")
        if factor_positive_evidence.shape != factor_features.shape[:2] or factor_negative_evidence.shape != factor_features.shape[:2]:
            raise ValueError("IC-DOR router evidence must be [B,F]")
        if factor_credibility is not None and factor_credibility.shape != factor_features.shape[:2]:
            raise ValueError("IC-DOR router credibility must be [B,F]")
        if factor_target_utility is not None:
            if factor_target_utility.shape == (self.factor_count, self.action_count):
                factor_target_utility = factor_target_utility.unsqueeze(0).expand(factor_features.shape[0], -1, -1)
            if factor_target_utility.shape != (factor_features.shape[0], self.factor_count, self.action_count):
                raise ValueError("IC-DOR router target utility must be [F,4] or [B,F,4]")
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
        support = self._route_one_direction(
            factor_scores, support_evidence, factor_credibility, factor_target_utility, 0, active[0]
        )
        veto = self._route_one_direction(
            factor_scores, veto_evidence, factor_credibility, factor_target_utility, 1, active[1]
        )
        return {
            "support_weights": support["weights"],
            "veto_weights": veto["weights"],
            "support_route_distribution": support["distribution"],
            "veto_route_distribution": veto["distribution"],
            "support_route_mass": support["mass"],
            "veto_route_mass": veto["mass"],
            "support_dustbin": support["dustbin"],
            "veto_dustbin": veto["dustbin"],
            "support_route_logits": support["masked_logits"],
            "veto_route_logits": veto["masked_logits"],
            "support_topk_factor_ids": support["topk_factor_ids"],
            "veto_topk_factor_ids": veto["topk_factor_ids"],
            "support_zero_mass_mask": support["zero_mass_mask"],
            "veto_zero_mass_mask": veto["zero_mass_mask"],
            "route_distribution": torch.stack((support["distribution"], veto["distribution"]), dim=1),
            "route_mass": torch.stack((support["mass"], veto["mass"]), dim=1),
            "topk_factor_ids": torch.stack((support["topk_factor_ids"], veto["topk_factor_ids"]), dim=1),
            "zero_mass_mask": torch.stack((support["zero_mass_mask"], veto["zero_mass_mask"]), dim=1),
            "active_edge_mask": active,
            "active_edge_polarity_mask": polarity,
            "support_route_evidence": support_evidence,
            "veto_route_evidence": veto_evidence,
            "action_target_utility_effective": (
                torch.ones_like(factor_scores)
                if factor_target_utility is None else factor_target_utility.detach()
            ),
            "route_mode_code": factor_scores.new_tensor({"off": 0, "shadow": 1, "admitted": 2}[route_mode]),
        }
