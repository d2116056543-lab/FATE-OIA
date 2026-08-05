from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from .lens_action_losses import action_asl_loss, action_cardinality_loss, action_soft_f1_loss
from .lens_grounding_losses import map_anchor_loss, route_sparsity_loss, state_anchor_loss, unknown_prior_loss, view_consistency_loss
from .lens_latent_losses import emission_loss, emission_prior_loss, state_loss
from .lens_reason_losses import reason_asl_loss, reason_rank_loss, reason_soft_f1_loss


class LENSLossRegistry:
    """One registry entry, one raw call and one weight application per loss."""

    def __init__(self, weights: dict[str, float]) -> None:
        self.weights = weights
        self.owners = {
            "action_final": "foundation+adaptive_evidence+latent_state+action_reread",
            "action_base": "foundation+adaptive_evidence+latent_state",
            "action_factor_aux": "adaptive_evidence+latent_state+action_reread",
            "reason_formal": "foundation+adaptive_evidence+latent_state+annotation_emission",
            "reason_latent_aux": "foundation+adaptive_evidence+latent_state",
            "state": "adaptive_evidence+latent_state",
            "emission": "annotation_emission",
            "map_anchor": "adaptive_evidence",
            "state_anchor": "latent_state",
        }

    def __call__(self, out: dict[str, Tensor], batch: dict[str, Tensor], responsibilities: dict[str, Tensor] | None = None, multipliers: dict[str, float] | None = None) -> tuple[Tensor, dict[str, Tensor]]:
        action, reason = batch["action"], batch["reason"]
        unknown_prior = out["state_unknown_prob"].new_full((21,), 0.5)
        raw: dict[str, Tensor] = {
            "action_final": action_asl_loss(out["action_logits_final"], action),
            "action_base": action_asl_loss(out["action_logits_base"], action),
            "action_factor_aux": action_asl_loss(out["action_logits_factor_aux"], action),
            "action_soft_f1": action_soft_f1_loss(out["action_logits_final"], action),
            "action_cardinality": action_cardinality_loss(out["action_logits_final"], action),
            "reason_formal": reason_asl_loss(out.get("reason_logits_formal_train", out["reason_logits_formal"]), reason),
            "reason_latent_aux": reason_asl_loss(out.get("reason_logits_latent_train", out["reason_logits_latent"]), reason),
            "reason_rank": reason_rank_loss(out.get("reason_logits_formal_train", out["reason_logits_formal"]), reason),
            "reason_soft_f1": reason_soft_f1_loss(out.get("reason_logits_formal_train", out["reason_logits_formal"]), reason),
            "emission_prior": emission_prior_loss(out.get("emission_prob_learned",out["emission_prob"])),
            "unknown_prior": unknown_prior_loss(out["state_unknown_prob"], unknown_prior),
            "route_sparsity": route_sparsity_loss(out["factor_selection"]),
        }
        if responsibilities is None:
            zero = out["state_prob"].sum() * 0.0
            raw.update({"state": zero, "emission": zero})
        else:
            raw["state"] = state_loss(out["state_prob"], responsibilities["gamma_state_order"])
            raw["emission"] = emission_loss(out.get("emission_prob_learned",out["emission_prob"]), reason, responsibilities["gamma_emission_order"])
        structured = batch.get("structured")
        if structured is None:
            zero = out["evidence_map"].sum() * 0.0
            raw.update({"map_anchor": zero, "state_anchor": zero})
        else:
            raw["map_anchor"] = map_anchor_loss(out["evidence_map"], structured.map_target.to(out["evidence_map"]), structured.map_mask.to(out["evidence_map"]))
            raw["state_anchor"] = state_anchor_loss(out["state_prob"], structured.state_target.to(out["state_prob"]), structured.state_mask.to(out["state_prob"]))
        raw["view_consistency"] = batch.get("view_consistency_loss", out["evidence_map"].sum() * 0.0)
        multipliers=multipliers or {}
        total = sum(self.weights.get(name, 0.0) * multipliers.get(name,1.0) * value for name, value in raw.items())
        return total, raw
