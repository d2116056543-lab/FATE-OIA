from __future__ import annotations

import torch
import torch.nn.functional as F

from fate_oia.losses.asymmetric_loss import asymmetric_loss_with_logits
from fate_oia.losses.acpr_threshold_losses import calalign_loss_bundle


def predicate_measurement_loss(q_pred: torch.Tensor, rho_pred: torch.Tensor, observations: dict, *, weights: dict[str, float] | None = None) -> tuple[torch.Tensor, dict[str, float]]:
    weights = weights or {}
    loss = q_pred.sum() * 0.0
    terms: dict[str, torch.Tensor] = {}
    for prefix in ["obs_reason", "obs_geometry"]:
        value = observations.get(f"{prefix}_value")
        mask = observations.get(f"{prefix}_mask")
        if torch.is_tensor(value) and torch.is_tensor(mask) and mask.sum() > 0:
            reliability = observations.get(f"{prefix}_reliability")
            if not torch.is_tensor(reliability):
                reliability = torch.ones_like(mask)
            bce = F.binary_cross_entropy(q_pred.clamp(1e-5, 1 - 1e-5), value.float(), reduction="none")
            term = (bce * mask.float() * reliability.float().clamp(0.0, 1.0)).sum() / (mask.float() * reliability.float()).sum().clamp_min(1.0)
        else:
            term = q_pred.sum() * 0.0
        terms[prefix] = term
        loss = loss + term
    rho_entropy = -(rho_pred.clamp(1e-5, 1 - 1e-5).log() * rho_pred + (1 - rho_pred).clamp(1e-5, 1).log() * (1 - rho_pred)).mean()
    loss = loss + 0.01 * rho_entropy
    return loss, {k: float(v.detach().cpu()) for k, v in terms.items()} | {"rho_entropy": float(rho_entropy.detach().cpu())}


def pu_reason_asl_loss(logits_deploy: torch.Tensor, pu_state: dict, *, tail_indices: list[int] | None = None, weights: dict[str, float] | None = None) -> tuple[torch.Tensor, dict[str, float]]:
    pos = pu_state["positive_mask"].float()
    neg = pu_state["reliable_negative_mask"].float()
    unk = pu_state["unknown_mask"].float()
    target = pos
    weight = pos + 0.5 * neg
    loss = F.binary_cross_entropy_with_logits(logits_deploy, target, weight=weight, reduction="sum") / weight.sum().clamp_min(1.0)
    unk_probs = torch.sigmoid(logits_deploy)
    unk_entropy = (unk * (unk_probs - 0.5).abs()).sum() / unk.sum().clamp_min(1.0)
    loss = loss + 0.02 * unk_entropy
    return loss, {
        "pu_positive_rate": float(pos.mean().detach().cpu()),
        "pu_negative_rate": float(neg.mean().detach().cpu()),
        "pu_unknown_rate": float(unk.mean().detach().cpu()),
        "pu_unknown_entropy": float(unk_entropy.detach().cpu()),
    }


def action_asl_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return asymmetric_loss_with_logits(logits, target.float(), gamma_neg=4, gamma_pos=0, clip=0.05)


def formula_reason_consistency_loss(reason_logits: torch.Tensor, formula_logits: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return (gate.detach() * F.smooth_l1_loss(reason_logits, formula_logits.detach(), reduction="none")).mean()


def action_predicate_consistency_loss(action_logits_deploy: torch.Tensor, action_targets: torch.Tensor, q_pred: torch.Tensor, compatible_action: torch.Tensor | None = None) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(action_logits_deploy, action_targets.float())


def reliability_regularizer(rho_pred: torch.Tensor) -> torch.Tensor:
    return F.relu(0.05 - rho_pred.mean()) + F.relu(rho_pred.mean() - 0.95)


def predicate_attention_compactness_loss(predicate_attention: torch.Tensor) -> torch.Tensor:
    entropy = -(predicate_attention.clamp_min(1e-8).log() * predicate_attention).sum(-1)
    return entropy.mean() / predicate_attention.shape[-1]


__all__ = [
    "predicate_measurement_loss",
    "pu_reason_asl_loss",
    "action_asl_loss",
    "formula_reason_consistency_loss",
    "action_predicate_consistency_loss",
    "reliability_regularizer",
    "predicate_attention_compactness_loss",
    "calalign_loss_bundle",
]
