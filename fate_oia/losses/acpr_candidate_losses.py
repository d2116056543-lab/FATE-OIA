from __future__ import annotations

import torch
import torch.nn.functional as F


def candidate_action_asl_loss(candidate_logits: torch.Tensor, targets: torch.Tensor, criterion=None) -> torch.Tensor:
    if criterion is not None:
        return criterion(candidate_logits, targets)
    return F.binary_cross_entropy_with_logits(candidate_logits, targets.float())


def action_candidate_nonregression_loss(
    candidate_logits: torch.Tensor,
    fallback_logits: torch.Tensor,
    targets: torch.Tensor,
    margin: float = 0.0,
) -> torch.Tensor:
    candidate_loss = candidate_action_asl_loss(candidate_logits, targets)
    fallback_loss = candidate_action_asl_loss(fallback_logits.detach(), targets)
    return torch.relu(candidate_loss - fallback_loss + float(margin))


def action_candidate_probe_loss(
    candidate_logits: torch.Tensor,
    fallback_logits: torch.Tensor,
    targets: torch.Tensor,
    candidate_weight: float = 0.5,
    nonreg_weight: float = 0.5,
) -> torch.Tensor:
    base = candidate_action_asl_loss(candidate_logits, targets)
    nonreg = action_candidate_nonregression_loss(candidate_logits, fallback_logits, targets)
    return float(candidate_weight) * base + float(nonreg_weight) * nonreg


def all_candidate_probe_loss(
    candidate_logits_dict: dict[str, torch.Tensor],
    fallback_logits: torch.Tensor,
    targets: torch.Tensor,
    candidate_names: tuple[str, ...] | list[str],
    candidate_weight: float = 0.5,
    nonreg_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    losses = []
    parts: dict[str, float] = {}
    for name in candidate_names:
        logits = candidate_logits_dict[name]
        base = candidate_action_asl_loss(logits, targets)
        nonreg = action_candidate_nonregression_loss(logits, fallback_logits, targets)
        loss = float(candidate_weight) * base + float(nonreg_weight) * nonreg
        losses.append(loss)
        parts[f"loss_candidate_{name}"] = float(base.detach().cpu())
        parts[f"nonreg_candidate_{name}"] = float(nonreg.detach().cpu())
        parts[f"probe_loss_candidate_{name}"] = float(loss.detach().cpu())
    if not losses:
        zero = fallback_logits.sum() * 0.0
        return zero, parts
    return torch.stack(losses).mean(), parts


def gate_sparsity_regularizer(gates: torch.Tensor, weight: float = 0.0) -> torch.Tensor:
    return float(weight) * gates.float().mean()
