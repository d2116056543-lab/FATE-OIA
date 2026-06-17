from __future__ import annotations

import torch


def fusionlite_delta_l2(delta_gate: torch.Tensor) -> torch.Tensor:
    return delta_gate.pow(2).mean()


def r2a_forbidden_prior_loss(reason_to_action_weight: torch.Tensor, forbidden_mask: torch.Tensor) -> torch.Tensor:
    return (reason_to_action_weight * forbidden_mask.to(reason_to_action_weight.device, reason_to_action_weight.dtype)).pow(2).mean()


def action_primary_cooldown_should_trigger(train_calib_scores: list[float], patience: int = 2, min_delta: float = 1e-4) -> bool:
    if len(train_calib_scores) <= patience:
        return False
    best_before = max(train_calib_scores[: -patience])
    recent_best = max(train_calib_scores[-patience:])
    return recent_best <= best_before + float(min_delta)
