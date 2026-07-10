from __future__ import annotations

import torch


def build_mosaic_state_loss(
    state_output: dict[str, torch.Tensor],
    *,
    sparsity_weight: float = 0.02,
    residual_weight: float = 0.02,
    uncertainty_weight: float = 0.02,
) -> dict[str, torch.Tensor]:
    if any(weight < 0 for weight in (sparsity_weight, residual_weight, uncertainty_weight)):
        raise ValueError("state loss weights must be non-negative")
    required = {
        "decision_state_prob",
        "decision_state_residual",
        "decision_state_uncertainty",
    }
    if not required <= set(state_output):
        raise KeyError(f"state output missing {sorted(required - set(state_output))}")
    probability = state_output["decision_state_prob"]
    residual = state_output["decision_state_residual"]
    uncertainty = state_output["decision_state_uncertainty"]
    if probability.shape != residual.shape or probability.shape != uncertainty.shape:
        raise ValueError("state loss tensors must have matching shapes")
    sparsity = probability.mean()
    residual_loss = residual.abs().mean()
    uncertainty_loss = uncertainty.mean()
    total = (
        sparsity_weight * sparsity
        + residual_weight * residual_loss
        + uncertainty_weight * uncertainty_loss
    )
    count = probability.new_tensor(probability.numel(), dtype=torch.long)
    return {
        "loss_state_sparsity": sparsity,
        "loss_state_residual": residual_loss,
        "loss_state_uncertainty": uncertainty_loss,
        "loss_state_total": total,
        "count_state_sparsity": count,
        "count_state_residual": count,
        "count_state_uncertainty": count,
    }
