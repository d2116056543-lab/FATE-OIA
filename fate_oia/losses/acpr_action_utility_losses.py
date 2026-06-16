from __future__ import annotations

import torch
import torch.nn.functional as F


def action_utility_nonregression_loss(action_logits_utility: torch.Tensor, action_logits_fallback: torch.Tensor, action_targets: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
    """Penalize utility only when it increases per-action BCE versus fallback."""
    util = F.binary_cross_entropy_with_logits(action_logits_utility, action_targets.float(), reduction="none")
    base = F.binary_cross_entropy_with_logits(action_logits_fallback.detach(), action_targets.float(), reduction="none")
    return F.relu(util - base + float(margin)).mean()


def action_delta_regularizer(r2a_delta: torch.Tensor, pred_delta: torch.Tensor) -> torch.Tensor:
    return r2a_delta.pow(2).mean() + pred_delta.pow(2).mean()


def action_gate_sparsity_loss(r2a_gate: torch.Tensor, pred_gate: torch.Tensor) -> torch.Tensor:
    return r2a_gate.abs().mean() + pred_gate.abs().mean()
