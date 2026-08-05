from __future__ import annotations

import torch
import torch.nn.functional as F


def map_anchor_loss(evidence_map: torch.Tensor, map_target: torch.Tensor, map_mask: torch.Tensor) -> torch.Tensor:
    if not map_mask.bool().any(): return evidence_map.sum() * 0.0
    return F.kl_div(evidence_map.clamp_min(1e-8).log(), map_target.clamp_min(1e-8), reduction="none").sum(-1).mul(map_mask).sum() / map_mask.sum().clamp_min(1.0)


def state_anchor_loss(state_prob: torch.Tensor, state_target: torch.Tensor, state_mask: torch.Tensor) -> torch.Tensor:
    if not state_mask.bool().any(): return state_prob.sum() * 0.0
    return F.kl_div(state_prob.clamp_min(1e-8).log(), state_target, reduction="none").sum(-1).mul(state_mask).sum() / state_mask.sum().clamp_min(1.0)


def view_consistency_loss(reference: torch.Tensor, augmented: torch.Tensor) -> torch.Tensor: return F.mse_loss(reference.sigmoid(), augmented.sigmoid())
def unknown_prior_loss(unknown: torch.Tensor, prior: torch.Tensor) -> torch.Tensor: return F.mse_loss(unknown.mean(0), prior)
def route_sparsity_loss(selection: torch.Tensor) -> torch.Tensor: return -(selection.clamp_min(1e-8) * selection.clamp_min(1e-8).log()).sum(-1).mean()
