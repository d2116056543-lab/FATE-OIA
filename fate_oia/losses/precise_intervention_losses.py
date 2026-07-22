from __future__ import annotations

import torch
from torch.nn import functional as F


def target_specific_intervention_loss(selected_effect: torch.Tensor, control_effect: torch.Tensor, wrong_effect: torch.Tensor, base_logits: torch.Tensor, intervened_logits: torch.Tensor, targets: torch.Tensor, margin: float = 0.10, nonreg_delta: float = 0.02) -> dict[str, torch.Tensor]:
    direct = (margin + control_effect - selected_effect).relu().mean()
    specific = (margin + wrong_effect - selected_effect).relu().mean()
    base = F.binary_cross_entropy_with_logits(base_logits.detach(), targets.float(), reduction="none")
    intervened = F.binary_cross_entropy_with_logits(intervened_logits, targets.float(), reduction="none")
    nonreg = (intervened - base - nonreg_delta).relu().mean()
    return {"loss_intervention_direct": direct, "loss_intervention_specific": specific, "loss_intervention_nonreg": nonreg, "loss_intervention": 0.10 * direct + 0.05 * specific + 0.05 * nonreg}


def matched_control_is_valid(selected_mask: torch.Tensor, control_mask: torch.Tensor, family_equal: torch.Tensor, sector_equal: torch.Tensor, part_equal: torch.Tensor, tolerance: float = 0.05) -> torch.Tensor:
    selected_mass = selected_mask.flatten(1).sum(1)
    control_mass = control_mask.flatten(1).sum(1)
    mass_error = (selected_mass - control_mass).abs() / selected_mass.clamp_min(1e-6)
    overlap = (selected_mask * control_mask).flatten(1).sum(1)
    return family_equal & sector_equal & part_equal & (mass_error <= tolerance) & (overlap == 0)
