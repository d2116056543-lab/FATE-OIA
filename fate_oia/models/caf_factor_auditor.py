from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class CriticalFactorAuditor(nn.Module):
    def __init__(self, margin: float = 0.02) -> None:
        super().__init__()
        self.margin = float(margin)

    def forward(self, z_fate: torch.Tensor, z_actor: torch.Tensor, gate: torch.Tensor, delta: torch.Tensor, selected_weights: torch.Tensor, y_action: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if y_action is None:
            zero = z_actor.new_zeros(())
            return {"loss": zero, "drop_selected": zero, "drop_random": zero, "selected_vs_random_action_loss_drop": zero}
        strength = selected_weights.mean(dim=-1).clamp(0.0, 1.0)
        masked_delta_selected = delta * (1.0 - strength)
        rand_strength = torch.roll(strength, shifts=1, dims=1)
        masked_delta_random = delta * (1.0 - rand_strength)
        z_without_selected = z_fate + gate * masked_delta_selected
        z_without_random = z_fate + gate * masked_delta_random
        full = F.binary_cross_entropy_with_logits(z_actor, y_action.float())
        loss_selected = F.binary_cross_entropy_with_logits(z_without_selected, y_action.float())
        loss_random = F.binary_cross_entropy_with_logits(z_without_random, y_action.float())
        drop_selected = loss_selected - full
        drop_random = loss_random - full
        loss = torch.relu(z_actor.new_tensor(self.margin) + drop_random - drop_selected)
        return {
            "loss": loss,
            "drop_selected": drop_selected.detach(),
            "drop_random": drop_random.detach(),
            "selected_vs_random_action_loss_drop": (drop_selected - drop_random).detach(),
            "z_actor_without_selected": z_without_selected.detach(),
            "z_actor_without_random": z_without_random.detach(),
        }
