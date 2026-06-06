from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class CriticalFactorAuditor(nn.Module):
    def __init__(self, margin: float = 0.02) -> None:
        super().__init__()
        self.margin = float(margin)

    def forward(
        self,
        z_actor_full: torch.Tensor,
        z_actor_without_selected: torch.Tensor,
        z_actor_without_random: torch.Tensor,
        y_action: torch.Tensor | None = None,
        per_action_group: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | str]:
        if y_action is None:
            zero = z_actor_full.new_zeros(())
            return {
                "method": "action_gt_loss_drop_on_evidence_mask",
                "loss": zero,
                "drop_selected": zero,
                "drop_random": zero,
                "selected_vs_random_action_loss_drop": zero,
            }
        full_per_action = F.binary_cross_entropy_with_logits(z_actor_full, y_action.float(), reduction="none")
        selected_per_action = F.binary_cross_entropy_with_logits(z_actor_without_selected, y_action.float(), reduction="none")
        random_per_action = F.binary_cross_entropy_with_logits(z_actor_without_random, y_action.float(), reduction="none")
        full = full_per_action.mean()
        loss_selected = selected_per_action.mean()
        loss_random = random_per_action.mean()
        drop_selected = loss_selected - full
        drop_random = loss_random - full
        per_action_drop_selected = (selected_per_action - full_per_action).detach().mean(0)
        per_action_drop_random = (random_per_action - full_per_action).detach().mean(0)
        loss = torch.relu(z_actor_full.new_tensor(self.margin) + drop_random - drop_selected)
        return {
            "method": "action_gt_loss_drop_on_evidence_mask",
            "loss": loss,
            "drop_selected": drop_selected.detach(),
            "drop_random": drop_random.detach(),
            "selected_vs_random_action_loss_drop": (drop_selected - drop_random).detach(),
            "per_action_drop_selected": per_action_drop_selected,
            "per_action_drop_random": per_action_drop_random,
            "per_action_selected_minus_random": per_action_drop_selected - per_action_drop_random,
            "per_action_group_selected_minus_random": per_action_group.detach() if per_action_group is not None else None,
            "z_actor_without_selected": z_actor_without_selected.detach(),
            "z_actor_without_random": z_actor_without_random.detach(),
        }
