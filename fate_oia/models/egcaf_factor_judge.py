from __future__ import annotations

import torch
from torch import nn

from fate_oia.losses.asymmetric_loss import AsymmetricLossMultiLabel


class FactorJudge(nn.Module):
    """Judge selected factors by action-GT loss drops, not logit means."""

    def __init__(self, margin: float = 0.02) -> None:
        super().__init__()
        self.margin = float(margin)
        self.criterion = AsymmetricLossMultiLabel()

    def forward(
        self,
        all_logits: torch.Tensor,
        selected_logits: torch.Tensor,
        without_selected_logits: torch.Tensor,
        without_random_logits: torch.Tensor | None = None,
        y_action: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # Backward compatible four-argument form: selected, without_selected, without_random, y.
        if y_action is None:
            if without_random_logits is None:
                raise ValueError("y_action is required")
            y_action = without_random_logits
            without_random_logits = without_selected_logits
            without_selected_logits = selected_logits
            selected_logits = all_logits
            all_logits = selected_logits
        loss_all = self.criterion(all_logits, y_action)
        loss_selected = self.criterion(selected_logits, y_action)
        loss_without_selected = self.criterion(without_selected_logits, y_action)
        loss_without_random = self.criterion(without_random_logits, y_action)
        suff = torch.relu(loss_selected - loss_all + self.margin)
        comp = torch.relu(self.margin + loss_without_random - loss_without_selected)
        drop_selected = loss_without_selected.detach() - loss_all.detach()
        drop_random = loss_without_random.detach() - loss_all.detach()
        selected_vs_random = drop_selected - drop_random
        return {
            "loss_all": loss_all,
            "loss_sufficiency": suff,
            "loss_comprehensiveness": comp,
            "drop_selected_loss": drop_selected,
            "drop_random_loss": drop_random,
            "selected_vs_random_action_loss_drop": selected_vs_random,
            "drop_selected": drop_selected,
            "drop_random": drop_random,
            "faith_score": selected_vs_random,
            "help_score": torch.relu(selected_vs_random),
            "hurt_score": torch.relu(-selected_vs_random),
        }
