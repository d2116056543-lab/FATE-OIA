from __future__ import annotations

import torch
from torch import nn

from fate_oia.losses.asymmetric_loss import AsymmetricLossMultiLabel


class FactorJudge(nn.Module):
    def __init__(self, margin: float = 0.02) -> None:
        super().__init__()
        self.margin = float(margin)
        self.criterion = AsymmetricLossMultiLabel()

    def forward(self, selected_logits: torch.Tensor, without_selected_logits: torch.Tensor, without_random_logits: torch.Tensor, y_action: torch.Tensor) -> dict[str, torch.Tensor]:
        loss_selected = self.criterion(selected_logits, y_action)
        loss_without_selected = self.criterion(without_selected_logits, y_action)
        loss_without_random = self.criterion(without_random_logits, y_action)
        comp = torch.relu(torch.as_tensor(self.margin, device=y_action.device) + loss_without_random - loss_without_selected)
        drop_selected = loss_without_selected.detach() - loss_selected.detach()
        drop_random = loss_without_random.detach() - loss_selected.detach()
        faith = loss_without_selected.detach() - loss_without_random.detach()
        return {
            "loss_sufficiency": loss_selected,
            "loss_comprehensiveness": comp,
            "drop_selected": drop_selected,
            "drop_random": drop_random,
            "faith_score": faith,
            "help_score": torch.relu(drop_selected - drop_random),
            "hurt_score": torch.relu(drop_random - drop_selected),
        }
