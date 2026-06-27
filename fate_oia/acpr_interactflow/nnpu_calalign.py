from __future__ import annotations

import torch
from torch import nn


def nnpu_binary_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, positive_prior: float = 0.1, beta: float = 0.0) -> torch.Tensor:
    pos = targets.float() * mask.float()
    unlabeled = (1.0 - targets.float()) * mask.float()
    loss_pos = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.ones_like(logits), reduction="none")
    loss_neg = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits), reduction="none")
    pos_risk = positive_prior * (loss_pos * pos).sum() / pos.sum().clamp_min(1.0)
    neg_risk = (loss_neg * unlabeled).sum() / unlabeled.sum().clamp_min(1.0) - positive_prior * (loss_neg * pos).sum() / pos.sum().clamp_min(1.0)
    return pos_risk + torch.clamp(neg_risk, min=beta)


class NNPUCalAlignHead(nn.Module):
    def __init__(self, action_dim: int = 4, exp_dim: int = 29) -> None:
        super().__init__()
        self.action_temp = nn.Parameter(torch.zeros(action_dim))
        self.action_bias = nn.Parameter(torch.zeros(action_dim))
        self.exp_temp = nn.Parameter(torch.zeros(exp_dim))
        self.exp_bias = nn.Parameter(torch.zeros(exp_dim))

    def forward(self, action_logits: torch.Tensor, exp_logits: torch.Tensor) -> dict[str, torch.Tensor]:
        action_scale = torch.exp(self.action_temp.clamp(-1.0, 1.0)).view(1, -1)
        exp_scale = torch.exp(self.exp_temp.clamp(-1.0, 1.0)).view(1, -1)
        return {
            "action_logits_calibrated": action_logits / action_scale + self.action_bias.clamp(-2, 2).view(1, -1),
            "exp29_logits_calibrated": exp_logits / exp_scale + self.exp_bias.clamp(-2, 2).view(1, -1),
            "action_temperature": action_scale.detach(),
            "exp29_temperature": exp_scale.detach(),
        }

