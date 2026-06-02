from __future__ import annotations

import torch
from torch import nn


class ActionReasonRouter(nn.Module):
    def __init__(self, action_dim: int = 4, reason_dim: int = 21, dim: int = 384, test_top_k: int = 12) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.reason_dim = reason_dim
        self.test_top_k = test_top_k
        self.scorer = nn.Sequential(nn.Linear(action_dim + 1 + dim, dim), nn.GELU(), nn.Linear(dim, 1))

    def forward(
        self,
        base_action: torch.Tensor,
        base_reason: torch.Tensor,
        label_tokens: torch.Tensor | None,
        reason_targets: torch.Tensor | None = None,
        train_mode: bool = False,
    ) -> dict[str, torch.Tensor]:
        b = base_reason.shape[0]
        if label_tokens is None:
            label_tokens = base_reason.new_zeros(b, self.action_dim + self.reason_dim, self.scorer[0].in_features - self.action_dim - 1)
        reason_tokens = label_tokens[:, self.action_dim : self.action_dim + self.reason_dim]
        action_ctx = base_action.unsqueeze(1).expand(-1, self.reason_dim, -1)
        reason_score = base_reason.unsqueeze(-1)
        scores = self.scorer(torch.cat([action_ctx, reason_score, reason_tokens], dim=-1)).squeeze(-1)
        uncertainty = torch.sigmoid(base_reason) * (1.0 - torch.sigmoid(base_reason))
        scores = scores + 0.25 * uncertainty
        if train_mode and reason_targets is not None:
            k = min(self.reason_dim, max(self.test_top_k, int(reason_targets.sum(1).max().item()) if reason_targets.numel() else self.test_top_k))
            top = torch.topk(scores, k=k, dim=1).indices
            mask = torch.zeros_like(scores, dtype=torch.bool)
            mask.scatter_(1, top, True)
            mask = mask | (reason_targets > 0.5)
            pos = reason_targets > 0.5
            recall = ((mask & pos).sum().float() / pos.sum().clamp_min(1).float()).detach()
        else:
            k = min(self.reason_dim, self.test_top_k)
            top = torch.topk(scores, k=k, dim=1).indices
            mask = torch.zeros_like(scores, dtype=torch.bool)
            mask.scatter_(1, top, True)
            recall = scores.new_tensor(0.0)
        return {
            "active_reason_mask": mask,
            "active_reason_scores": scores,
            "active_reason_recall_train": recall,
            "reason_budget": mask.sum(1).float(),
            "action_uncertainty": torch.sigmoid(base_action) * (1.0 - torch.sigmoid(base_action)),
            "reason_uncertainty": uncertainty,
        }
