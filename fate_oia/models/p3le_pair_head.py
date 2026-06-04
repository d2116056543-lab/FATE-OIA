from __future__ import annotations

import torch
from torch import nn


class PairAwareTensorHead(nn.Module):
    """Low-rank action-reason tensor H[B, A, R]."""

    def __init__(self, dim: int = 384, action_dim: int = 4, reason_dim: int = 21, rank: int = 32) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.reason_dim = int(reason_dim)
        self.rank = int(rank)
        self.action_proj = nn.Linear(dim, rank)
        self.reason_proj = nn.Linear(dim, rank)
        self.context_proj = nn.Linear(dim, rank)
        self.bias = nn.Parameter(torch.zeros(action_dim, reason_dim))
        self.action_support = nn.Linear(action_dim, action_dim)
        self.reason_support = nn.Linear(reason_dim, reason_dim)

    def forward(self, action_tokens: torch.Tensor, reason_tokens: torch.Tensor, shared_context: torch.Tensor) -> dict[str, torch.Tensor]:
        a = self.action_proj(action_tokens)
        r = self.reason_proj(reason_tokens)
        c = self.context_proj(shared_context).unsqueeze(1).unsqueeze(2)
        h = (a.unsqueeze(2) * r.unsqueeze(1) * c).sum(dim=-1) / (self.rank ** 0.5)
        h = h + self.bias.unsqueeze(0)
        pair_action_support = self.action_support(h.mean(dim=2))
        pair_reason_support = self.reason_support(h.mean(dim=1))
        return {
            "pair_tensor": h,
            "pair_action_support": pair_action_support,
            "pair_reason_support": pair_reason_support,
            "pair_tensor_mean": h.mean(),
            "pair_tensor_std": h.std(unbiased=False),
        }


def build_pair_seed_targets(action: torch.Tensor, reason: torch.Tensor, action_dim: int = 4, reason_dim: int = 21) -> torch.Tensor:
    """Soft low-risk seeds only, not full positive action x positive reason labels."""
    seeds = action.new_zeros(action.shape[0], action_dim, reason_dim)
    # stop, left, right, forward groups. These are soft priors for regularization.
    groups = {
        1: [0, 3, 4, 7, 15, 18],
        2: [10, 11, 13, 16],
        3: [10, 11, 13, 17],
        0: [1, 2, 8, 19, 20],
    }
    for action_idx, reason_indices in groups.items():
        if action_idx >= action_dim:
            continue
        valid_reasons = [idx for idx in reason_indices if idx < reason_dim]
        if not valid_reasons:
            continue
        seeds[:, action_idx, valid_reasons] = action[:, action_idx].unsqueeze(1) * reason[:, valid_reasons]
    return seeds
