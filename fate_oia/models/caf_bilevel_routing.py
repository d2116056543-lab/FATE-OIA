from __future__ import annotations

import torch
from torch import nn

from fate_oia.models.caf_sparse_selection import sparsemax


class BiLevelFactorRouter(nn.Module):
    def __init__(self, dim: int = 384, action_dim: int = 4, num_groups: int = 7, factor_topk: int = 3, group_topk: int = 3, lambda_exp_init: float = 0.0, lambda_exp_max: float = 0.30) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.num_groups = int(num_groups)
        self.factor_topk = int(factor_topk)
        self.group_topk = int(group_topk)
        self.lambda_exp_max = float(lambda_exp_max)
        self.group_mlp = nn.Sequential(nn.Linear(dim * 2 + 1, dim), nn.GELU(), nn.Linear(dim, 1))
        self.factor_mlp = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, 1))
        self.register_buffer("faith_ema", torch.zeros(action_dim, num_groups))
        self.register_buffer("help_ema", torch.zeros(action_dim, num_groups))
        self.register_buffer("hurt_ema", torch.zeros(action_dim, num_groups))
        self.register_buffer("lambda_exp", torch.full((action_dim, num_groups), float(lambda_exp_init)))

    def update_reliability(
        self,
        selected_vs_random_per_action_group: torch.Tensor | float,
        help_delta_per_action_group: torch.Tensor | None = None,
        hurt_delta_per_action_group: torch.Tensor | None = None,
    ) -> None:
        if torch.is_tensor(selected_vs_random_per_action_group):
            value = selected_vs_random_per_action_group.detach().to(self.faith_ema.device).float()
            if value.numel() == 1:
                value = value.expand_as(self.faith_ema)
            else:
                value = value.reshape_as(self.faith_ema)
        else:
            value = torch.full_like(self.faith_ema, float(selected_vs_random_per_action_group))
        help_delta = value if help_delta_per_action_group is None else help_delta_per_action_group.detach().to(self.faith_ema.device).float().reshape_as(self.faith_ema)
        hurt_delta = torch.relu(-value) if hurt_delta_per_action_group is None else hurt_delta_per_action_group.detach().to(self.faith_ema.device).float().reshape_as(self.faith_ema)
        momentum = 0.9
        self.faith_ema.mul_(momentum).add_(torch.relu(value) * (1.0 - momentum))
        self.help_ema.mul_(momentum).add_(torch.relu(help_delta) * (1.0 - momentum))
        self.hurt_ema.mul_(momentum).add_(torch.relu(hurt_delta) * (1.0 - momentum))
        updated = self.help_ema + self.faith_ema - self.hurt_ema
        self.lambda_exp.copy_(torch.clamp(updated, 0.0, self.lambda_exp_max))

    def forward(self, action_tokens: torch.Tensor, factor_tokens: torch.Tensor, factor_group_ids: torch.Tensor, action_uncertainty: torch.Tensor | None = None, exp_prior: torch.Tensor | None = None, exp_reliability: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        b, a, d = action_tokens.shape
        m = factor_tokens.shape[1]
        if action_uncertainty is None:
            action_uncertainty = action_tokens.new_zeros(b, a)
        group_tokens = []
        for g in range(self.num_groups):
            mask = (factor_group_ids == g).float().unsqueeze(-1)
            denom = mask.sum(1).clamp_min(1.0)
            group_tokens.append((factor_tokens * mask).sum(1) / denom)
        group_tokens = torch.stack(group_tokens, dim=1)
        action_expand = action_tokens.unsqueeze(2).expand(b, a, self.num_groups, d)
        group_expand = group_tokens.unsqueeze(1).expand(b, a, self.num_groups, d)
        unc = action_uncertainty.unsqueeze(-1).unsqueeze(-1).expand(b, a, self.num_groups, 1)
        group_scores = self.group_mlp(torch.cat([action_expand, group_expand, unc], dim=-1)).squeeze(-1)
        top_groups = torch.topk(group_scores, k=min(self.group_topk, self.num_groups), dim=-1).indices
        group_mask = torch.zeros_like(group_scores).scatter_(-1, top_groups, 1.0)
        factor_group_for_action = factor_group_ids.unsqueeze(1).expand(b, a, m)
        allowed = torch.gather(group_mask, 2, factor_group_for_action)
        act = action_tokens.unsqueeze(2).expand(b, a, m, d)
        fac = factor_tokens.unsqueeze(1).expand(b, a, m, d)
        visual_scores = self.factor_mlp(torch.cat([act, fac], dim=-1)).squeeze(-1)
        weak_exp_scores = torch.zeros_like(visual_scores)
        if exp_prior is not None:
            weak_exp_scores = weak_exp_scores + exp_prior.to(visual_scores.device)
        if exp_reliability is not None:
            weak_exp_scores = weak_exp_scores * exp_reliability.to(visual_scores.device)
        lambda_group = torch.clamp(self.lambda_exp, 0.0, self.lambda_exp_max).unsqueeze(0).expand(b, -1, -1)
        lambda_factor = torch.gather(lambda_group, 2, factor_group_for_action.clamp(0, self.num_groups - 1))
        scores = visual_scores + lambda_factor * weak_exp_scores
        scores = scores.masked_fill(allowed <= 0, -1e4)
        sparse_weights = sparsemax(scores, dim=-1)
        top = torch.topk(sparse_weights, k=min(self.factor_topk, m), dim=-1)
        return {
            "group_scores": group_scores,
            "factor_group_scores": group_scores,
            "factor_scores": scores,
            "visual_scores": visual_scores,
            "weak_exp_scores": weak_exp_scores,
            "lambda_exp": torch.clamp(self.lambda_exp, 0.0, self.lambda_exp_max).detach(),
            "lambda_factor": lambda_factor.detach(),
            "faith_ema": self.faith_ema.detach().clone(),
            "help_ema": self.help_ema.detach().clone(),
            "hurt_ema": self.hurt_ema.detach().clone(),
            "selected_factor_indices": top.indices,
            "selected_factor_weights": top.values,
            "sparse_weights": sparse_weights,
        }
