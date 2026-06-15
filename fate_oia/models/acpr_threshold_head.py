from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import nn


def _as_logit(prob: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return torch.logit(prob.clamp(eps, 1.0 - eps))


class ACPRThresholdHead(nn.Module):
    """Learn train-calib threshold offsets for deployment-time fixed 0.5 logits.

    The deployed decision is sigmoid(base_logit - theta_l) > 0.5, equivalent to
    sigmoid(base_logit) > sigmoid(theta_l). Group shrinkage keeps rare labels
    from learning unconstrained per-label thresholds.
    """

    def __init__(
        self,
        action_dim: int = 4,
        reason_dim: int = 21,
        label_group_ids: torch.Tensor | None = None,
        label_delta_scale: torch.Tensor | None = None,
        action_threshold_min: float = 0.10,
        action_threshold_max: float = 0.90,
        reason_threshold_min: float = 0.02,
        reason_threshold_max: float = 0.85,
        tail_reason_indices: list[int] | None = None,
        tail_reason_threshold_min: float = 0.01,
        tail_reason_threshold_max: float = 0.65,
        use_group_shrinkage: bool = True,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.reason_dim = int(reason_dim)
        self.num_labels = self.action_dim + self.reason_dim
        self.use_group_shrinkage = bool(use_group_shrinkage)
        tail_reason_indices = tail_reason_indices or [12, 9, 5, 14, 6, 11, 10, 13]

        if label_group_ids is None:
            # Group 0: actions, group 1: common reasons, group 2: tail reasons.
            group_ids = torch.ones(self.num_labels, dtype=torch.long)
            group_ids[: self.action_dim] = 0
            for rid in tail_reason_indices:
                if 0 <= rid < self.reason_dim:
                    group_ids[self.action_dim + rid] = 2
        else:
            group_ids = label_group_ids.long().view(-1)
            if group_ids.numel() != self.num_labels:
                raise ValueError(f"label_group_ids must have {self.num_labels} entries")

        if label_delta_scale is None:
            delta_scale = torch.ones(self.num_labels, dtype=torch.float32)
            for rid in tail_reason_indices:
                if 0 <= rid < self.reason_dim:
                    delta_scale[self.action_dim + rid] = 0.5
        else:
            delta_scale = label_delta_scale.float().view(-1)
            if delta_scale.numel() != self.num_labels:
                raise ValueError(f"label_delta_scale must have {self.num_labels} entries")

        min_prob = torch.full((self.num_labels,), float(reason_threshold_min))
        max_prob = torch.full((self.num_labels,), float(reason_threshold_max))
        min_prob[: self.action_dim] = float(action_threshold_min)
        max_prob[: self.action_dim] = float(action_threshold_max)
        for rid in tail_reason_indices:
            if 0 <= rid < self.reason_dim:
                min_prob[self.action_dim + rid] = float(tail_reason_threshold_min)
                max_prob[self.action_dim + rid] = float(tail_reason_threshold_max)

        self.register_buffer("label_group_ids", group_ids)
        self.register_buffer("label_delta_scale", delta_scale)
        self.register_buffer("theta_teacher", torch.zeros(self.num_labels))
        self.register_buffer("teacher_pred_rate", torch.zeros(self.num_labels))
        self.register_buffer("train_prior_theta", torch.zeros(self.num_labels))
        self.register_buffer("threshold_min_logit", _as_logit(min_prob))
        self.register_buffer("threshold_max_logit", _as_logit(max_prob))

        num_groups = int(group_ids.max().item()) + 1
        self.theta_group = nn.Parameter(torch.zeros(num_groups))
        self.theta_delta = nn.Parameter(torch.zeros(self.num_labels))
        self.log_temperature = nn.Parameter(torch.zeros(self.num_labels))

    def compose_theta(self) -> torch.Tensor:
        if self.use_group_shrinkage:
            theta = self.theta_group[self.label_group_ids] + self.label_delta_scale * self.theta_delta
        else:
            theta = self.theta_delta
        return torch.max(torch.min(theta, self.threshold_max_logit), self.threshold_min_logit)

    def forward(
        self,
        action_logits_base: torch.Tensor,
        reason_logits_base: torch.Tensor,
        apply_temperature: bool = True,
    ) -> dict[str, torch.Tensor]:
        base = torch.cat([action_logits_base, reason_logits_base], dim=-1)
        theta = self.compose_theta()
        deploy = base - theta.view(1, -1)
        temperature = torch.exp(self.log_temperature).clamp(0.5, 3.0)
        calibrated = deploy / temperature.view(1, -1) if apply_temperature else deploy
        return {
            "logits_base": base,
            "logits_deploy": deploy,
            "logits_calibrated": calibrated,
            "action_logits_deploy": deploy[:, : self.action_dim],
            "reason_logits_deploy": deploy[:, self.action_dim :],
            "action_logits_calibrated": calibrated[:, : self.action_dim],
            "reason_logits_calibrated": calibrated[:, self.action_dim :],
            "threshold_logit": theta,
            "threshold_prob": torch.sigmoid(theta),
            "temperature": temperature,
            "action_threshold_prob": torch.sigmoid(theta[: self.action_dim]),
            "reason_threshold_prob": torch.sigmoid(theta[self.action_dim :]),
        }

    @torch.no_grad()
    def initialize_from_label_stats(
        self,
        action_pos_rate: torch.Tensor,
        reason_pos_rate: torch.Tensor,
        reason_groups: Iterable[int] | None = None,
    ) -> None:
        action_pos_rate = action_pos_rate.float().view(self.action_dim)
        reason_pos_rate = reason_pos_rate.float().view(self.reason_dim)
        action_prior = torch.full_like(action_pos_rate, 0.50)
        median_reason = reason_pos_rate[reason_pos_rate > 0].median() if (reason_pos_rate > 0).any() else torch.tensor(0.05, device=reason_pos_rate.device)
        ratio = torch.sqrt((reason_pos_rate + 1e-5) / (median_reason + 1e-5))
        reason_prior = (0.50 * ratio).clamp(0.03, 0.65)
        # Tail labels get a lower ceiling to avoid requiring common-label confidence.
        tail_mask = (self.threshold_max_logit[self.action_dim :] <= math.log(0.65 / 0.35) + 1e-6).to(reason_prior.device)
        reason_prior = torch.where(tail_mask, reason_prior.clamp(0.01, 0.50), reason_prior)
        theta_prior = _as_logit(torch.cat([action_prior, reason_prior]).to(self.theta_delta.device))
        self.train_prior_theta.copy_(theta_prior)
        self.theta_teacher.copy_(theta_prior)
        self.teacher_pred_rate.copy_(torch.cat([action_pos_rate, reason_pos_rate]).to(self.teacher_pred_rate.device))

        for gid in torch.unique(self.label_group_ids).tolist():
            mask = self.label_group_ids == int(gid)
            group_mean = theta_prior[mask].mean()
            self.theta_group[int(gid)].copy_(group_mean)
            scale = self.label_delta_scale[mask].clamp_min(1e-4)
            self.theta_delta[mask].copy_((theta_prior[mask] - group_mean) / scale)

    @torch.no_grad()
    def update_teacher(
        self,
        theta_teacher: torch.Tensor,
        pred_rate_teacher: torch.Tensor | None = None,
        ema: float = 0.20,
        copy_to_params: bool = False,
    ) -> None:
        theta_teacher = theta_teacher.to(self.theta_teacher.device, self.theta_teacher.dtype).view(self.num_labels)
        self.theta_teacher.mul_(1.0 - float(ema)).add_(theta_teacher, alpha=float(ema))
        if pred_rate_teacher is not None:
            pred_rate_teacher = pred_rate_teacher.to(self.teacher_pred_rate.device, self.teacher_pred_rate.dtype).view(self.num_labels)
            self.teacher_pred_rate.mul_(1.0 - float(ema)).add_(pred_rate_teacher, alpha=float(ema))
        if copy_to_params:
            target = self.theta_teacher.clamp(self.threshold_min_logit, self.threshold_max_logit)
            for gid in torch.unique(self.label_group_ids).tolist():
                mask = self.label_group_ids == int(gid)
                group_mean = target[mask].mean()
                self.theta_group[int(gid)].copy_(group_mean)
                scale = self.label_delta_scale[mask].clamp_min(1e-4)
                self.theta_delta[mask].copy_((target[mask] - group_mean) / scale)
