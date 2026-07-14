from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _probability_logit(value: torch.Tensor) -> torch.Tensor:
    return torch.logit(value.clamp(1e-6, 1.0 - 1e-6))


class MOSAICGroupThresholdHead(nn.Module):
    def __init__(
        self,
        *,
        action_dim: int = 4,
        reason_dim: int = 21,
        tail_reason_indices: list[int] | None = None,
        label_delta_max: float = 1.0,
    ) -> None:
        super().__init__()
        if action_dim != 4 or reason_dim != 21:
            raise ValueError("MOSAIC threshold head requires 4 action and 21 reason labels")
        if not 0.0 < label_delta_max <= 2.0:
            raise ValueError("label_delta_max must be in (0,2]")
        tail_reason_indices = tail_reason_indices or [12, 9, 5, 14, 6, 11, 10, 13]
        if any(type(index) is not int or index not in range(reason_dim) for index in tail_reason_indices):
            raise ValueError("tail reason indices must be valid exact integers")
        if len(set(tail_reason_indices)) != len(tail_reason_indices):
            raise ValueError("tail reason indices must be unique")

        self.action_dim = action_dim
        self.reason_dim = reason_dim
        self.num_labels = action_dim + reason_dim
        self.label_delta_max = float(label_delta_max)
        group_ids = torch.ones(self.num_labels, dtype=torch.long)
        group_ids[:action_dim] = 0
        for reason_id in tail_reason_indices:
            group_ids[action_dim + reason_id] = 2
        self.register_buffer("label_group_ids", group_ids, persistent=True)

        min_probability = torch.full((self.num_labels,), 0.02)
        max_probability = torch.full((self.num_labels,), 0.85)
        min_probability[:action_dim] = 0.10
        max_probability[:action_dim] = 0.90
        for reason_id in tail_reason_indices:
            min_probability[action_dim + reason_id] = 0.01
            max_probability[action_dim + reason_id] = 0.65
        self.register_buffer("threshold_min_logit", _probability_logit(min_probability), persistent=True)
        self.register_buffer("threshold_max_logit", _probability_logit(max_probability), persistent=True)
        self.theta_group = nn.Parameter(torch.zeros(3))
        self.theta_delta = nn.Parameter(torch.zeros(self.num_labels))

    @property
    def label_delta(self) -> torch.Tensor:
        return self.label_delta_max * torch.tanh(self.theta_delta)

    def compose_theta(self) -> torch.Tensor:
        theta = self.theta_group[self.label_group_ids] + self.label_delta
        return torch.maximum(torch.minimum(theta, self.threshold_max_logit), self.threshold_min_logit)

    def forward(self, action_logits_raw: torch.Tensor, reason_logits_raw: torch.Tensor) -> dict[str, torch.Tensor]:
        if action_logits_raw.ndim != 2 or reason_logits_raw.ndim != 2:
            raise ValueError("threshold head expects rank-2 raw logits")
        if action_logits_raw.shape[0] != reason_logits_raw.shape[0] or action_logits_raw.shape[1] != 4 or reason_logits_raw.shape[1] != 21:
            raise ValueError("threshold head expects action [B,4] and reason [B,21]")
        raw = torch.cat((action_logits_raw.detach(), reason_logits_raw.detach()), dim=-1)
        theta = self.compose_theta()
        deploy = raw - theta.unsqueeze(0)
        return {
            "logits_raw": raw,
            "logits_deploy": deploy,
            "action_logits_deploy": deploy[:, : self.action_dim],
            "reason_logits_deploy": deploy[:, self.action_dim :],
            "threshold_logit": theta,
            "threshold_prob": torch.sigmoid(theta),
            "action_threshold_prob": torch.sigmoid(theta[: self.action_dim]),
            "reason_threshold_prob": torch.sigmoid(theta[self.action_dim :]),
        }

    def calibration_objective(
        self,
        action_logits_raw: torch.Tensor,
        reason_logits_raw: torch.Tensor,
        action_targets: torch.Tensor,
        reason_targets: torch.Tensor,
        *,
        surrogate_temperature: float = 0.20,
        soft_f1_weight: float = 1.00,
        bce_weight: float = 0.05,
        rate_weight: float = 0.02,
        delta_weight: float = 0.01,
        cardinality_weight: float = 0.02,
        valid_label_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if surrogate_temperature <= 0:
            raise ValueError("calibration surrogate temperature must be positive")
        if any(weight < 0 for weight in (soft_f1_weight, bce_weight, rate_weight, delta_weight, cardinality_weight)):
            raise ValueError("calibration loss weights must be non-negative")
        if action_logits_raw.shape != action_targets.shape or reason_logits_raw.shape != reason_targets.shape:
            raise ValueError("calibration logits and targets must have matching shapes")
        raw = torch.cat((action_logits_raw.detach(), reason_logits_raw.detach()), dim=-1)
        targets = torch.cat((action_targets, reason_targets), dim=-1).to(dtype=raw.dtype)
        if raw.ndim != 2 or raw.shape[1] != self.num_labels or targets.shape != raw.shape:
            raise ValueError("calibration objective expects action [B,4] and reason [B,21]")
        theta = self.compose_theta()
        surrogate_logits = (raw - theta.unsqueeze(0)) / float(surrogate_temperature)
        probability = torch.sigmoid(surrogate_logits)
        positive_support = targets.sum(dim=0) > 0
        if valid_label_mask is None:
            data_label_mask = torch.ones_like(positive_support)
            soft_f1_label_mask = positive_support
        else:
            if valid_label_mask.shape != (self.num_labels,):
                raise ValueError("valid_label_mask must have shape [25]")
            data_label_mask = valid_label_mask.to(device=raw.device, dtype=torch.bool) & positive_support
            soft_f1_label_mask = data_label_mask
        numerator = 2.0 * (probability * targets).sum(dim=0) + 1e-8
        denominator = probability.sum(dim=0) + targets.sum(dim=0) + 1e-8
        per_label_soft_f1 = numerator / denominator
        if soft_f1_label_mask.any():
            loss_soft_f1 = 1.0 - per_label_soft_f1[soft_f1_label_mask].mean()
            per_label_bce = F.binary_cross_entropy_with_logits(surrogate_logits, targets, reduction="none").mean(0)
            loss_bce = per_label_bce[data_label_mask].mean()
            per_label_rate = (probability.mean(dim=0) - targets.mean(dim=0)).square()
            loss_rate = per_label_rate[data_label_mask].mean()
        else:
            loss_soft_f1 = surrogate_logits.sum() * 0.0
            loss_bce = surrogate_logits.sum() * 0.0
            loss_rate = surrogate_logits.sum() * 0.0
        loss_delta = self.label_delta.square().mean()
        action_mask = data_label_mask[: self.action_dim]
        loss_cardinality = (
            F.smooth_l1_loss(
                probability[:, : self.action_dim][:, action_mask].sum(dim=-1),
                targets[:, : self.action_dim][:, action_mask].sum(dim=-1),
                beta=1.0,
            )
            if action_mask.any()
            else surrogate_logits.sum() * 0.0
        )
        total = (
            soft_f1_weight * loss_soft_f1
            + bce_weight * loss_bce
            + rate_weight * loss_rate
            + delta_weight * loss_delta
            + cardinality_weight * loss_cardinality
        )
        return {
            "loss_calibration_soft_f1": loss_soft_f1,
            "loss_calibration_bce": loss_bce,
            "loss_calibration_rate": loss_rate,
            "loss_calibration_delta": loss_delta,
            "loss_calibration_cardinality": loss_cardinality,
            "loss_calibration_total": total,
            "soft_f1_valid_label_count": soft_f1_label_mask.sum().detach(),
            "surrogate_temperature": raw.new_tensor(float(surrogate_temperature)),
            "threshold_logit": theta,
            "threshold_prob": torch.sigmoid(theta),
        }
