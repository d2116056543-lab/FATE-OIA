from __future__ import annotations

import torch

from fate_oia.utils.acpr_thresholds import acpr_metric_views, standard_joint


def evaluate_ntmcal_tensors(action_base: torch.Tensor, reason_base: torch.Tensor, action_deploy: torch.Tensor, reason_deploy: torch.Tensor, labels_action: torch.Tensor, labels_reason: torch.Tensor) -> dict:
    base = acpr_metric_views(action_base, reason_base, labels_action, labels_reason)
    deploy = acpr_metric_views(action_deploy, reason_deploy, labels_action, labels_reason)
    return {"metrics_base_fixed": base["metrics_raw_fixed"], "metrics_deploy_fixed": deploy["metrics_raw_fixed"], "metrics_oracle_diagnostic": deploy["metrics_per_label_threshold"], "deploy_fixed_joint": standard_joint(deploy["metrics_raw_fixed"]), "base_fixed_joint": standard_joint(base["metrics_raw_fixed"])}
