from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class METERCalibrationResult:
    theta: Tensor
    model_state_hash_before: str
    model_state_hash_after: str
    fit_split: str
    representation_updated: bool


def fit_train_calib_deploy_theta(logits: Tensor, labels: Tensor, *, model_state_hash: str, fit_split: str = "train_calib") -> METERCalibrationResult:
    if fit_split != "train_calib":
        raise ValueError("METER calibration can only fit train_calib")
    probability = torch.sigmoid(logits.detach())
    positive_rate = labels.detach().float().mean(dim=0).clamp(0.01, 0.99)
    target_logit = torch.logit(positive_rate)
    current_logit = torch.logit(probability.mean(dim=0).clamp(0.01, 0.99))
    theta = (current_logit - target_logit).clamp(-2.0, 2.0)
    return METERCalibrationResult(theta=theta, model_state_hash_before=model_state_hash, model_state_hash_after=model_state_hash, fit_split=fit_split, representation_updated=False)


def apply_meter_deploy(logits: Tensor, calibration: METERCalibrationResult) -> Tensor:
    if calibration.representation_updated or calibration.fit_split != "train_calib":
        raise ValueError("Invalid post-hoc calibration result")
    return logits - calibration.theta.to(device=logits.device, dtype=logits.dtype)
