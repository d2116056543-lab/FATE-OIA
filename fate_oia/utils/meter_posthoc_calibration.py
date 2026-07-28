from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from fate_oia.metrics import multilabel_metrics_from_logits


@dataclass(frozen=True)
class METERCalibrationResult:
    theta: Tensor
    model_state_hash_before: str
    model_state_hash_after: str
    fit_split: str
    representation_updated: bool
    accepted: bool = True
    fallback_reason: str = ""
    train_calib_raw_joint: float | None = None
    train_calib_deploy_joint: float | None = None


def fit_train_calib_deploy_theta(logits: Tensor, labels: Tensor, *, model_state_hash: str, fit_split: str = "train_calib") -> METERCalibrationResult:
    if fit_split != "train_calib":
        raise ValueError("METER calibration can only fit train_calib")
    probability = torch.sigmoid(logits.detach())
    positive_rate = labels.detach().float().mean(dim=0).clamp(0.01, 0.99)
    target_logit = torch.logit(positive_rate)
    current_logit = torch.logit(probability.mean(dim=0).clamp(0.01, 0.99))
    theta = (current_logit - target_logit).clamp(-2.0, 2.0)
    return METERCalibrationResult(theta=theta, model_state_hash_before=model_state_hash, model_state_hash_after=model_state_hash, fit_split=fit_split, representation_updated=False)


def _joint_score(action_logits: Tensor, action_labels: Tensor, reason_logits: Tensor, reason_labels: Tensor) -> float:
    action = multilabel_metrics_from_logits(action_logits, action_labels, prefix="Act_")
    reason = multilabel_metrics_from_logits(reason_logits, reason_labels, prefix="Exp_")
    return 0.5 * (float(action.get("Act_mF1", 0.0)) + float(reason.get("Exp_mF1", 0.0)))


def guard_train_calib_deploy_theta(
    action_logits: Tensor,
    action_labels: Tensor,
    reason_logits: Tensor,
    reason_labels: Tensor,
    candidate: METERCalibrationResult,
    *,
    fallback_on_deploy_degradation: bool = True,
    min_joint_delta: float = 0.0,
) -> METERCalibrationResult:
    """Accept train-calib theta only when its fit-split joint is not worse."""
    if candidate.fit_split != "train_calib":
        raise ValueError("Calibration guard can only evaluate train_calib")
    if candidate.representation_updated:
        raise ValueError("Post-hoc calibration cannot update representation")
    expected = action_logits.shape[1] + reason_logits.shape[1]
    if candidate.theta.ndim != 1 or candidate.theta.numel() != expected:
        raise ValueError("Calibration theta must concatenate action and reason thresholds")
    action_theta = candidate.theta[: action_logits.shape[1]].to(action_logits)
    reason_theta = candidate.theta[action_logits.shape[1] :].to(reason_logits)
    raw_joint = _joint_score(action_logits, action_labels, reason_logits, reason_labels)
    deploy_joint = _joint_score(action_logits - action_theta, action_labels, reason_logits - reason_theta, reason_labels)
    if fallback_on_deploy_degradation and deploy_joint < raw_joint + float(min_joint_delta):
        return METERCalibrationResult(
            theta=torch.zeros_like(candidate.theta),
            model_state_hash_before=candidate.model_state_hash_before,
            model_state_hash_after=candidate.model_state_hash_after,
            fit_split="train_calib",
            representation_updated=False,
            accepted=False,
            fallback_reason="train_calib_deploy_joint_degradation",
            train_calib_raw_joint=raw_joint,
            train_calib_deploy_joint=deploy_joint,
        )
    return METERCalibrationResult(
        theta=candidate.theta,
        model_state_hash_before=candidate.model_state_hash_before,
        model_state_hash_after=candidate.model_state_hash_after,
        fit_split="train_calib",
        representation_updated=False,
        accepted=True,
        fallback_reason="",
        train_calib_raw_joint=raw_joint,
        train_calib_deploy_joint=deploy_joint,
    )


def apply_meter_deploy(logits: Tensor, calibration: METERCalibrationResult) -> Tensor:
    if calibration.representation_updated or calibration.fit_split != "train_calib":
        raise ValueError("Invalid post-hoc calibration result")
    return logits - calibration.theta.to(device=logits.device, dtype=logits.dtype)
