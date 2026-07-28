from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from fate_oia.metrics import binary_average_precision, multilabel_metrics_from_logits


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
    temperature: Tensor | None = None
    strategy: str = "per_label"
    fallback_theta: Tensor | None = None
    fallback_temperature: Tensor | None = None
    map_max_abs_delta: float | None = None
    threshold_rms_ratio: float | None = None


def _macro_f1(logits: Tensor, labels: Tensor) -> float:
    return float(multilabel_metrics_from_logits(logits, labels).get("mF1", 0.0))


def _best_scalar_threshold(logits: Tensor, labels: Tensor) -> Tensor:
    scale = logits.detach().float().std().clamp_min(0.25)
    candidates = torch.linspace(-2.0, 2.0, 81, device=logits.device) * scale
    scores = torch.stack([
        torch.tensor(_macro_f1(logits - threshold, labels), device=logits.device)
        for threshold in candidates
    ])
    return candidates[int(scores.argmax())]


def _best_label_threshold(logits: Tensor, labels: Tensor) -> Tensor:
    values = []
    for label in range(logits.shape[1]):
        values.append(_best_scalar_threshold(logits[:, label : label + 1], labels[:, label : label + 1]))
    return torch.stack(values)


def _group_threshold(per_label: Tensor, label_groups: tuple[int, ...]) -> Tensor:
    result = torch.empty_like(per_label)
    for group in sorted(set(label_groups)):
        indices = [index for index, value in enumerate(label_groups) if value == group]
        result[indices] = per_label[indices].mean()
    return result


def fit_train_calib_deploy_theta(
    logits: Tensor,
    labels: Tensor,
    *,
    model_state_hash: str,
    fit_split: str = "train_calib",
    label_groups: tuple[int, ...] | None = None,
) -> METERCalibrationResult:
    if fit_split != "train_calib":
        raise ValueError("METER calibration can only fit train_calib")
    logits = logits.detach().float()
    labels = labels.detach().float()
    label_groups = label_groups or tuple(0 for _ in range(logits.shape[1]))
    if len(label_groups) != logits.shape[1]:
        raise ValueError("label_groups must have one entry per label")
    candidates: list[tuple[float, str, Tensor, Tensor]] = []
    for scalar_temperature in (0.75, 1.0, 1.25, 1.5):
        temperature = torch.full(
            (logits.shape[1],),
            float(scalar_temperature),
            device=logits.device,
        )
        scaled = logits / temperature
        per_label = _best_label_threshold(scaled, labels)
        global_theta = torch.full_like(per_label, float(_best_scalar_threshold(scaled, labels)))
        group_theta = _group_threshold(per_label, label_groups)
        for name, theta in (("global", global_theta), ("group", group_theta), ("per_label", per_label)):
            candidates.append((_macro_f1(scaled - theta, labels), name, theta, temperature))
        for shrinkage in (0.25, 0.50, 0.75):
            theta = shrinkage * per_label + (1.0 - shrinkage) * group_theta
            candidates.append((_macro_f1(scaled - theta, labels), "group_shrinkage", theta, temperature))
    _, strategy, theta, temperature = max(candidates, key=lambda item: item[0])
    rms_limit = 0.35 * logits.square().mean().sqrt()
    theta_rms = theta.square().mean().sqrt()
    if theta_rms > rms_limit and theta_rms > 0:
        theta = theta * (rms_limit / theta_rms)
    global_candidates = [candidate for candidate in candidates if candidate[1] in {"global", "group"}]
    _, _, fallback_theta, fallback_temperature = max(global_candidates, key=lambda item: item[0])
    return METERCalibrationResult(
        theta=theta,
        temperature=temperature,
        strategy=strategy,
        fallback_theta=fallback_theta,
        fallback_temperature=fallback_temperature,
        model_state_hash_before=model_state_hash,
        model_state_hash_after=model_state_hash,
        fit_split=fit_split,
        representation_updated=False,
    )


def _joint_score(action_logits: Tensor, action_labels: Tensor, reason_logits: Tensor, reason_labels: Tensor) -> float:
    action = multilabel_metrics_from_logits(action_logits, action_labels, prefix="Act_")
    reason = multilabel_metrics_from_logits(reason_logits, reason_labels, prefix="Exp_")
    return 0.5 * (float(action.get("Act_mF1", 0.0)) + float(reason.get("Exp_mF1", 0.0)))


def _ranking_map(scores: Tensor, labels: Tensor) -> float:
    per_label = [
        binary_average_precision(scores[:, index], labels[:, index])
        for index in range(labels.shape[1])
    ]
    finite = [value for value in per_label if torch.isfinite(torch.tensor(value))]
    return float(sum(finite) / len(finite)) if finite else float("nan")


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
    temperature = candidate.temperature
    if temperature is None:
        temperature = torch.ones_like(candidate.theta)
    action_temperature = temperature[: action_logits.shape[1]].to(action_logits)
    reason_temperature = temperature[action_logits.shape[1] :].to(reason_logits)
    raw_joint = _joint_score(action_logits, action_labels, reason_logits, reason_labels)
    deploy_joint = _joint_score(
        action_logits / action_temperature - action_theta,
        action_labels,
        reason_logits / reason_temperature - reason_theta,
        reason_labels,
    )
    map_delta = max(
        abs(
            _ranking_map(action_logits, action_labels)
            - _ranking_map(action_logits / action_temperature - action_theta, action_labels)
        ),
        abs(
            _ranking_map(reason_logits, reason_labels)
            - _ranking_map(reason_logits / reason_temperature - reason_theta, reason_labels)
        ),
    )
    if map_delta > 1e-7:
        raise RuntimeError("Post-hoc calibration changed ranking mAP")
    concatenated_logits = torch.cat([action_logits, reason_logits], dim=1)
    threshold_rms_ratio = float(
        candidate.theta.float().square().mean().sqrt()
        / concatenated_logits.float().square().mean().sqrt().clamp_min(1e-6)
    )
    threshold_limit_violated = threshold_rms_ratio > 0.35 + 1e-6
    if fallback_on_deploy_degradation and (
        deploy_joint < raw_joint + float(min_joint_delta)
        or threshold_limit_violated
    ):
        fallback_theta = candidate.fallback_theta
        fallback_temperature = candidate.fallback_temperature
        if fallback_theta is None:
            fallback_theta = torch.zeros_like(candidate.theta)
        if fallback_temperature is None:
            fallback_temperature = torch.ones_like(candidate.theta)
        fallback_action_theta = fallback_theta[: action_logits.shape[1]].to(action_logits)
        fallback_reason_theta = fallback_theta[action_logits.shape[1] :].to(reason_logits)
        fallback_action_temperature = fallback_temperature[: action_logits.shape[1]].to(action_logits)
        fallback_reason_temperature = fallback_temperature[action_logits.shape[1] :].to(reason_logits)
        fallback_joint = _joint_score(
            action_logits / fallback_action_temperature - fallback_action_theta,
            action_labels,
            reason_logits / fallback_reason_temperature - fallback_reason_theta,
            reason_labels,
        )
        fallback_strategy = "group_or_global_fallback"
        if fallback_joint < raw_joint:
            fallback_theta = torch.zeros_like(candidate.theta)
            fallback_temperature = torch.ones_like(candidate.theta)
            fallback_joint = raw_joint
            fallback_strategy = "global_raw_fallback"
        return METERCalibrationResult(
            theta=fallback_theta,
            temperature=fallback_temperature,
            strategy=fallback_strategy,
            model_state_hash_before=candidate.model_state_hash_before,
            model_state_hash_after=candidate.model_state_hash_after,
            fit_split="train_calib",
            representation_updated=False,
            accepted=False,
            fallback_reason=(
                "threshold_rms_limit"
                if threshold_limit_violated
                else "train_calib_deploy_joint_degradation"
            ),
            train_calib_raw_joint=raw_joint,
            train_calib_deploy_joint=fallback_joint,
            map_max_abs_delta=map_delta,
            threshold_rms_ratio=threshold_rms_ratio,
        )
    return METERCalibrationResult(
        theta=candidate.theta,
        temperature=temperature,
        strategy=candidate.strategy,
        fallback_theta=candidate.fallback_theta,
        fallback_temperature=candidate.fallback_temperature,
        model_state_hash_before=candidate.model_state_hash_before,
        model_state_hash_after=candidate.model_state_hash_after,
        fit_split="train_calib",
        representation_updated=False,
        accepted=True,
        fallback_reason="",
        train_calib_raw_joint=raw_joint,
        train_calib_deploy_joint=deploy_joint,
        map_max_abs_delta=map_delta,
        threshold_rms_ratio=threshold_rms_ratio,
    )


def apply_meter_deploy(logits: Tensor, calibration: METERCalibrationResult) -> Tensor:
    if calibration.representation_updated or calibration.fit_split != "train_calib":
        raise ValueError("Invalid post-hoc calibration result")
    temperature = calibration.temperature
    if temperature is None:
        temperature = torch.ones_like(calibration.theta)
    return (
        logits / temperature.to(device=logits.device, dtype=logits.dtype).clamp_min(1e-3)
        - calibration.theta.to(device=logits.device, dtype=logits.dtype)
    )
