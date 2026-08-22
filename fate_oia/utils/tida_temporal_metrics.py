from __future__ import annotations

import torch


def _robust_standardize(value: torch.Tensor) -> torch.Tensor:
    value = value.float()
    median = value.median()
    mad = (value - median).abs().median()
    scale = (1.4826 * mad).clamp_min(1e-6)
    return (value - median) / scale


def robust_motion_score(velocity: torch.Tensor, acceleration: torch.Tensor) -> torch.Tensor:
    """Label-independent sample motion from robust log velocity/acceleration."""
    if velocity.shape != acceleration.shape or velocity.ndim < 2:
        raise ValueError("velocity and acceleration must have identical [N,...] shapes")
    reduce_dims = tuple(range(1, velocity.ndim))
    velocity_size = velocity.float().square().sum(-1).sqrt().mean(tuple(range(1, velocity.ndim - 1)))
    acceleration_size = acceleration.float().square().sum(-1).sqrt().mean(tuple(range(1, acceleration.ndim - 1)))
    if velocity.ndim == 2:
        velocity_size = velocity.float().abs().mean(-1)
        acceleration_size = acceleration.float().abs().mean(-1)
    del reduce_dims
    return _robust_standardize(torch.log1p(velocity_size)) + 0.5 * _robust_standardize(
        torch.log1p(acceleration_size)
    )


def _bootstrap_interval(value: torch.Tensor, samples: int, seed: int) -> tuple[float, float]:
    if value.numel() == 0:
        return float("nan"), float("nan")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    indices = torch.randint(value.numel(), (int(samples), value.numel()), generator=generator)
    means = value.float().cpu()[indices].mean(-1)
    return float(torch.quantile(means, 0.025)), float(torch.quantile(means, 0.975))


def _summary(
    signed_delta: torch.Tensor,
    target: torch.Tensor,
    sample_mask: torch.Tensor,
    *,
    bootstrap_samples: int,
    seed: int,
    epsilon: float,
) -> dict[str, float | int | bool]:
    selected = signed_delta[sample_mask]
    selected_target = target[sample_mask]
    if selected.numel() == 0:
        return {"available": False, "count": 0}
    per_sample = selected.mean(-1)
    ci_low, ci_high = _bootstrap_interval(per_sample, bootstrap_samples, seed)
    positive = selected_target.sum(0)
    negative = selected_target.shape[0] - positive
    eligible = (positive > 0) & (negative > 0)
    return {
        "available": True,
        "count": int(selected.shape[0]),
        "signed_margin_mean": float(selected.mean()),
        "signed_margin_median": float(selected.median()),
        "benefit_rate": float((selected > epsilon).float().mean()),
        "harm_rate": float((selected < -epsilon).float().mean()),
        "neutral_rate": float((selected.abs() <= epsilon).float().mean()),
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "eligible_label_count": int(eligible.sum()),
    }


def _quartile_summaries(
    score: torch.Tensor,
    signed_delta: torch.Tensor,
    target: torch.Tensor,
    *,
    bootstrap_samples: int,
    seed: int,
    epsilon: float,
) -> list[dict[str, float | int | bool]]:
    boundaries = torch.quantile(score.float(), torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]))
    rows = []
    for index in range(4):
        lower, upper = boundaries[index], boundaries[index + 1]
        mask = (score >= lower) & ((score <= upper) if index == 3 else (score < upper))
        row = _summary(
            signed_delta, target, mask, bootstrap_samples=bootstrap_samples,
            seed=seed + index + 1, epsilon=epsilon,
        )
        row.update({"quartile": index + 1, "score_low": float(lower), "score_high": float(upper)})
        rows.append(row)
    return rows


def paired_temporal_contribution(
    image_logits: torch.Tensor,
    video_logits: torch.Tensor,
    target: torch.Tensor,
    *,
    motion_score: torch.Tensor,
    bootstrap_samples: int = 2000,
    seed: int = 20260823,
    epsilon: float = 1e-4,
) -> dict[str, object]:
    if image_logits.shape != video_logits.shape or image_logits.shape != target.shape:
        raise ValueError("image/video logits and targets must have identical shapes")
    if motion_score.shape != image_logits.shape[:1]:
        raise ValueError("motion_score must be [N]")
    signed_delta = (2.0 * target.float() - 1.0) * (video_logits.float() - image_logits.float())
    uncertainty = (4.0 * image_logits.float().sigmoid() * (1.0 - image_logits.float().sigmoid())).mean(-1)
    all_samples = torch.ones(image_logits.shape[0], dtype=torch.bool)
    per_label = []
    for label in range(image_logits.shape[1]):
        label_target = target[:, label]
        class_available = bool((label_target > 0.5).any() and (label_target <= 0.5).any())
        values = signed_delta[:, label]
        ci_low, ci_high = _bootstrap_interval(values, bootstrap_samples, seed + 100 + label)
        per_label.append({
            "label_id": label,
            "class_metrics_available": class_available,
            "signed_margin_mean": float(values.mean()),
            "sign_accuracy": float((values > epsilon).float().mean()),
            "harm_rate": float((values < -epsilon).float().mean()),
            "bootstrap_ci_low": ci_low,
            "bootstrap_ci_high": ci_high,
        })
    return {
        "full": _summary(
            signed_delta, target, all_samples, bootstrap_samples=bootstrap_samples,
            seed=seed, epsilon=epsilon,
        ),
        "motion_quartiles": _quartile_summaries(
            motion_score, signed_delta, target, bootstrap_samples=bootstrap_samples,
            seed=seed + 1000, epsilon=epsilon,
        ),
        "uncertainty_quartiles": _quartile_summaries(
            uncertainty, signed_delta, target, bootstrap_samples=bootstrap_samples,
            seed=seed + 2000, epsilon=epsilon,
        ),
        "per_label": per_label,
    }
