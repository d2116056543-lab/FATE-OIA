from __future__ import annotations

import math
from typing import Any

import torch


def schedule_values(update: int, total_updates: int, *, warmup_ratio: float = 0.05, ramp_end_ratio: float = 0.20, min_lr_ratio: float = 0.10) -> dict[str, float]:
    progress = max(0.0, min(float(update) / max(int(total_updates), 1), 1.0))
    if progress <= warmup_ratio:
        temporal_scale = 0.0
        lr_scale = max(progress / max(warmup_ratio, 1e-8), 1e-3)
        phase = "FLOW_FOUNDATION"
    else:
        ramp_progress = min((progress - warmup_ratio) / max(ramp_end_ratio - warmup_ratio, 1e-8), 1.0)
        temporal_scale = 0.5 - 0.5 * math.cos(math.pi * ramp_progress)
        cosine_progress = (progress - warmup_ratio) / max(1.0 - warmup_ratio, 1e-8)
        lr_scale = min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * cosine_progress))
        phase = "FLOW_CREDIT" if progress < ramp_end_ratio else "SAFE_JOINT"
    return {"progress": progress, "temporal_scale": temporal_scale, "lr_scale": lr_scale, "phase": phase}


def resolve_schedule_total_updates(
    *, updates_per_epoch: int, configured_epochs: int, schedule_total_updates: int | None = None
) -> int:
    formal_total = int(updates_per_epoch) * int(configured_epochs)
    if formal_total <= 0:
        raise ValueError("schedule requires positive updates_per_epoch and configured_epochs")
    if schedule_total_updates is None:
        return formal_total
    if int(schedule_total_updates) <= 0:
        raise ValueError("schedule_total_updates must be positive")
    return int(schedule_total_updates)


def validate_training_protocol(config: dict[str, Any]) -> None:
    experiment, runtime, training = config["experiment"], config["runtime"], config["training"]
    errors = []
    if experiment.get("eval_splits") != ["test"]:
        errors.append("eval_splits must be [test]")
    if experiment.get("best_selection_split") != "test":
        errors.append("best_selection_split must be test")
    if not runtime.get("test_every_epoch") or not runtime.get("foreground_only"):
        errors.append("test_every_epoch and foreground_only are required")
    if not training.get("no_metric_early_stop") or int(training.get("epochs", 0)) != 10:
        errors.append("exactly ten epochs without metric early stop are required")
    if not runtime.get("no_feature_cache") or not runtime.get("require_no_token_compression"):
        errors.append("cache/compression are forbidden")
    if errors:
        raise ValueError("; ".join(errors))


def choose_memory_candidate(rows: list[dict[str, Any]], max_reserved_gib: float = 45.0, max_growth_gib: float = 0.25) -> dict[str, Any]:
    safe = [row for row in rows if float(row["peak_reserved_gib"]) <= max_reserved_gib and float(row["growth_gib"]) <= max_growth_gib and not row.get("oom", False)]
    if not safe:
        raise RuntimeError("no safe memory candidate")
    fastest = max(float(row["samples_per_second"]) for row in safe)
    near = [row for row in safe if float(row["samples_per_second"]) >= fastest * 0.97]
    return min(near, key=lambda row: (float(row["peak_reserved_gib"]), -float(row["samples_per_second"])))


def validate_git_binding(local_head: str, remote_head: str, review_head: str) -> list[str]:
    failures = []
    if local_head != remote_head:
        failures.append("local_head != remote_head")
    if local_head != review_head:
        failures.append("local_head != review_head")
    return failures


def _best_label_threshold(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    grid = torch.linspace(0.05, 0.95, 91, device=logits.device)
    probabilities = torch.sigmoid(logits)
    prediction = probabilities[:, :, None] >= grid
    truth = target.bool()[:, :, None]
    tp = (prediction & truth).sum(0).float()
    fp = (prediction & ~truth).sum(0).float()
    fn = (~prediction & truth).sum(0).float()
    f1 = (2 * tp) / (2 * tp + fp + fn).clamp_min(1)
    return grid[f1.argmax(-1)]


def select_reason_beta(
    train_calib_image_logits: torch.Tensor,
    train_calib_video_delta: torch.Tensor,
    train_calib_targets: torch.Tensor,
    train_audit_image_logits: torch.Tensor,
    train_audit_video_delta: torch.Tensor,
    train_audit_targets: torch.Tensor,
    *,
    candidates: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    folds: int = 5,
) -> dict[str, torch.Tensor]:
    image = torch.cat([train_calib_image_logits, train_audit_image_logits], dim=0)
    delta = torch.cat([train_calib_video_delta, train_audit_video_delta], dim=0)
    target = torch.cat([train_calib_targets, train_audit_targets], dim=0)
    candidate_tensor = torch.tensor(candidates, device=image.device, dtype=image.dtype)
    scores = torch.zeros(len(candidates), image.shape[1], device=image.device)
    fold_id = torch.arange(image.shape[0], device=image.device) % int(folds)
    for fold in range(int(folds)):
        fit = fold_id != fold
        holdout = ~fit
        if not holdout.any() or not fit.any():
            continue
        for index, beta in enumerate(candidate_tensor):
            fit_logits = image[fit] + beta * delta[fit]
            threshold = _best_label_threshold(fit_logits, target[fit])
            prediction = torch.sigmoid(image[holdout] + beta * delta[holdout]) >= threshold
            truth = target[holdout].bool()
            tp = (prediction & truth).sum(0).float()
            fp = (prediction & ~truth).sum(0).float()
            fn = (~prediction & truth).sum(0).float()
            scores[index] += (2 * tp) / (2 * tp + fp + fn).clamp_min(1)
    best_index = scores.argmax(0)
    beta = candidate_tensor[best_index]
    threshold = _best_label_threshold(image + beta[None] * delta, target)
    return {"reason_beta": beta, "reason_threshold": threshold, "oof_scores": scores / max(int(folds), 1)}
