from __future__ import annotations

import torch

from .tida_contracts import _best_label_threshold


def _binary_f1(logits: torch.Tensor, target: torch.Tensor, threshold: torch.Tensor) -> float:
    prediction = torch.sigmoid(logits) >= threshold
    positive = target > 0.5
    tp = (prediction & positive).sum().float()
    fp = (prediction & ~positive).sum().float()
    fn = (~prediction & positive).sum().float()
    return float((2.0 * tp / (2.0 * tp + fp + fn).clamp_min(1.0)).cpu())


def _label_f1(
    logits: torch.Tensor, target: torch.Tensor, thresholds: torch.Tensor
) -> torch.Tensor:
    prediction = torch.sigmoid(logits) >= thresholds.view(1, -1)
    positive = target > 0.5
    tp = (prediction & positive).sum(0).float()
    fp = (prediction & ~positive).sum(0).float()
    fn = (~prediction & positive).sum(0).float()
    return 2.0 * tp / (2.0 * tp + fp + fn).clamp_min(1.0)


def apply_action_traffic_calibration(
    semantic_logits: torch.Tensor,
    traffic_delta: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    """Apply a train-calib-locked per-action traffic contribution scale."""
    if semantic_logits.shape != traffic_delta.shape:
        raise ValueError("semantic logits and traffic delta must have identical shapes")
    if scales.numel() != semantic_logits.shape[1]:
        raise ValueError("one traffic scale is required per action")
    return semantic_logits + scales.to(semantic_logits).view(1, -1) * traffic_delta


def apply_action_traffic_utility(
    semantic_logits: torch.Tensor,
    candidate_delta: torch.Tensor,
    utility_gate: torch.Tensor,
    scales: torch.Tensor,
    cutoffs: torch.Tensor,
) -> torch.Tensor:
    """Apply train-calib-locked selective traffic correction per action."""
    if not (semantic_logits.shape == candidate_delta.shape == utility_gate.shape):
        raise ValueError("semantic logits, candidate delta, and utility gate must match")
    if scales.numel() != semantic_logits.shape[1] or cutoffs.numel() != semantic_logits.shape[1]:
        raise ValueError("one scale and utility cutoff are required per action")
    selected = utility_gate >= cutoffs.to(utility_gate).view(1, -1)
    return semantic_logits + scales.to(semantic_logits).view(1, -1) * candidate_delta * selected


def fit_action_traffic_calibration(
    semantic_logits: torch.Tensor,
    traffic_delta: torch.Tensor,
    target: torch.Tensor,
    *,
    candidates: tuple[float, ...] = (-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0),
) -> dict[str, torch.Tensor | list[float]]:
    if semantic_logits.shape != traffic_delta.shape or semantic_logits.shape != target.shape:
        raise ValueError("semantic logits, traffic delta, and target must have identical shapes")
    scales, thresholds, scores = [], [], []
    for label in range(target.shape[1]):
        options = []
        for scale in candidates:
            logits = semantic_logits[:, label : label + 1] + float(scale) * traffic_delta[:, label : label + 1]
            threshold = _best_label_threshold(logits, target[:, label : label + 1])[0]
            score = _binary_f1(logits[:, 0], target[:, label], threshold)
            options.append((score, -abs(float(scale)), float(scale), threshold))
        score, _, scale, threshold = max(options, key=lambda row: (row[0], row[1]))
        scales.append(scale)
        thresholds.append(threshold)
        scores.append(score)
    return {
        "scales": semantic_logits.new_tensor(scales),
        "thresholds": torch.stack(thresholds).to(semantic_logits),
        "calib_f1_by_action": scores,
    }


def fit_action_traffic_calibration_oof(
    semantic_logits: torch.Tensor,
    traffic_delta: torch.Tensor,
    target: torch.Tensor,
    *,
    candidates: tuple[float, ...] = (
        -32.0, -16.0, -8.0, -4.0, -2.0, -1.0, -0.5,
        0.0,
        0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0,
    ),
    folds: int = 5,
    min_oof_gain: float = 0.0,
    seed: int = 3407,
) -> dict[str, torch.Tensor | list[float]]:
    """Select trajectory utility using train-calib-only out-of-fold F1.

    The zero candidate is mandatory and acts as a no-harm fallback. Thresholds
    are fitted inside each training fold, so scale selection never sees the
    held-out fold's labels. Final thresholds are fitted only after selection.
    """
    if semantic_logits.shape != traffic_delta.shape or semantic_logits.shape != target.shape:
        raise ValueError("semantic logits, traffic delta, and target must have identical shapes")
    if semantic_logits.ndim != 2 or semantic_logits.shape[0] < 2:
        raise ValueError("OOF traffic calibration requires a non-trivial [N,A] tensor")
    if int(folds) < 2 or int(folds) > semantic_logits.shape[0]:
        raise ValueError("folds must be between 2 and the sample count")
    if 0.0 not in candidates:
        raise ValueError("candidates must contain the zero no-harm fallback")

    candidate_tensor = semantic_logits.new_tensor(candidates)
    # Manifest rows can be grouped by source or label. Shuffle deterministically
    # before round-robin assignment so a fold cannot accidentally become a
    # source/label block while keeping deployment fitting exactly reproducible.
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    permutation = torch.randperm(semantic_logits.shape[0], generator=generator)
    fold_ids = torch.empty(semantic_logits.shape[0], dtype=torch.long)
    fold_ids[permutation] = torch.arange(semantic_logits.shape[0]) % int(folds)
    fold_ids = fold_ids.to(semantic_logits.device)
    scores = semantic_logits.new_zeros((len(candidates), semantic_logits.shape[1]))
    counts = semantic_logits.new_zeros((len(candidates), semantic_logits.shape[1]))
    for fold in range(int(folds)):
        fit = fold_ids != fold
        holdout = ~fit
        if not fit.any() or not holdout.any():
            continue
        for index, scale in enumerate(candidate_tensor):
            fit_logits = semantic_logits[fit] + scale * traffic_delta[fit]
            thresholds = _best_label_threshold(fit_logits, target[fit])
            scores[index] += _label_f1(
                semantic_logits[holdout] + scale * traffic_delta[holdout],
                target[holdout],
                thresholds,
            )
            counts[index] += 1
    scores = scores / counts.clamp_min(1.0)

    zero_index = list(candidates).index(0.0)
    selected_indices = []
    gains = []
    for label in range(target.shape[1]):
        ranked = sorted(
            range(len(candidates)),
            key=lambda index: (
                float(scores[index, label]),
                -abs(float(candidates[index])),
            ),
            reverse=True,
        )
        best = ranked[0]
        gain = scores[best, label] - scores[zero_index, label]
        if float(gain) <= float(min_oof_gain):
            best = zero_index
            gain = gain.new_zeros(())
        selected_indices.append(best)
        gains.append(gain)

    selected = torch.tensor(selected_indices, device=semantic_logits.device, dtype=torch.long)
    scales = candidate_tensor[selected]
    deployed = apply_action_traffic_calibration(semantic_logits, traffic_delta, scales)
    thresholds = _best_label_threshold(deployed, target)
    return {
        "scales": scales,
        "thresholds": thresholds,
        "oof_gain_by_action": torch.stack(gains),
        "oof_scores": scores,
        "calib_f1_by_action": [
            float(value) for value in _label_f1(deployed, target, thresholds).cpu()
        ],
        "candidates": list(candidates),
    }


def fit_action_traffic_utility_oof(
    semantic_logits: torch.Tensor,
    candidate_delta: torch.Tensor,
    utility_gate: torch.Tensor,
    target: torch.Tensor,
    *,
    cutoffs: tuple[float, ...] = (0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
    scales: tuple[float, ...] = (-8.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 8.0),
    folds: int = 5,
    min_oof_gain: float = 0.0,
    seed: int = 3407,
) -> dict[str, torch.Tensor | list[float]]:
    """Jointly lock utility cutoff, signed scale, and threshold using OOF labels."""
    if not (
        semantic_logits.shape == candidate_delta.shape == utility_gate.shape == target.shape
    ):
        raise ValueError("utility calibration tensors must have identical [N,A] shapes")
    if semantic_logits.ndim != 2 or semantic_logits.shape[0] < 2:
        raise ValueError("OOF utility calibration requires a non-trivial [N,A] tensor")
    if int(folds) < 2 or int(folds) > semantic_logits.shape[0]:
        raise ValueError("folds must be between 2 and the sample count")
    if 0.0 not in scales:
        raise ValueError("scales must contain the zero no-harm fallback")

    candidates = [(float(cutoff), float(scale)) for cutoff in cutoffs for scale in scales]
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    permutation = torch.randperm(semantic_logits.shape[0], generator=generator)
    fold_ids = torch.empty(semantic_logits.shape[0], dtype=torch.long)
    fold_ids[permutation] = torch.arange(semantic_logits.shape[0]) % int(folds)
    fold_ids = fold_ids.to(semantic_logits.device)
    scores = semantic_logits.new_zeros((len(candidates), semantic_logits.shape[1]))
    counts = torch.zeros_like(scores)
    for fold in range(int(folds)):
        fit = fold_ids != fold
        holdout = ~fit
        for index, (cutoff, scale) in enumerate(candidates):
            fit_logits = semantic_logits[fit] + scale * candidate_delta[fit] * (
                utility_gate[fit] >= cutoff
            )
            thresholds = _best_label_threshold(fit_logits, target[fit])
            holdout_logits = semantic_logits[holdout] + scale * candidate_delta[holdout] * (
                utility_gate[holdout] >= cutoff
            )
            scores[index] += _label_f1(holdout_logits, target[holdout], thresholds)
            counts[index] += 1
    scores = scores / counts.clamp_min(1.0)

    zero_indices = [index for index, (_, scale) in enumerate(candidates) if scale == 0.0]
    zero_index = zero_indices[0]
    selected_indices, gains = [], []
    for label in range(target.shape[1]):
        best = max(
            range(len(candidates)),
            key=lambda index: (
                float(scores[index, label]),
                -abs(candidates[index][1]),
                candidates[index][0],
            ),
        )
        gain = scores[best, label] - scores[zero_index, label]
        if float(gain) <= float(min_oof_gain):
            best = zero_index
            gain = gain.new_zeros(())
        selected_indices.append(best)
        gains.append(gain)

    selected_pairs = [candidates[index] for index in selected_indices]
    selected_cutoffs = semantic_logits.new_tensor([pair[0] for pair in selected_pairs])
    selected_scales = semantic_logits.new_tensor([pair[1] for pair in selected_pairs])
    deployed = apply_action_traffic_utility(
        semantic_logits, candidate_delta, utility_gate, selected_scales, selected_cutoffs
    )
    thresholds = _best_label_threshold(deployed, target)
    return {
        "scales": selected_scales,
        "cutoffs": selected_cutoffs,
        "thresholds": thresholds,
        "oof_gain_by_action": torch.stack(gains),
        "oof_scores": scores,
        "calib_f1_by_action": [
            float(value) for value in _label_f1(deployed, target, thresholds).cpu()
        ],
        "candidates": [{"cutoff": cutoff, "scale": scale} for cutoff, scale in candidates],
    }
