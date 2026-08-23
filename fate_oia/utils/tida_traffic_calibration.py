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
