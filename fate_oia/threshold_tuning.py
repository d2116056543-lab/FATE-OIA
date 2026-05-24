from __future__ import annotations

import torch

from fate_oia.metrics import multilabel_metrics_from_logits


def tune_global_threshold(logits: torch.Tensor, targets: torch.Tensor, grid: torch.Tensor | None = None) -> tuple[float, dict]:
    if grid is None:
        grid = torch.linspace(0.05, 0.95, 19)
    best_t = 0.5
    best_m = None
    best_score = -1.0
    for t in grid:
        m = multilabel_metrics_from_logits(logits, targets, float(t.item()))
        score = m["mF1"]
        if score > best_score:
            best_score = score
            best_t = float(t.item())
            best_m = m
    return best_t, best_m or {}


def tune_per_label_thresholds(logits: torch.Tensor, targets: torch.Tensor, grid: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
    if grid is None:
        grid = torch.linspace(0.05, 0.95, 19)
    probs = torch.sigmoid(logits.float())
    thresholds = []
    eps = 1e-9
    for c in range(targets.shape[1]):
        y = targets[:, c].float()
        best_t, best_f1 = 0.5, -1.0
        for t in grid:
            pred = (probs[:, c] >= t).float()
            tp = (pred * y).sum()
            fp = (pred * (1 - y)).sum()
            fn = ((1 - pred) * y).sum()
            f1 = float((2 * tp / (2 * tp + fp + fn + eps)).item())
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t.item())
        thresholds.append(best_t)
    thr = torch.tensor(thresholds, dtype=torch.float32)
    return thr, multilabel_metrics_from_logits(logits, targets, thr)
