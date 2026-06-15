from __future__ import annotations

import torch


def _f1_for_predictions(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = target.float()
    pred = pred.float()
    tp = (pred * target).sum(0)
    fp = (pred * (1.0 - target)).sum(0)
    fn = ((1.0 - pred) * target).sum(0)
    return (2.0 * tp) / (2.0 * tp + fp + fn).clamp_min(1e-6)


def search_best_thresholds_for_f1(
    logits: torch.Tensor,
    targets: torch.Tensor,
    grid: torch.Tensor | None = None,
    min_threshold: float = 0.01,
    max_threshold: float = 0.95,
) -> dict[str, torch.Tensor]:
    logits = logits.detach().float().cpu()
    targets = targets.detach().float().cpu()
    probs = torch.sigmoid(logits)
    if grid is None:
        grid = torch.arange(min_threshold, max_threshold + 1e-9, 0.01)
    grid = grid.float().cpu().clamp(min_threshold, max_threshold)
    num_labels = logits.shape[1]
    best_thresholds = torch.full((num_labels,), 0.5)
    best_f1 = torch.zeros(num_labels)
    pred_rate = torch.zeros(num_labels)
    for label in range(num_labels):
        label_probs = probs[:, label].view(-1, 1)
        label_target = targets[:, label].view(-1, 1)
        pred = (label_probs >= grid.view(1, -1)).float()
        target = label_target.expand_as(pred)
        tp = (pred * target).sum(0)
        fp = (pred * (1.0 - target)).sum(0)
        fn = ((1.0 - pred) * target).sum(0)
        f1 = (2.0 * tp) / (2.0 * tp + fp + fn).clamp_min(1e-6)
        idx = int(torch.argmax(f1).item())
        best_thresholds[label] = grid[idx]
        best_f1[label] = f1[idx]
        pred_rate[label] = pred[:, idx].mean()
    return {
        "threshold_prob": best_thresholds,
        "threshold_logit": torch.logit(best_thresholds.clamp(1e-5, 1 - 1e-5)),
        "best_f1": best_f1,
        "pred_rate": pred_rate,
        "support_pos": targets.sum(0),
        "support_neg": (1.0 - targets).sum(0),
    }


def compute_fixed_metrics_at_thresholds(logits: torch.Tensor, targets: torch.Tensor, thresholds: torch.Tensor) -> dict:
    logits = logits.detach().float().cpu()
    targets = targets.detach().float().cpu()
    thresholds = thresholds.detach().float().cpu().view(1, -1)
    probs = torch.sigmoid(logits)
    pred = (probs >= thresholds).float()
    per_label_f1 = _f1_for_predictions(pred, targets)
    exact_match = (pred == targets).all(-1).float().mean().item() if pred.numel() else 0.0
    return {
        "macro_f1": float(per_label_f1.mean().item()) if per_label_f1.numel() else 0.0,
        "per_label_f1": per_label_f1.tolist(),
        "pred_rate": pred.mean(0).tolist() if pred.numel() else [],
        "exact_match": exact_match,
    }
