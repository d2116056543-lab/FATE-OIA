from __future__ import annotations

import torch


def multilabel_metrics(logits: torch.Tensor, labels: torch.Tensor, threshold: torch.Tensor | float = 0.5) -> dict[str, float | list[float]]:
    probs = logits.sigmoid(); prediction = probs >= threshold
    target = labels.bool(); tp = (prediction & target).sum(0).float(); fp = (prediction & ~target).sum(0).float(); fn = (~prediction & target).sum(0).float()
    f1 = 2 * tp / (2 * tp + fp + fn).clamp_min(1.0)
    precision = tp / (tp + fp).clamp_min(1.0); recall = tp / (tp + fn).clamp_min(1.0)
    overall = 2 * tp.sum() / (2 * tp.sum() + fp.sum() + fn.sum()).clamp_min(1.0)
    return {"mF1": float(f1.mean()), "oF1": float(overall), "per_label_F1": f1.tolist(), "precision": float(precision.mean()), "recall": float(recall.mean())}


def deploy_joint(action: dict[str, float], reason: dict[str, float]) -> float:
    return 0.5 * float(action["mF1"]) + 0.5 * float(reason["mF1"])
