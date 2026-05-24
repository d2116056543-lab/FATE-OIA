from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch


def sigmoid_probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits.float())


def binary_average_precision(scores: torch.Tensor, targets: torch.Tensor) -> float:
    scores = scores.detach().float().flatten()
    targets = targets.detach().float().flatten()
    pos = float(targets.sum().item())
    if pos <= 0:
        return float("nan")
    order = torch.argsort(scores, descending=True)
    y = targets[order]
    tp = torch.cumsum(y, 0)
    rank = torch.arange(1, y.numel() + 1, device=y.device, dtype=torch.float32)
    precision = tp / rank
    ap = (precision * y).sum() / pos
    return float(ap.item())


def multilabel_confusion(probs: torch.Tensor, targets: torch.Tensor, threshold: float | torch.Tensor = 0.5) -> dict[str, torch.Tensor]:
    if isinstance(threshold, torch.Tensor):
        thr = threshold.to(probs.device).view(1, -1)
    else:
        thr = torch.tensor(float(threshold), device=probs.device).view(1, 1)
    pred = (probs >= thr).float()
    y = targets.float()
    tp = (pred * y).sum(0)
    fp = (pred * (1 - y)).sum(0)
    fn = ((1 - pred) * y).sum(0)
    tn = ((1 - pred) * (1 - y)).sum(0)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def multilabel_metrics_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float | torch.Tensor = 0.5,
    prefix: str = "",
) -> dict[str, Any]:
    probs = sigmoid_probs(logits)
    conf = multilabel_confusion(probs, targets, threshold)
    eps = 1e-9
    precision = conf["tp"] / (conf["tp"] + conf["fp"] + eps)
    recall = conf["tp"] / (conf["tp"] + conf["fn"] + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    valid = torch.isfinite(f1)
    aps = [binary_average_precision(probs[:, i], targets[:, i]) for i in range(targets.shape[1])]
    ap_valid = [x for x in aps if not math.isnan(x)]
    exact = ((probs >= (threshold if isinstance(threshold, float) else threshold.view(1, -1))).float() == targets.float()).all(1).float().mean()
    return {
        f"{prefix}mF1": float(f1[valid].mean().item()) if bool(valid.any()) else 0.0,
        f"{prefix}oF1": float((2 * conf["tp"].sum() / (2 * conf["tp"].sum() + conf["fp"].sum() + conf["fn"].sum() + eps)).item()),
        f"{prefix}mAP": float(sum(ap_valid) / len(ap_valid)) if ap_valid else float("nan"),
        f"{prefix}exact_match": float(exact.item()),
        f"{prefix}per_label_f1": [float(x) for x in f1.detach().cpu()],
        f"{prefix}per_label_precision": [float(x) for x in precision.detach().cpu()],
        f"{prefix}per_label_recall": [float(x) for x in recall.detach().cpu()],
        f"{prefix}per_label_ap": aps,
    }
