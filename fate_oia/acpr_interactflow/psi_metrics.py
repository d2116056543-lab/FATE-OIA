from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


def _safe_div(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    return num / den.clamp_min(1e-9)


@dataclass
class PSIActionMetrics:
    Act_oAcc: float
    Act_mAcc: float
    Act_macroF1: float
    Act_weightedF1: float
    Act_stopF1: float
    soft_target_KL: float
    ECE: float
    prediction_rate: list[float]
    per_class_precision: list[float]
    per_class_recall: list[float]
    per_class_f1: list[float]
    confusion: list[list[int]]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PSIExp29Metrics:
    Exp_oF1: float
    Exp_mF1: float
    Exp_mAP: float
    Exp_microF1: float
    Exp_macroF1: float
    Exp_positive_mask_only_mF1: float
    all_zero_unknown_count: int
    per_label_precision: list[float]
    per_label_recall: list[float]
    per_label_f1: list[float]
    per_label_ap: list[float]

    def to_dict(self) -> dict:
        return asdict(self)


def compute_ece(probs: torch.Tensor, targets: torch.Tensor, bins: int = 15) -> float:
    conf, pred = probs.max(-1)
    correct = (pred == targets).float()
    ece = torch.zeros((), device=probs.device)
    edges = torch.linspace(0, 1, bins + 1, device=probs.device)
    for i in range(bins):
        mask = (conf > edges[i]) & (conf <= edges[i + 1])
        if mask.any():
            ece = ece + mask.float().mean() * (conf[mask].mean() - correct[mask].mean()).abs()
    return float(ece.detach().cpu())


def compute_psi_action_metrics(logits: torch.Tensor, majority: torch.Tensor, soft_targets: torch.Tensor) -> dict:
    probs = torch.softmax(logits, dim=-1)
    pred = probs.argmax(-1)
    majority = majority.long()
    num_classes = logits.shape[-1]
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long, device=logits.device)
    for t, p in zip(majority.view(-1), pred.view(-1)):
        confusion[t, p] += 1
    tp = confusion.diag().float()
    pred_count = confusion.sum(0).float()
    true_count = confusion.sum(1).float()
    precision = _safe_div(tp, pred_count)
    recall = _safe_div(tp, true_count)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    weighted = (f1 * _safe_div(true_count, true_count.sum())).sum()
    kl = (soft_targets.clamp_min(1e-9) * (soft_targets.clamp_min(1e-9).log() - probs.clamp_min(1e-9).log())).sum(-1).mean()
    stop_index = min(2, num_classes - 1)
    metrics = PSIActionMetrics(
        Act_oAcc=float((pred == majority).float().mean().detach().cpu()),
        Act_mAcc=float(recall.mean().detach().cpu()),
        Act_macroF1=float(f1.mean().detach().cpu()),
        Act_weightedF1=float(weighted.detach().cpu()),
        Act_stopF1=float(f1[stop_index].detach().cpu()) if num_classes > 0 else 0.0,
        soft_target_KL=float(kl.detach().cpu()),
        ECE=compute_ece(probs, majority),
        prediction_rate=[float(x) for x in torch.bincount(pred, minlength=num_classes).float().div(pred.numel()).detach().cpu()],
        per_class_precision=[float(x) for x in precision.detach().cpu()],
        per_class_recall=[float(x) for x in recall.detach().cpu()],
        per_class_f1=[float(x) for x in f1.detach().cpu()],
        confusion=confusion.detach().cpu().tolist(),
    ).to_dict()
    # DAMO-style aliases are kept alongside the original keys for parity audits.
    metrics.update(
        {
            "Act_macro_F1": metrics["Act_macroF1"],
            "Act_weighted_F1": metrics["Act_weightedF1"],
            "Maintain_F1": metrics["per_class_f1"][0] if num_classes > 0 else 0.0,
            "Reduce_F1": metrics["per_class_f1"][1] if num_classes > 1 else 0.0,
            "Stop_F1": metrics["per_class_f1"][stop_index] if num_classes > 0 else 0.0,
            "Maintain_recall": metrics["per_class_recall"][0] if num_classes > 0 else 0.0,
            "Reduce_recall": metrics["per_class_recall"][1] if num_classes > 1 else 0.0,
            "Stop_recall": metrics["per_class_recall"][stop_index] if num_classes > 0 else 0.0,
        }
    )
    return metrics


def _average_precision(scores: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    order = scores.argsort(descending=True)
    y = target[order].float()
    positives = y.sum()
    if positives <= 0:
        return torch.zeros((), device=scores.device)
    precision_at_k = y.cumsum(0) / torch.arange(1, y.numel() + 1, device=scores.device).float()
    return (precision_at_k * y).sum() / positives


def compute_psi_exp29_metrics(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None = None, threshold: float = 0.5) -> dict:
    probs = torch.sigmoid(logits)
    targets = targets.float()
    if mask is None:
        mask = torch.ones_like(targets)
    mask = mask.float()
    unknown_rows = (mask.sum(-1) == 0).sum()
    pred = (probs >= threshold).float()
    tp = (pred * targets * mask).sum(0)
    fp = (pred * (1 - targets) * mask).sum(0)
    fn = ((1 - pred) * targets * mask).sum(0)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    micro_tp, micro_fp, micro_fn = tp.sum(), fp.sum(), fn.sum()
    micro_f1 = _safe_div(2 * micro_tp, 2 * micro_tp + micro_fp + micro_fn)
    ap = torch.stack([_average_precision(probs[:, i][mask[:, i] > 0], targets[:, i][mask[:, i] > 0]) if (mask[:, i] > 0).any() else torch.zeros((), device=logits.device) for i in range(logits.shape[-1])])
    positive_labels = targets.sum(0) > 0
    positive_mask_f1 = f1[positive_labels].mean() if positive_labels.any() else torch.zeros((), device=logits.device)
    metrics = PSIExp29Metrics(
        Exp_oF1=float(micro_f1.detach().cpu()),
        Exp_mF1=float(f1.mean().detach().cpu()),
        Exp_mAP=float(ap.mean().detach().cpu()),
        Exp_microF1=float(micro_f1.detach().cpu()),
        Exp_macroF1=float(f1.mean().detach().cpu()),
        Exp_positive_mask_only_mF1=float(positive_mask_f1.detach().cpu()),
        all_zero_unknown_count=int(unknown_rows.detach().cpu()),
        per_label_precision=[float(x) for x in precision.detach().cpu()],
        per_label_recall=[float(x) for x in recall.detach().cpu()],
        per_label_f1=[float(x) for x in f1.detach().cpu()],
        per_label_ap=[float(x) for x in ap.detach().cpu()],
    ).to_dict()
    metrics.update(
        {
            "Exp_micro_F1": metrics["Exp_microF1"],
            "Exp_macro_F1": metrics["Exp_macroF1"],
            "Exp_positive_mask_only_F1": metrics["Exp_positive_mask_only_mF1"],
        }
    )
    return metrics
