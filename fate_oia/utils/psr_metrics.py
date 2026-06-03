from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class PSRMetricBundle:
    Act_mF1: float
    Act_oF1: float
    Act_mAP: float
    Exp_mF1: float
    Exp_oF1: float
    Exp_mAP: float
    standard_joint: float
    action_primary_score: float
    per_action_F1: list[float]
    per_reason_F1: list[float]
    per_action_AP: list[float]
    per_reason_AP: list[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "Act_mF1": self.Act_mF1,
            "Act_oF1": self.Act_oF1,
            "Act_mAP": self.Act_mAP,
            "Exp_mF1": self.Exp_mF1,
            "Exp_oF1": self.Exp_oF1,
            "Exp_mAP": self.Exp_mAP,
            "standard_joint": self.standard_joint,
            "action_primary_score": self.action_primary_score,
            "per_action_F1": self.per_action_F1,
            "per_reason_F1": self.per_reason_F1,
            "per_action_AP": self.per_action_AP,
            "per_reason_AP": self.per_reason_AP,
        }


def sigmoid_np(logits: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(logits, torch.Tensor):
        arr = logits.detach().cpu().float().numpy()
    else:
        arr = np.asarray(logits, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-arr))


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def per_label_f1_from_probs(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> list[float]:
    pred = probs >= threshold
    y = labels.astype(bool)
    out: list[float] = []
    for i in range(y.shape[1]):
        tp = float(np.logical_and(pred[:, i], y[:, i]).sum())
        fp = float(np.logical_and(pred[:, i], ~y[:, i]).sum())
        fn = float(np.logical_and(~pred[:, i], y[:, i]).sum())
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        out.append(_safe_div(2.0 * precision * recall, precision + recall))
    return out


def overall_f1_from_probs(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> float:
    pred = probs >= threshold
    y = labels.astype(bool)
    tp = float(np.logical_and(pred, y).sum())
    fp = float(np.logical_and(pred, ~y).sum())
    fn = float(np.logical_and(~pred, y).sum())
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return _safe_div(2.0 * precision * recall, precision + recall)


def average_precision_1d(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(np.float32)
    positives = float(labels.sum())
    if positives <= 0:
        return 0.0
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels)
    precision = tp / (np.arange(len(sorted_labels), dtype=np.float32) + 1.0)
    return float((precision * sorted_labels).sum() / positives)


def per_label_ap_from_probs(probs: np.ndarray, labels: np.ndarray) -> list[float]:
    return [average_precision_1d(probs[:, i], labels[:, i]) for i in range(labels.shape[1])]


def compute_psr_metrics(
    action_logits: torch.Tensor,
    reason_logits: torch.Tensor,
    labels_action: torch.Tensor,
    labels_reason: torch.Tensor,
    threshold: float = 0.5,
) -> PSRMetricBundle:
    action_probs = sigmoid_np(action_logits)
    reason_probs = sigmoid_np(reason_logits)
    action_y = labels_action.detach().cpu().numpy().astype(np.float32)
    reason_y = labels_reason.detach().cpu().numpy().astype(np.float32)
    action_f1 = per_label_f1_from_probs(action_probs, action_y, threshold)
    reason_f1 = per_label_f1_from_probs(reason_probs, reason_y, threshold)
    action_ap = per_label_ap_from_probs(action_probs, action_y)
    reason_ap = per_label_ap_from_probs(reason_probs, reason_y)
    act_mf1 = float(np.mean(action_f1)) if action_f1 else 0.0
    exp_mf1 = float(np.mean(reason_f1)) if reason_f1 else 0.0
    act_map = float(np.mean(action_ap)) if action_ap else 0.0
    exp_map = float(np.mean(reason_ap)) if reason_ap else 0.0
    act_of1 = overall_f1_from_probs(action_probs, action_y, threshold)
    exp_of1 = overall_f1_from_probs(reason_probs, reason_y, threshold)
    return PSRMetricBundle(
        Act_mF1=act_mf1,
        Act_oF1=act_of1,
        Act_mAP=act_map,
        Exp_mF1=exp_mf1,
        Exp_oF1=exp_of1,
        Exp_mAP=exp_map,
        standard_joint=0.5 * act_mf1 + 0.5 * exp_mf1,
        action_primary_score=0.6 * act_mf1 + 0.25 * exp_mf1 + 0.15 * exp_map,
        per_action_F1=[float(x) for x in action_f1],
        per_reason_F1=[float(x) for x in reason_f1],
        per_action_AP=[float(x) for x in action_ap],
        per_reason_AP=[float(x) for x in reason_ap],
    )


def metric_subset(metrics: dict[str, Any]) -> dict[str, float]:
    keys = ["Act_mF1", "Act_oF1", "Act_mAP", "Exp_mF1", "Exp_oF1", "Exp_mAP", "standard_joint", "action_primary_score"]
    return {k: float(metrics.get(k, 0.0)) for k in keys}
