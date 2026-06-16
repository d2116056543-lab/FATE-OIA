from __future__ import annotations

import torch


def _safe_div(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    return num / den.clamp_min(1e-8)


def _per_action_stats(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> dict:
    labels = labels.float()
    pred = (torch.sigmoid(logits) >= float(threshold)).float()
    tp = (pred * labels).sum(0)
    fp = (pred * (1.0 - labels)).sum(0)
    fn = ((1.0 - pred) * labels).sum(0)
    f1 = _safe_div(2.0 * tp, 2.0 * tp + fp + fn)
    op = pred.sum()
    og = labels.sum()
    otp = (pred * labels).sum()
    of1 = _safe_div(2.0 * otp, op + og)
    pred_count = pred.sum(1)
    label_count = labels.sum(1)
    combo = label_count > 1
    combo_single = ((pred_count == 1) & combo).float().mean() if combo.any() else torch.tensor(0.0)
    superset = ((pred >= labels).all(1) & (pred_count > label_count)).float().mean()
    all_high = (pred_count >= logits.shape[1]).float().mean()
    return {
        "Act_mF1": float(f1.mean().detach().cpu()),
        "Act_oF1": float(of1.detach().cpu()),
        "per_action_F1": [float(x) for x in f1.detach().cpu()],
        "predicted_positive_rate_per_action": [float(x) for x in pred.mean(0).detach().cpu()],
        "false_positive_rate_per_action": [float(x) for x in _safe_div(fp, (1.0 - labels).sum(0)).detach().cpu()],
        "false_negative_rate_per_action": [float(x) for x in _safe_div(fn, labels.sum(0)).detach().cpu()],
        "all_high_rate": float(all_high.detach().cpu()),
        "superset_pred_rate": float(superset.detach().cpu()),
        "combo_gt_single_pred_rate": float(combo_single.detach().cpu()),
        "action_count_mean": float(pred_count.mean().detach().cpu()),
    }


def compute_candidate_metrics(candidate_logits: dict[str, torch.Tensor], action_labels: torch.Tensor, threshold: float = 0.5) -> dict:
    return {name: _per_action_stats(logits, action_labels, threshold=threshold) for name, logits in candidate_logits.items()}


def compare_candidates_to_fallback(metrics_by_candidate: dict, fallback_name: str = "fallback") -> dict:
    fallback = metrics_by_candidate[fallback_name]
    out = {"fallback": fallback, "candidate_delta_f1": {}, "best_candidate_per_action": []}
    n = len(fallback["per_action_F1"])
    for name, metrics in metrics_by_candidate.items():
        if name == fallback_name:
            continue
        out["candidate_delta_f1"][name] = [
            float(metrics["per_action_F1"][i] - fallback["per_action_F1"][i]) for i in range(n)
        ]
    names = [n for n in metrics_by_candidate if n != fallback_name]
    for idx in range(n):
        best = max(names, key=lambda nm: metrics_by_candidate[nm]["per_action_F1"][idx]) if names else fallback_name
        out["best_candidate_per_action"].append(best)
    return out
