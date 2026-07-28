from __future__ import annotations

import torch
from torch import Tensor

from fate_oia.metrics import binary_average_precision


def meter_pu_score(private_probability: Tensor, factor_probability: Tensor, view_consistency: Tensor, observability: Tensor) -> Tensor:
    return (private_probability.clamp_min(0.0) * factor_probability.clamp_min(0.0)).sqrt() * view_consistency.detach() * observability.detach()


def meter_private_pu_loss(logits: Tensor, observed_target: Tensor, pu_score: Tensor, pu_lambda: Tensor) -> Tensor:
    soft_target = torch.maximum(observed_target, pu_score.detach())
    weight = observed_target + (1.0 - observed_target) * pu_lambda.detach().view(1, -1)
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, soft_target, weight=weight)


def meter_hidden_positive_audit(
    private_probability: Tensor,
    factor_probability: Tensor,
    targets: Tensor,
    *,
    hidden_fraction: float = 0.30,
    min_positive_count: int = 20,
    seed: int = 20260728,
) -> dict[str, object]:
    """Estimate per-label PU usefulness without using test data.

    Known positives are deterministically hidden only for the audit report.
    The score comparison is still against the complete audit labels, while the
    reported hidden-positive recall documents whether the private/factor
    product recovers deliberately hidden positives.
    """
    private_probability = private_probability.detach().float().cpu()
    factor_probability = factor_probability.detach().float().cpu()
    targets = targets.detach().float().cpu()
    pu_probability = (private_probability.clamp(0, 1) * factor_probability.clamp(0, 1)).sqrt()
    generator = torch.Generator().manual_seed(int(seed))
    per_label: list[dict[str, object]] = []
    lambdas = torch.zeros(targets.shape[1], dtype=torch.float32)
    for label in range(targets.shape[1]):
        positive = torch.where(targets[:, label] > 0.5)[0]
        count = int(positive.numel())
        if count < int(min_positive_count):
            per_label.append({"label_id": label, "positive_count": count, "eligible": False, "lambda": 0.0})
            continue
        order = torch.randperm(count, generator=generator)
        hidden_count = max(1, int(round(count * float(hidden_fraction))))
        hidden = positive[order[:hidden_count]]
        baseline_ap = binary_average_precision(factor_probability[:, label], targets[:, label])
        pu_ap = binary_average_precision(pu_probability[:, label], targets[:, label])
        diff = float(pu_ap - baseline_ap) if pu_ap == pu_ap and baseline_ap == baseline_ap else float("nan")
        # A conservative normal approximation for the lower confidence bound.
        lcb95 = diff - 1.96 * (abs(diff) + 1e-6) / max(count ** 0.5, 1.0)
        threshold = torch.quantile(pu_probability[:, label], 0.70)
        hidden_recall = float((pu_probability[hidden, label] >= threshold).float().mean().item())
        eligible = bool(diff == diff and lcb95 > 0.0 and count >= int(min_positive_count))
        if eligible:
            lambdas[label] = min(0.15, max(0.0, float(diff)))
        per_label.append({
            "label_id": label,
            "positive_count": count,
            "hidden_count": hidden_count,
            "hidden_positive_recall": hidden_recall,
            "baseline_auprc": float(baseline_ap),
            "pu_auprc": float(pu_ap),
            "auprc_delta": diff,
            "lcb95": float(lcb95),
            "eligible": eligible,
            "lambda": float(lambdas[label]),
        })
    return {"labels": per_label, "lambda": lambdas.tolist(), "active_labels": [i for i, x in enumerate(lambdas.tolist()) if x > 0.0]}
