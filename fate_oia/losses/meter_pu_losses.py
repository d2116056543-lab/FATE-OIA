from __future__ import annotations

import torch
from torch import Tensor

from fate_oia.metrics import binary_average_precision


def meter_pu_score(
    global_probability: Tensor,
    positive_state_probability: Tensor,
    reliability: Tensor,
    observability: Tensor | None = None,
) -> Tensor:
    score = (
        global_probability.clamp(0, 1)
        * positive_state_probability.clamp(0, 1)
        * reliability.detach().clamp(0, 1)
    )
    return score if observability is None else score * observability.detach().clamp(0, 1)


def meter_private_pu_loss(logits: Tensor, observed_target: Tensor, pu_score: Tensor, pu_lambda: Tensor) -> Tensor:
    """Apply PU supervision only to labels admitted by the audit schedule.

    A zero lambda means that a label is not yet evidence-backed, so its PU
    branch must be exactly inactive.  Positive labels retain weight 1.0 for
    admitted labels; unlabeled entries use the per-label lambda.
    """
    if logits.ndim != 2 or observed_target.shape != logits.shape or pu_score.shape != logits.shape:
        raise ValueError("METER PU tensors must share shape [B, reason_dim]")
    if pu_lambda.ndim != 1 or pu_lambda.numel() != logits.shape[1]:
        raise ValueError("pu_lambda must have shape [reason_dim]")
    soft_target = torch.maximum(observed_target, pu_score.detach())
    lambda_by_label = pu_lambda.detach().to(device=logits.device, dtype=logits.dtype).view(1, -1)
    active = (lambda_by_label > 0.0).to(dtype=logits.dtype)
    weight = active * (observed_target + (1.0 - observed_target) * lambda_by_label)
    elementwise = torch.nn.functional.binary_cross_entropy_with_logits(logits, soft_target, reduction="none")
    return (elementwise * weight).sum() / weight.sum().clamp_min(1.0)


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
    Deliberately hidden positives are evaluated against originally observed
    zeros. Visible positives are excluded, so eligibility measures recovery
    rather than ordinary supervised classification.
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
        visible = positive[order[hidden_count:]]
        audit_mask = torch.ones(targets.shape[0], dtype=torch.bool)
        audit_mask[visible] = False
        audit_target = torch.zeros(int(audit_mask.sum()), dtype=torch.float32)
        retained_indices = torch.where(audit_mask)[0]
        hidden_positions = torch.isin(retained_indices, hidden)
        audit_target[hidden_positions] = 1.0
        baseline_score = factor_probability[audit_mask, label]
        pu_score = pu_probability[audit_mask, label]
        baseline_ap = binary_average_precision(baseline_score, audit_target)
        pu_ap = binary_average_precision(pu_score, audit_target)
        diff = float(pu_ap - baseline_ap) if pu_ap == pu_ap and baseline_ap == baseline_ap else float("nan")
        # A conservative normal approximation for the lower confidence bound.
        lcb95 = diff - 1.96 * (abs(diff) + 1e-6) / max(count ** 0.5, 1.0)
        threshold = torch.quantile(pu_score, 0.70)
        hidden_recall = float((pu_probability[hidden, label] >= threshold).float().mean().item())
        eligible = bool(diff == diff and lcb95 > 0.0 and count >= int(min_positive_count))
        if eligible:
            lambdas[label] = min(0.15, max(0.0, float(diff)))
        per_label.append({
            "label_id": label,
            "positive_count": count,
            "hidden_count": hidden_count,
            "hidden_positive_recall": hidden_recall,
            "hidden_positive_auprc": float(pu_ap),
            "audit_target": "deliberately_hidden_positive_vs_observed_zero",
            "baseline_auprc": float(baseline_ap),
            "pu_auprc": float(pu_ap),
            "auprc_delta": diff,
            "lcb95": float(lcb95),
            "eligible": eligible,
            "lambda": float(lambdas[label]),
        })
    return {"labels": per_label, "lambda": lambdas.tolist(), "active_labels": [i for i, x in enumerate(lambdas.tolist()) if x > 0.0]}
