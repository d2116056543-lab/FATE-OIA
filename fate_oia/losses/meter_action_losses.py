from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def asymmetric_multilabel_elements(
    logits: Tensor, target: Tensor, gamma_negative: float = 2.0
) -> Tensor:
    probability = torch.sigmoid(logits)
    return -target * torch.log(probability.clamp_min(1e-6)) - (
        (1.0 - target)
        * probability.pow(gamma_negative)
        * torch.log((1.0 - probability).clamp_min(1e-6))
    )


def asymmetric_multilabel_loss(logits: Tensor, target: Tensor) -> Tensor:
    return asymmetric_multilabel_elements(logits, target).mean()


def soft_f1_loss(logits: Tensor, target: Tensor) -> Tensor:
    probability = torch.sigmoid(logits)
    tp = (probability * target).sum(0)
    fp = (probability * (1.0 - target)).sum(0)
    fn = ((1.0 - probability) * target).sum(0)
    return 1.0 - ((2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)).mean()


def action_delta_pairwise_ranking_loss(
    action_evidence_delta: Tensor,
    target: Tensor,
    *,
    normalizer: Tensor | None = None,
    margin: float = 0.10,
    **_: float,
) -> Tensor:
    """HECA credit rank: positive samples require larger signed credit."""
    delta = action_evidence_delta
    if normalizer is not None:
        scale = normalizer.detach().to(delta)
        delta = delta / scale.clamp_min(1e-6)
    # Credit ranking is an ordering objective. Saturating the normalized
    # credit prevents one runaway sample from dominating the shared update.
    delta = torch.tanh(delta)
    terms: list[Tensor] = []
    for action_id in range(delta.shape[1]):
        positive = target[:, action_id] > 0.5
        negative = ~positive
        if bool(positive.any()) and bool(negative.any()):
            difference = delta[positive, action_id, None] - delta[negative, action_id]
            terms.append(torch.relu(float(margin) - difference).mean())
    return torch.stack(terms).mean() if terms else delta.new_zeros(())


def action_nonregression_loss(
    visual_logits: Tensor,
    action_evidence_delta: Tensor,
    target: Tensor,
    *,
    min_margin: float = 0.05,
    confidence_quantile: float = 0.75,
) -> Tensor:
    """Protect the correct visual anchor without updating that anchor itself."""
    if not 0.0 <= float(confidence_quantile) <= 1.0:
        raise ValueError("confidence_quantile must be in [0, 1]")
    sign = target * 2.0 - 1.0
    visual_margin = sign * visual_logits.detach()
    eligible = visual_margin >= float(min_margin)
    if not bool(eligible.any()):
        return action_evidence_delta.new_zeros(())
    threshold = torch.quantile(
        visual_margin[eligible].detach(), float(confidence_quantile)
    ).clamp_min(float(min_margin))
    mask = visual_margin >= threshold
    if not bool(mask.any()):
        return action_evidence_delta.new_zeros(())
    final_margin = visual_margin + sign * action_evidence_delta
    return torch.relu(visual_margin - final_margin)[mask].mean()


def meter_action_loss(
    output: dict[str, Tensor],
    target: Tensor,
    weights: dict[str, float] | None = None,
) -> dict[str, Tensor]:
    weights = weights or {}
    final = asymmetric_multilabel_loss(output["action_logits_final"], target)
    visual = asymmetric_multilabel_loss(output["action_logits_visual"], target)
    credit_rank = action_delta_pairwise_ranking_loss(
        output["action_evidence_delta"],
        target,
        normalizer=output.get("action_correction_kappa"),
    )
    contribution = output["action_factor_contribution"]
    sign = target * 2.0 - 1.0
    necessity = torch.relu(0.05 - sign * output["action_evidence_delta"]).mean()
    specificity = output.get(
        "action_specificity_loss",
        (contribution.abs().mean(-1) * (1.0 - target)).mean(),
    )
    nonreg = action_nonregression_loss(
        output["action_logits_visual"], output["action_evidence_delta"], target
    )
    soft_f1 = soft_f1_loss(output["action_logits_final"], target)
    cardinality = F.smooth_l1_loss(
        torch.sigmoid(output["action_logits_final"]).sum(-1), target.sum(-1)
    )
    logit_scale = torch.relu(
        output["action_logits_visual"].float().square().mean().sqrt() - 8.0
    ).square()
    terms = {
        "final": final,
        "visual": visual,
        "credit_rank": credit_rank,
        "necessity": necessity,
        "specificity": specificity,
        "nonreg": nonreg,
        "soft_f1": soft_f1,
        "cardinality": cardinality,
        "logit_scale": logit_scale,
    }
    total = sum(
        float(weights.get(key_name, default)) * terms[term_name]
        for term_name, key_name, default in (
            ("final", "action_final", 1.00),
            ("visual", "action_visual", 0.30),
            ("credit_rank", "action_credit_rank", 0.10),
            ("necessity", "action_necessity", 0.05),
            ("specificity", "action_specificity", 0.05),
            ("nonreg", "action_nonreg", 0.05),
            ("soft_f1", "action_soft_f1", 0.03),
            ("cardinality", "action_cardinality", 0.02),
            ("logit_scale", "action_logit_scale", 0.01),
        )
    )
    return {**terms, "total": total}


def meter_action_loss_per_sample(
    output: dict[str, Tensor], target: Tensor, weights: dict[str, float] | None = None
) -> Tensor:
    weights = weights or {}
    return (
        float(weights.get("action_final", 1.0))
        * asymmetric_multilabel_elements(output["action_logits_final"], target).mean(-1)
        + float(weights.get("action_visual", 0.30))
        * asymmetric_multilabel_elements(output["action_logits_visual"], target).mean(-1)
    )
