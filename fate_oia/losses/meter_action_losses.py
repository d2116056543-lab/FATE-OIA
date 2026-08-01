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
    visual_logits: Tensor | None = None,
    normalizer: Tensor | None = None,
    margin: float = 0.10,
    hard_temperature: float = 0.20,
    **_: float,
) -> Tensor:
    """Train credit on visual hard pairs without reoptimising the anchor.

    A residual should repair pairs that the visual branch orders incorrectly or
    ambiguously. Ranking every positive delta above every negative delta makes
    the residual a global label bias, which improves neither evidence use nor
    fixed-threshold stability. The visual branch is detached here so this term
    updates only the action-credit route.
    """
    if hard_temperature <= 0.0:
        raise ValueError("hard_temperature must be positive")
    delta = action_evidence_delta
    if normalizer is not None:
        scale = normalizer.detach().to(delta)
        delta = delta / scale.clamp_min(1e-6)
    delta = torch.tanh(delta)
    visual = torch.zeros_like(delta) if visual_logits is None else visual_logits.detach()
    if visual.shape != delta.shape:
        raise ValueError("visual_logits must match action_evidence_delta")
    corrected = visual + delta
    terms: list[Tensor] = []
    for action_id in range(delta.shape[1]):
        positive = target[:, action_id] > 0.5
        negative = ~positive
        if bool(positive.any()) and bool(negative.any()):
            visual_gap = (
                visual[positive, action_id, None]
                - visual[negative, action_id]
            )
            corrected_gap = (
                corrected[positive, action_id, None]
                - corrected[negative, action_id]
            )
            # Misranked and close pairs get the learning budget. Easy visual
            # pairs remain an anchor rather than a target for a global shift.
            hard_weight = torch.sigmoid(
                (float(margin) - visual_gap) / float(hard_temperature)
            ).detach()
            hinge = torch.relu(float(margin) - corrected_gap)
            terms.append((hard_weight * hinge).sum() / hard_weight.sum().clamp_min(1e-6))
    return torch.stack(terms).mean() if terms else delta.new_zeros(())


def action_nonregression_loss(
    visual_logits: Tensor,
    action_evidence_delta: Tensor,
    target: Tensor,
    *,
    min_margin: float = 0.05,
    confidence_quantile: float = 0.75,
    boundary_margin: float = 0.02,
) -> Tensor:
    """Protect both confident and correct near-boundary visual decisions.

    The high-confidence term preserves the visual margin.  The boundary term
    prevents the evidence residual from turning an already-correct fixed-zero
    decision into the wrong class, while still allowing it to repair visual
    errors.  Both terms use a detached visual anchor, so gradients remain
    confined to the action evidence route.
    """
    if not 0.0 <= float(confidence_quantile) <= 1.0:
        raise ValueError("confidence_quantile must be in [0, 1]")
    if float(boundary_margin) < 0.0:
        raise ValueError("boundary_margin must be non-negative")
    sign = target * 2.0 - 1.0
    visual_margin = sign * visual_logits.detach()
    final_margin = visual_margin + sign * action_evidence_delta
    correct_visual = visual_margin >= 0.0
    # The high-confidence branch below already protects margins at or above
    # min_margin. Restrict the boundary term to correct low-margin examples
    # so the same regression cannot be penalized twice.
    boundary_mask = correct_visual & (visual_margin < float(min_margin))
    boundary = (
        torch.relu(float(boundary_margin) - final_margin)[boundary_mask].mean()
        if bool(boundary_mask.any())
        else action_evidence_delta.new_zeros(())
    )
    eligible = visual_margin >= float(min_margin)
    if not bool(eligible.any()):
        return boundary
    threshold = torch.quantile(
        visual_margin[eligible].detach(), float(confidence_quantile)
    ).clamp_min(float(min_margin))
    mask = visual_margin >= threshold
    confident = (
        torch.relu(visual_margin - final_margin)[mask].mean()
        if bool(mask.any())
        else action_evidence_delta.new_zeros(())
    )
    # Protect every already-correct visual prediction from a sign flip. The
    # boundary and high-confidence guards already cover their own rows, so the
    # extra term applies only to the former middle-confidence gap.
    middle = correct_visual & ~boundary_mask & ~mask
    sign_guard = (
        torch.relu(-final_margin)[middle].mean()
        if bool(middle.any())
        else action_evidence_delta.new_zeros(())
    )
    return confident + boundary + sign_guard


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
        visual_logits=output["action_logits_visual"],
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
        output["action_logits_visual"],
        output["action_evidence_delta"],
        target,
        min_margin=float(weights.get("action_nonreg_min_margin", 0.05)),
        confidence_quantile=float(
            weights.get("action_nonreg_confidence_quantile", 0.75)
        ),
        boundary_margin=float(weights.get("action_nonreg_boundary_margin", 0.02)),
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
