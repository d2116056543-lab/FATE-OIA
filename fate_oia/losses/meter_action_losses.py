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
    # `action_logits_final` deploys the raw bounded residual directly.  The
    # legacy kappa argument remains accepted for checkpoint/config compatibility
    # but must not re-scale the objective into a different logit space.
    del normalizer
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
            # Only pairs already misranked or inside the visual margin receive
            # residual credit. This prevents a harmful residual from spending
            # budget on an already-easy visual pair and becoming a label bias.
            hard_pair = visual_gap < float(margin)
            if bool(hard_pair.any()):
                hinge = torch.relu(float(margin) - corrected_gap[hard_pair])
                terms.append(hinge.mean())
    # Keep the zero connected to the residual so an all-easy batch is a valid
    # no-op backward pass with exactly zero action-credit gradient.
    return torch.stack(terms).mean() if terms else delta.sum() * 0.0


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
    # Fixed-threshold evaluation predicts positive at logit zero. Only a
    # negative target needs a strictly-positive buffer: a positive target at
    # zero is already a correct prediction under the deployed rule.
    fixed_threshold_buffer = torch.where(
        target < 0.5,
        final_margin.new_tensor(1e-4),
        final_margin.new_zeros(()),
    )
    sign_guard = (
        torch.relu(fixed_threshold_buffer - final_margin)[middle].mean()
        if bool(middle.any())
        else action_evidence_delta.new_zeros(())
    )
    return confident + boundary + sign_guard


def action_target_effectiveness_loss(
    visual_logits: Tensor,
    action_evidence_delta: Tensor,
    counterfactual_action_delta: Tensor,
    target: Tensor,
    action_factor_weights: Tensor,
    factor_reliability: Tensor,
    factor_action_ownership: Tensor,
    factor_groundable_mask: Tensor,
    *,
    margin: float = 0.02,
    relative_margin_fraction: float | None = None,
    min_margin: float = 0.002,
    max_margin: float = 0.02,
    directional_relative_margin_fraction: float = 0.01,
    directional_min_margin: float = 0.001,
    directional_max_margin: float = 0.005,
    hard_visual_margin: float = 0.25,
    min_support: float = 0.10,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Require selected evidence to improve a supported hard action decision.

    This is intentionally a *route comparison*, not a label-signed penalty on
    every residual.  The corrupted route is a detached control constructed
    from the same image, so the loss can only improve the selected action
    transport route.  Both visual logits and support gates are detached: the
    objective cannot win by modifying the visual anchor or by hiding support.
    """
    if action_evidence_delta.shape != target.shape:
        raise ValueError("action_evidence_delta must match action target")
    if visual_logits.shape != target.shape:
        raise ValueError("visual_logits must match action target")
    if counterfactual_action_delta.shape != target.shape:
        raise ValueError("counterfactual_action_delta must match action target")
    if action_factor_weights.shape[:2] != target.shape:
        raise ValueError("action_factor_weights must be [B,A,F]")
    if factor_reliability.shape != (
        target.shape[0],
        action_factor_weights.shape[-1],
    ):
        raise ValueError("factor_reliability must be [B,F]")
    factor_count = action_factor_weights.shape[-1]
    if factor_action_ownership.reshape(-1).shape[0] != factor_count:
        raise ValueError("factor_action_ownership must be [F]")
    if factor_groundable_mask.reshape(-1).shape[0] != factor_count:
        raise ValueError("factor_groundable_mask must be [F]")
    if (
        margin < 0.0
        or min_margin < 0.0
        or max_margin < min_margin
        or directional_relative_margin_fraction < 0.0
        or directional_min_margin < 0.0
        or directional_max_margin < directional_min_margin
        or hard_visual_margin < 0.0
        or min_support < 0.0
    ):
        raise ValueError("target-effectiveness thresholds must be non-negative")
    if relative_margin_fraction is not None and relative_margin_fraction < 0.0:
        raise ValueError("relative_margin_fraction must be non-negative")

    support = (
        action_factor_weights.detach()
        * factor_reliability.detach().clamp(0.0, 1.0).unsqueeze(1)
        * factor_action_ownership.detach().to(action_factor_weights).reshape(1, 1, -1).clamp(0.0, 1.0)
        * factor_groundable_mask.detach().to(action_factor_weights).reshape(1, 1, -1).clamp(0.0, 1.0)
    ).sum(-1)
    sign = target * 2.0 - 1.0
    visual_margin = sign * visual_logits.detach()
    # A bounded residual is credible only near the visual decision boundary;
    # easy anchor decisions are protected by the non-regression objective.
    active = (
        visual_margin.abs() <= float(hard_visual_margin)
    ) & (support >= float(min_support))
    selected_margin = sign * (visual_logits.detach() + action_evidence_delta)
    control_margin = sign * (
        visual_logits.detach() + counterfactual_action_delta.detach()
    )
    target_effect = selected_margin - control_margin
    visual_rms = visual_logits.detach().float().square().mean(0).sqrt()
    if relative_margin_fraction is None:
        required_margin = target_effect.new_full(target_effect.shape, float(margin))
    else:
        required_margin = (
            float(relative_margin_fraction) * visual_rms
        ).clamp(float(min_margin), float(max_margin)).to(target_effect).unsqueeze(0)
    if bool(active.any()):
        contrastive_loss = torch.relu(
            required_margin.expand_as(target_effect)[active] - target_effect[active]
        ).mean()

        # A selected route can be less harmful than its corruption while still
        # moving a positive action in the wrong direction.  The old relative
        # objective allowed exactly that failure mode.  On the same certified,
        # near-boundary route, require the selected delta itself to agree with
        # the action target.  Positive and negative active examples are
        # averaged per action first so common negative labels cannot turn this
        # into a global suppression bias.
        directional_margin = (
            float(directional_relative_margin_fraction) * visual_rms
        ).clamp(
            float(directional_min_margin), float(directional_max_margin)
        ).to(action_evidence_delta).unsqueeze(0)
        directional_effect = sign * action_evidence_delta
        directional_terms: list[Tensor] = []
        for action_id in range(target.shape[1]):
            for is_positive in (False, True):
                class_active = active[:, action_id] & (
                    (target[:, action_id] > 0.5) == is_positive
                )
                if bool(class_active.any()):
                    directional_terms.append(
                        torch.relu(
                            directional_margin[:, action_id].expand_as(
                                directional_effect[:, action_id]
                            )[class_active]
                            - directional_effect[:, action_id][class_active]
                        ).mean()
                    )
        directional_loss = (
            torch.stack(directional_terms).mean()
            if directional_terms
            else action_evidence_delta.sum() * 0.0
        )
        loss = contrastive_loss + directional_loss
    else:
        # Preserve a valid no-op backward path for batches with no certified
        # factor support rather than inventing a global action prior.
        loss = action_evidence_delta.sum() * 0.0
        contrastive_loss = loss
        directional_loss = loss
        directional_effect = sign * action_evidence_delta
    stats = {
        "active_count": active.sum().to(action_evidence_delta.dtype),
        "active_fraction": active.float().mean(),
        "support_mean": support.mean(),
        "required_margin_mean": required_margin.mean().detach(),
        "target_effect_mean": (
            target_effect[active].mean().detach()
            if bool(active.any())
            else action_evidence_delta.new_zeros(())
        ),
        "directional_effect_mean": (
            directional_effect[active].mean().detach()
            if bool(active.any())
            else action_evidence_delta.new_zeros(())
        ),
        "contrastive_loss": contrastive_loss.detach(),
        "directional_loss": directional_loss.detach(),
    }
    return loss, stats


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
    counterfactual_delta = output.get("action_counterfactual_delta")
    if counterfactual_delta is None:
        # Audits that invoke the isolated loss graph still receive a connected
        # zero, but production training must supply a real same-image control.
        necessity = output["action_evidence_delta"].sum() * 0.0
        necessity_stats = {
            "active_count": necessity.detach(),
            "active_fraction": necessity.detach(),
            "support_mean": necessity.detach(),
            "required_margin_mean": necessity.detach(),
            "target_effect_mean": necessity.detach(),
        }
    else:
        necessity, necessity_stats = action_target_effectiveness_loss(
            output["action_logits_visual"],
            output["action_evidence_delta"],
            counterfactual_delta,
            target,
            output["action_factor_weights"],
            output["factor_reliability"],
            output["factor_action_ownership"],
            output["factor_groundable_mask"],
            margin=float(weights.get("action_necessity_margin", 0.02)),
            relative_margin_fraction=float(
                weights.get("action_necessity_relative_margin_fraction", 0.05)
            ),
            min_margin=float(weights.get("action_necessity_min_margin", 0.002)),
            max_margin=float(weights.get("action_necessity_max_margin", 0.02)),
            directional_relative_margin_fraction=float(
                weights.get("action_necessity_directional_relative_margin_fraction", 0.01)
            ),
            directional_min_margin=float(
                weights.get("action_necessity_directional_min_margin", 0.001)
            ),
            directional_max_margin=float(
                weights.get("action_necessity_directional_max_margin", 0.005)
            ),
            hard_visual_margin=float(
                weights.get("action_necessity_visual_hard_margin", 0.25)
            ),
            min_support=float(weights.get("action_necessity_min_support", 0.10)),
        )
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
    return {**terms, **{f"necessity_{key}": value for key, value in necessity_stats.items()}, "total": total}


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
