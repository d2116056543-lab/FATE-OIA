from __future__ import annotations

import torch
import torch.nn.functional as F

from fate_oia.acpr_interactflow.nnpu_calalign import nnpu_binary_loss


def action_soft_ce_loss(logits: torch.Tensor, soft_targets: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    loss = -(soft_targets * F.log_softmax(logits, dim=-1)).sum(-1)
    if weights is not None:
        loss = loss * weights
    return loss.mean()


def action_soft_kl_loss(logits: torch.Tensor, soft_targets: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    probs_log = F.log_softmax(logits, dim=-1)
    loss = (soft_targets.clamp_min(1e-9) * (soft_targets.clamp_min(1e-9).log() - probs_log)).sum(-1)
    if weights is not None:
        loss = loss * weights
    return loss.mean()


def action_majority_ce_loss(logits: torch.Tensor, majority: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    loss = F.cross_entropy(logits, majority.long(), reduction="none")
    if weights is not None:
        loss = loss * weights
    return loss.mean()


def exp29_masked_asl_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    gamma_neg: float = 4.0,
    gamma_pos: float = 0.0,
    clip: float = 0.05,
) -> torch.Tensor:
    """Masked asymmetric loss for sparse PSI explanation labels.

    The previous implementation used plain BCE under the ASL name. With 29
    sparse labels this makes the easiest optimum "predict no positives" at a
    fixed 0.5 threshold. ASL keeps positive gradients strong while down-weighting
    easy negatives, without using test-derived thresholds.
    """
    targets = targets.float()
    mask = mask.float()
    probs = torch.sigmoid(logits)
    pos_probs = probs
    neg_probs = 1.0 - probs
    if clip > 0:
        neg_probs = (neg_probs + clip).clamp(max=1.0)

    pos_loss = targets * torch.log(pos_probs.clamp_min(1e-8))
    neg_loss = (1.0 - targets) * torch.log(neg_probs.clamp_min(1e-8))

    if gamma_neg > 0 or gamma_pos > 0:
        pos_weight = (1.0 - probs).pow(gamma_pos)
        neg_weight = probs.pow(gamma_neg)
        loss = pos_loss * pos_weight + neg_loss * neg_weight
    else:
        loss = pos_loss + neg_loss
    return (-(loss) * mask).sum() / mask.sum().clamp_min(1.0)


def exp29_soft_f1_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    targets = targets.float()
    mask = mask.float()
    probs = torch.sigmoid(logits) * mask
    targets = targets * mask
    tp = (probs * targets).sum(dim=0)
    fp = (probs * (1.0 - targets) * mask).sum(dim=0)
    fn = ((1.0 - probs) * targets).sum(dim=0)
    valid = (targets.sum(dim=0) > 0).float()
    f1 = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    return 1.0 - (f1 * valid).sum() / valid.sum().clamp_min(1.0)


def exp29_positive_unlabeled_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, positive_prior: float = 0.08) -> torch.Tensor:
    return nnpu_binary_loss(logits, targets.float(), mask.float(), positive_prior=positive_prior)


def predicate_bag_loss(predicate_logits: torch.Tensor, weak_targets: torch.Tensor | None = None, mask: torch.Tensor | None = None) -> torch.Tensor:
    if weak_targets is None:
        return predicate_logits.new_zeros(())
    loss = F.binary_cross_entropy_with_logits(predicate_logits, weak_targets.float(), reduction="none")
    if mask is not None:
        loss = loss * mask.float()
        return loss.sum() / mask.float().sum().clamp_min(1.0)
    return loss.mean()


def flow_sparsity_loss(flow_edges: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(flow_edges, dim=-1)
    entropy = -(probs.clamp_min(1e-9).log() * probs).sum(-1)
    return entropy.mean()


def ledger_identity_loss(identity_error: torch.Tensor) -> torch.Tensor:
    return identity_error.float()


def non_degradation_soft_kl_hinge_loss(
    final_logits: torch.Tensor,
    base_logits: torch.Tensor,
    soft_targets: torch.Tensor,
    margin: float = 0.01,
) -> torch.Tensor:
    final_kl = action_soft_kl_loss(final_logits, soft_targets, weights=None)
    with torch.no_grad():
        base_kl = action_soft_kl_loss(base_logits, soft_targets, weights=None)
    return F.relu(final_kl - base_kl + margin)


def predicate_pu_loss(predicate_logits_trajectory: torch.Tensor, weak_targets: torch.Tensor | None = None, mask: torch.Tensor | None = None) -> torch.Tensor:
    if weak_targets is None:
        # Conservative PU proxy: avoid all-zero collapse by treating confident
        # trajectory activations as unlabeled rather than negatives.
        weak_targets = torch.zeros_like(predicate_logits_trajectory)
        mask = torch.ones_like(predicate_logits_trajectory)
    return nnpu_binary_loss(predicate_logits_trajectory, weak_targets.float(), (mask if mask is not None else torch.ones_like(predicate_logits_trajectory)).float(), positive_prior=0.08)


def exp29_pu_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, positive_prior: float = 0.08) -> torch.Tensor:
    return nnpu_binary_loss(logits, targets.float(), mask.float(), positive_prior=positive_prior)


def exp29_positive_rate_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, pi_min: float = 0.03, pi_max: float = 0.35) -> torch.Tensor:
    known_pos = ((targets > 0.5) & (mask > 0.5)).float()
    valid = (mask > 0.5).float()
    prior = known_pos.sum(0) / valid.sum(0).clamp_min(1.0)
    prior = prior.clamp(pi_min, pi_max).detach()
    pred = torch.sigmoid(logits).mean(0)
    return (pred - prior).abs().mean()


def exp29_cardinality_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    target_card = ((targets > 0.5) & (mask > 0.5)).float().sum(-1).mean().detach()
    pred_card = torch.sigmoid(logits).sum(-1).mean()
    return (pred_card - target_card).abs()


def exp29_pairwise_rank_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, margin: float = 0.10) -> torch.Tensor:
    positive = (targets > 0.5) & (mask > 0.5)
    known_negative = (targets <= 0.5) & (mask > 0.5)
    rows = []
    for b in range(logits.shape[0]):
        pos = logits[b][positive[b]]
        neg = logits[b][known_negative[b]]
        if pos.numel() == 0 or neg.numel() == 0:
            continue
        rows.append(F.relu(margin - (pos[:, None] - neg[None, :])).mean())
    if not rows:
        return logits.new_zeros(())
    return torch.stack(rows).mean()


def contribution_alignment_js_loss(contribution_terms: dict[str, torch.Tensor]) -> torch.Tensor:
    components = [v.softmax(-1) for k, v in contribution_terms.items() if k in {"visual", "motion", "predicate", "flow"}]
    if len(components) < 2:
        return next(iter(contribution_terms.values())).new_zeros(())
    mean = torch.stack(components, 0).mean(0).clamp_min(1e-8)
    js = torch.stack([(p.clamp_min(1e-8) * (p.clamp_min(1e-8).log() - mean.log())).sum(-1) for p in components], 0).mean()
    return js


def exp29_ledger_alignment_js_loss(exp_attention: torch.Tensor, gated_state_contributions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    contrib = gated_state_contributions.abs().sum(-1)
    q = contrib / contrib.sum(-1, keepdim=True).clamp_min(1e-8)
    p = exp_attention.clamp_min(1e-8)
    q = q.unsqueeze(1).expand_as(p).clamp_min(1e-8)
    valid = ((targets > 0.5) & (mask > 0.5)).float()
    if valid.sum() <= 0:
        return exp_attention.new_zeros(())
    m = 0.5 * (p + q)
    js = 0.5 * (p * (p.log() - m.log())).sum(-1) + 0.5 * (q * (q.log() - m.log())).sum(-1)
    return (js * valid).sum() / valid.sum().clamp_min(1.0)


def lag_entropy_loss(lag_weights: torch.Tensor) -> torch.Tensor:
    return (-(lag_weights.clamp_min(1e-9).log() * lag_weights).sum(-1)).mean()


def intervention_margin_loss(selected_drop: torch.Tensor | None = None, random_drop: torch.Tensor | None = None, margin: float = 0.01) -> torch.Tensor:
    if selected_drop is None or random_drop is None:
        return torch.tensor(0.0)
    return F.relu(margin - (selected_drop - random_drop)).mean()


def interaction_state_semantic_loss(state_logits: torch.Tensor, action_majority: torch.Tensor) -> torch.Tensor:
    """Weakly align interaction-state groups with the PSI action target.

    The PSI package does not provide dense interaction-state labels. The plan
    still requires the state branch to receive a semantic training signal, so we
    use a conservative action-derived weak target instead of leaving the loss at
    zero. Group order follows configs/acpr_interactflow_state_grammar.yaml.
    """
    if state_logits.numel() == 0:
        return state_logits.new_zeros(())
    b, g = state_logits.shape
    target = state_logits.new_zeros(b, g)
    action = action_majority.long().clamp_min(0)

    # maintain_speed: drivable/ego/global context should be active.
    maintain = action == 0
    if g > 4:
        target[maintain, 4] = 1.0
    if g > 5:
        target[maintain, 5] = 1.0
    if g > 7:
        target[maintain, 7] = 0.5

    # reduce_speed: caution from front object, actor crossing, lane or geometry.
    reduce = action == 1
    for idx in (1, 2, 3, 6):
        if g > idx:
            target[reduce, idx] = 1.0

    # stop_car: traffic/front-object/actor-crossing/ego-stop groups.
    stop = action == 2
    for idx in (0, 1, 2, 5):
        if g > idx:
            target[stop, idx] = 1.0

    # Unknown/extra classes, if any, get a weak global-context target only.
    unknown = action > 2
    if g > 7:
        target[unknown, 7] = 0.5
    return F.binary_cross_entropy_with_logits(state_logits, target)


def temporal_consistency_loss(predicate_probs_trajectory: torch.Tensor, motion_tokens: torch.Tensor | None = None) -> torch.Tensor:
    """Smooth predicate trajectories without forcing them constant.

    Adjacent predicate probabilities should not jitter frame-to-frame, but the
    penalty is Huber-style so real motion changes can still pass through. Motion
    tokens reduce the penalty for highly dynamic clips.
    """
    if predicate_probs_trajectory.ndim != 3 or predicate_probs_trajectory.shape[1] < 2:
        return predicate_probs_trajectory.new_zeros(())
    diff = predicate_probs_trajectory[:, 1:] - predicate_probs_trajectory[:, :-1]
    penalty = F.smooth_l1_loss(diff, torch.zeros_like(diff), reduction="none").mean(-1)
    if motion_tokens is not None and motion_tokens.ndim == 3 and motion_tokens.shape[1] >= 2:
        motion_delta = (motion_tokens[:, 1:] - motion_tokens[:, :-1]).norm(dim=-1)
        motion_gate = torch.exp(-motion_delta.detach()).clamp(0.2, 1.0)
        penalty = penalty * motion_gate
    return penalty.mean()


DEFAULT_INTERACTFLOW_LOSS_WEIGHTS = {
    "action_final_soft_kl": 1.00,
    "action_global_soft_kl": 0.40,
    "ledger_residual_soft_kl": 0.15,
    "non_degradation_soft_kl_hinge": 0.08,
    "benefit_gate_advantage_bce": 0.04,
    "predicate_pu": 0.08,
    "predicate_structural_weak": 0.03,
    "exp29_raw_asl": 0.12,
    "exp29_calibrated_asl": 0.20,
    "exp29_pu": 0.04,
    "exp29_soft_f1": 0.08,
    "exp29_positive_rate": 0.04,
    "exp29_cardinality": 0.02,
    "exp29_pairwise_rank": 0.04,
    "exp29_ledger_alignment_js": 0.06,
    "interaction_state_semantic": 0.04,
    "response_lag_consistency": 0.01,
    "response_lag_temporal_consistency": 0.02,
    "contribution_alignment_js": 0.0,
    "temporal_consistency": 0.02,
    "gate_prior_noncollapse": 0.004,
    "factor_sparsity": 0.001,
}


def compute_interactflow_losses(output, batch, weights: dict[str, float] | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    w = {**DEFAULT_INTERACTFLOW_LOSS_WEIGHTS, **(weights or {})}
    terms: dict[str, torch.Tensor] = {}
    terms["action_final_soft_kl"] = action_soft_kl_loss(output.action_logits, batch.action_soft, batch.paper_effective_weight)
    terms["action_global_soft_kl"] = action_soft_kl_loss(output.ledger.global_logits, batch.action_soft, batch.paper_effective_weight)
    residual_logits = output.ledger.flow_delta_logits + output.ledger.calibration_delta
    terms["ledger_residual_soft_kl"] = action_soft_kl_loss(output.action_logits - residual_logits.detach(), batch.action_soft, batch.paper_effective_weight)
    calibrated_exp_logits = output.aux.get("exp29_logits_calibrated", output.exp29_logits)
    terms["exp29_raw_asl"] = exp29_masked_asl_loss(output.exp29_logits, batch.exp29, batch.exp29_mask)
    terms["exp29_masked_asl"] = terms["exp29_raw_asl"]
    terms["exp29_calibrated_asl"] = exp29_masked_asl_loss(calibrated_exp_logits, batch.exp29, batch.exp29_mask)
    terms["exp29_soft_f1"] = exp29_soft_f1_loss(calibrated_exp_logits, batch.exp29, batch.exp29_mask)
    terms["exp29_pu"] = exp29_pu_loss(calibrated_exp_logits, batch.exp29, batch.exp29_mask)
    terms["exp29_positive_rate"] = exp29_positive_rate_loss(calibrated_exp_logits, batch.exp29, batch.exp29_mask)
    terms["exp29_cardinality"] = exp29_cardinality_loss(calibrated_exp_logits, batch.exp29, batch.exp29_mask)
    terms["exp29_pairwise_rank"] = exp29_pairwise_rank_loss(calibrated_exp_logits, batch.exp29, batch.exp29_mask)
    terms["predicate_pu"] = predicate_pu_loss(output.predicates.predicate_logits_trajectory)
    predicate_rate = torch.sigmoid(output.predicates.predicate_logits_trajectory).mean()
    terms["predicate_structural_weak"] = temporal_consistency_loss(output.predicates.predicate_probs_trajectory) + (predicate_rate - 0.10).abs()
    terms["interaction_state_semantic"] = interaction_state_semantic_loss(output.flow.state_logits, batch.action_majority)
    terms["factor_sparsity"] = flow_sparsity_loss(output.flow.flow_edges)
    terms["ledger_identity"] = ledger_identity_loss(output.ledger.identity_error)
    base_logits = output.ledger.visual_logits + 0.35 * output.ledger.motion_logits + 0.25 * output.ledger.predicate_logits
    terms["non_degradation_soft_kl_hinge"] = non_degradation_soft_kl_hinge_loss(output.action_logits, base_logits, batch.action_soft)
    if output.ledger.benefit_target is not None:
        target = output.ledger.benefit_target
        pred = output.ledger.benefit_gate.mean(-1, keepdim=True)
        terms["benefit_gate_advantage_bce"] = F.binary_cross_entropy(pred.clamp(1e-6, 1 - 1e-6), target)
    else:
        terms["benefit_gate_advantage_bce"] = output.action_logits.new_zeros(())
    terms["exp29_ledger_alignment_js"] = exp29_ledger_alignment_js_loss(
        output.exp29.cluster_attention_to_factors,
        output.ledger.gated_state_contributions,
        batch.exp29,
        batch.exp29_mask,
    )
    terms["contribution_alignment_js"] = contribution_alignment_js_loss(output.ledger.contribution_terms)
    terms["response_lag_consistency"] = lag_entropy_loss(output.flow.lag_weights)
    terms["response_lag_temporal_consistency"] = temporal_consistency_loss(output.flow.factor_probs_trajectory)
    terms["temporal_consistency"] = temporal_consistency_loss(
        output.predicates.predicate_probs_trajectory,
        output.visual.fast_motion_tokens,
    )
    terms["gate_prior_noncollapse"] = lag_entropy_loss(output.ledger.gate)
    total = sum(float(w.get(k, 0.0)) * v for k, v in terms.items())
    terms["total_loss"] = total
    return total, terms
