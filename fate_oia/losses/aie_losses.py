from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .asymmetric_loss import asymmetric_loss_with_logits
from fate_oia.models.aie_predicate_naming import spatial_soft_iou


def reason_negative_weight(target: Tensor, counter_confidence: Tensor, floor: float = 0.25) -> Tensor:
    counter = counter_confidence.detach().clamp(0, 1)
    negative = float(floor) + (1.0 - float(floor)) * counter
    return torch.where(target > 0.5, torch.ones_like(negative), negative)


def evidence_censored_reason_asl_loss(logits: Tensor, target: Tensor, counter_confidence: Tensor) -> Tensor:
    weights = reason_negative_weight(target, counter_confidence)
    raw = asymmetric_loss_with_logits(logits, target, reduction="none")
    return (raw * weights).sum() / weights.sum().clamp_min(1.0)


def soft_f1_loss(logits: Tensor, target: Tensor, weights: Tensor | None = None) -> Tensor:
    probs = torch.sigmoid(logits)
    sample_weight = torch.ones_like(target) if weights is None else weights
    tp = (probs * target * sample_weight).sum(0)
    fp = (probs * (1 - target) * sample_weight).sum(0)
    fn = ((1 - probs) * target * sample_weight).sum(0)
    return 1.0 - ((2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)).mean()


def action_cardinality_loss(logits: Tensor, target: Tensor) -> Tensor:
    return (torch.sigmoid(logits).sum(-1) - target.sum(-1)).square().mean()


def reason_ranking_loss(
    logits: Tensor,
    target: Tensor,
    negative_weight: Tensor | None = None,
    margin: float = 0.2,
) -> Tensor:
    positive = target > 0.5
    negative = ~positive
    weights = torch.ones_like(target) if negative_weight is None else negative_weight.detach()
    losses = []
    for row in range(logits.shape[0]):
        if positive[row].any() and negative[row].any():
            raw = F.relu(float(margin) - logits[row][positive[row]][:, None] + logits[row][negative[row]][None])
            pair_weight = weights[row][negative[row]][None].expand_as(raw)
            losses.append((raw * pair_weight).sum() / pair_weight.sum().clamp_min(1e-8))
    return torch.stack(losses).mean() if losses else logits.sum() * 0


def predicate_masked_bce(logits: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (raw * mask).sum() / mask.sum().clamp_min(1.0)


def predicate_masked_asl_loss(
    logits: Tensor,
    positive_target: Tensor,
    positive_mask: Tensor,
    counter_mask: Tensor,
    reliability: Tensor,
) -> Tensor:
    """Masked predicate ASL with explicit counters and fail-closed unknowns."""
    observed = positive_mask.bool() | counter_mask.bool()
    target = positive_target.float()
    raw = asymmetric_loss_with_logits(logits, target, reduction="none")
    positive_weight = reliability.detach().clamp(0.1, 1.0)
    weight = torch.where(positive_mask.bool(), positive_weight, torch.ones_like(positive_weight))
    weight = weight * observed.float()
    return (raw * weight).sum() / weight.sum().clamp_min(1.0)


def predicate_reason_alignment_pu_loss(
    predicate_probs: Tensor,
    reason_targets: Tensor,
    grammar_positive_matrix: Tensor,
    grammar_contradiction_matrix: Tensor,
    negative_weight: Tensor,
) -> Tensor:
    pos_support = predicate_probs @ grammar_positive_matrix.t().to(predicate_probs.device, predicate_probs.dtype)
    neg_support = predicate_probs @ grammar_contradiction_matrix.t().to(predicate_probs.device, predicate_probs.dtype)
    support = (pos_support - neg_support).clamp(-1, 1)
    positive = reason_targets.float()
    weights = torch.where(positive > 0.5, torch.ones_like(positive), negative_weight.detach())
    raw = positive * F.softplus(-support) + (1 - positive) * F.softplus(support)
    return (raw * weights).sum() / weights.sum().clamp_min(1.0)


def predicate_map_loss(attention: Tensor, target_map: Tensor, map_mask: Tensor) -> Tensor:
    target = target_map / target_map.sum(-1, keepdim=True).clamp_min(1e-8)
    prediction = attention / attention.sum(-1, keepdim=True).clamp_min(1e-8)
    kl = (target * (target.clamp_min(1e-8).log() - prediction.clamp_min(1e-8).log())).sum(-1)
    dice = 1 - (2 * (prediction * target).sum(-1) + 1e-6) / (prediction.square().sum(-1) + target.square().sum(-1) + 1e-6)
    raw = 0.5 * kl + 0.5 * dice
    return (raw * map_mask).sum() / map_mask.sum().clamp_min(1.0)


def target_signed_margin(logits: Tensor, target: Tensor) -> Tensor:
    return (2.0 * target - 1.0) * logits


def counterfactual_necessity_loss(selected_drop: Tensor, control_drop: Tensor, valid: Tensor, margin: float = 0.05) -> Tensor:
    raw = F.softplus(float(margin) - selected_drop + control_drop)
    return (raw * valid).sum() / valid.sum().clamp_min(1.0)


def contribution_effect_loss(supportive_contribution: Tensor, selected_minus_control: Tensor, valid: Tensor) -> Tensor:
    raw = F.smooth_l1_loss(supportive_contribution, selected_minus_control.detach(), reduction="none")
    return (raw * valid).sum() / valid.sum().clamp_min(1.0)


def probe_duplicate_loss(evidence_map: Tensor, bounded_contribution: Tensor, action_target: Tensor) -> Tensor:
    supportive = ((2 * action_target - 1)[..., None] * bounded_contribution) > 1e-3
    normalized = evidence_map / evidence_map.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    similarity = torch.einsum("bakn,baln->bakl", normalized, normalized)
    pair_mask = supportive[..., :, None] & supportive[..., None, :]
    pair_mask = pair_mask & ~torch.eye(evidence_map.shape[2], device=evidence_map.device, dtype=torch.bool)[None, None]
    return (similarity * pair_mask).sum() / pair_mask.sum().clamp_min(1)


def predicate_map_compactness_loss(attention: Tensor, grid_hw: tuple[int, int] = (45, 80)) -> Tensor:
    h, w = grid_hw
    maps = attention.reshape(*attention.shape[:-1], h, w)
    horizontal = (maps[..., :, 1:] - maps[..., :, :-1]).abs().mean()
    vertical = (maps[..., 1:, :] - maps[..., :-1, :]).abs().mean()
    return horizontal + vertical


def naming_alignment_loss(
    evidence_map: Tensor,
    quality: Tensor,
    predicate_map_target: Tensor,
    reliable_positive: Tensor,
    supportive: Tensor,
    selected_minus_control: Tensor,
    valid: Tensor,
    margin: float = 0.08,
) -> Tensor:
    eligible = valid.bool() & supportive.bool() & (selected_minus_control > 0)
    if not bool(eligible.any()) or not bool(reliable_positive.any()):
        return evidence_map.sum() * 0
    iou = spatial_soft_iou(evidence_map[..., None, :], predicate_map_target[:, None, None, :, :])
    masked_iou = iou.masked_fill(~reliable_positive[:, None, None, :].bool(), -1)
    correct_id = masked_iou.argmax(-1)
    correct_iou = masked_iou.gather(-1, correct_id[..., None]).squeeze(-1).clamp_min(0)
    correct_quality = quality.gather(-1, correct_id[..., None]).squeeze(-1)
    wrong_quality = quality.masked_fill(
        torch.nn.functional.one_hot(correct_id, quality.shape[-1]).bool(), -1
    ).max(-1).values
    raw = 1 - correct_iou + torch.relu(float(margin) + wrong_quality - correct_quality)
    return (raw * eligible).sum() / eligible.sum().clamp_min(1)
