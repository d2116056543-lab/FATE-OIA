from __future__ import annotations

import torch
import torch.nn.functional as F

from fate_oia.losses.asymmetric_loss import asymmetric_loss_with_logits
from fate_oia.utils.acpr_pair_mining import action_vectors_to_subset_id


def action_asl_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return asymmetric_loss_with_logits(logits, target, gamma_neg=4, gamma_pos=0, clip=0.05)


def partial_label_reason_loss(logits: torch.Tensor, target: torch.Tensor, contradiction_scores: torch.Tensor | None = None, neg_min_weight: float = 0.2) -> torch.Tensor:
    pos = target.float()
    if contradiction_scores is None:
        contradiction_scores = torch.zeros_like(pos)
    neg_weight = neg_min_weight + (1.0 - neg_min_weight) * contradiction_scores.detach().clamp(0, 1)
    weights = torch.where(pos > 0.5, torch.ones_like(pos), neg_weight)
    return F.binary_cross_entropy_with_logits(logits, pos, weight=weights)


def reason_soft_f1_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    tp = (probs * target).sum(0)
    fp = (probs * (1 - target)).sum(0)
    fn = ((1 - probs) * target).sum(0)
    f1 = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)
    return 1.0 - f1.mean()


def predicate_weak_bce_mil_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, reliability: torch.Tensor | None = None) -> torch.Tensor:
    if mask.sum() <= 0:
        return logits.sum() * 0.0
    reliability = torch.ones_like(mask) if reliability is None else reliability.clamp(0.1, 1.0)
    loss = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
    w = mask.float() * reliability
    return (loss * w).sum() / w.sum().clamp_min(1.0)


def predicate_reason_alignment_loss(
    predicate_probs: torch.Tensor,
    reason_targets: torch.Tensor,
    grammar_positive_matrix: torch.Tensor,
    grammar_contradiction_matrix: torch.Tensor,
) -> torch.Tensor:
    pos_support = predicate_probs @ grammar_positive_matrix.t().to(predicate_probs.device, predicate_probs.dtype)
    neg_support = predicate_probs @ grammar_contradiction_matrix.t().to(predicate_probs.device, predicate_probs.dtype)
    support_score = (pos_support - neg_support).clamp(-1, 1)
    target_sign = reason_targets.float() * 2.0 - 1.0
    return F.softplus(-target_sign * support_score).mean()


def matched_pair_logit_loss(
    reason_logits: torch.Tensor,
    pair_pos_idx: torch.Tensor | dict,
    pair_neg_idx: torch.Tensor | None = None,
    pair_reason_idx: torch.Tensor | None = None,
    pair_weights: torch.Tensor | None = None,
    margin: float = 0.25,
) -> torch.Tensor:
    if isinstance(pair_pos_idx, dict):
        pairs = pair_pos_idx
        pos = pairs.get("pair_pos_indices")
        neg = pairs.get("pair_neg_indices")
        rid = pairs.get("pair_reason_ids")
        weights = pairs.get("pair_weights")
    else:
        pos, neg, rid, weights = pair_pos_idx, pair_neg_idx, pair_reason_idx, pair_weights
    if pos is None or neg is None or rid is None or pos.numel() == 0:
        return reason_logits.sum() * 0.0
    valid = (pos.long() >= 0) & (pos.long() < reason_logits.shape[0]) & (neg.long() >= 0) & (neg.long() < reason_logits.shape[0]) & (rid.long() >= 0) & (rid.long() < reason_logits.shape[1])
    if not valid.any():
        return reason_logits.sum() * 0.0
    pos = pos[valid]
    neg = neg[valid]
    rid = rid[valid]
    if weights is not None:
        weights = weights[valid]
    z_pos = reason_logits[pos.long(), rid.long()]
    z_neg = reason_logits[neg.long(), rid.long()]
    w = torch.ones_like(z_pos) if weights is None else weights.to(z_pos.device, z_pos.dtype)
    return (w * F.relu(margin - z_pos + z_neg)).sum() / w.sum().clamp_min(1.0)


def matched_pair_embedding_loss(
    reason_embeddings: torch.Tensor,
    pair_pos_idx: torch.Tensor | dict,
    pair_neg_idx: torch.Tensor | None = None,
    pair_reason_idx: torch.Tensor | None = None,
    pair_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if isinstance(pair_pos_idx, dict):
        pairs = pair_pos_idx
        pos = pairs.get("pair_pos_indices")
        neg = pairs.get("pair_neg_indices")
        weights = pairs.get("pair_weights")
    else:
        pos, neg, weights = pair_pos_idx, pair_neg_idx, pair_weights
    if pos is None or neg is None or pos.numel() == 0:
        return reason_embeddings.sum() * 0.0
    valid = (pos.long() >= 0) & (pos.long() < reason_embeddings.shape[0]) & (neg.long() >= 0) & (neg.long() < reason_embeddings.shape[0])
    if not valid.any():
        return reason_embeddings.sum() * 0.0
    pos = pos[valid]
    neg = neg[valid]
    if weights is not None:
        weights = weights[valid]
    sim_pos = (reason_embeddings[pos.long()] * reason_embeddings[pos.long()]).sum(-1)
    sim_neg = (reason_embeddings[pos.long()] * reason_embeddings[neg.long()]).sum(-1)
    w = torch.ones_like(sim_pos) if weights is None else weights.to(sim_pos.device, sim_pos.dtype)
    return (w * F.relu(0.2 - sim_pos + sim_neg)).sum() / w.sum().clamp_min(1.0)


def action_combo_ce_loss(action_set_logits: torch.Tensor, action_target: torch.Tensor) -> torch.Tensor:
    subset_id = action_vectors_to_subset_id(action_target).clamp(0, action_set_logits.shape[1] - 1)
    target = F.one_hot(subset_id, num_classes=action_set_logits.shape[1]).to(action_set_logits.dtype)
    return -(target * F.log_softmax(action_set_logits, dim=-1)).sum(-1).mean()


def action_combo_drop_add_loss(action_set_logits: torch.Tensor, action_target: torch.Tensor, margin: float = 0.25, return_stats: bool = False):
    subset_id = action_vectors_to_subset_id(action_target)
    true_score = action_set_logits.gather(1, subset_id.view(-1, 1)).squeeze(1)
    drop_losses = []
    add_losses = []
    all_ids = torch.arange(16, device=action_set_logits.device, dtype=torch.long)
    membership = torch.stack([((all_ids >> bit) & 1) for bit in range(4)], dim=1).float()
    for a in range(4):
        pos = action_target[:, a] > 0.5
        neg = ~pos
        if pos.any():
            candidates = membership[:, a] < 0.5
            masked = action_set_logits[pos].masked_fill(~candidates.view(1, -1), -1e4)
            drop_losses.append(F.relu(margin - true_score[pos] + masked.max(dim=1).values))
        if neg.any():
            candidates = membership[:, a] > 0.5
            masked = action_set_logits[neg].masked_fill(~candidates.view(1, -1), -1e4)
            add_losses.append(F.relu(margin - true_score[neg] + masked.max(dim=1).values))
    loss = action_set_logits.sum() * 0.0
    if drop_losses:
        loss = loss + torch.cat(drop_losses).mean()
    if add_losses:
        loss = loss + torch.cat(add_losses).mean()
    if not return_stats:
        return loss
    pred = action_set_logits.argmax(-1)
    stats = {
        "drop_margin_mean": float(torch.cat(drop_losses).mean().detach().cpu()) if drop_losses else 0.0,
        "add_margin_mean": float(torch.cat(add_losses).mean().detach().cpu()) if add_losses else 0.0,
        "combo_gt_single_pred_rate": float(((action_target.sum(-1) > 1) & (torch.tensor([bin(int(x)).count("1") for x in pred.detach().cpu()], device=action_target.device) <= 1)).float().mean().detach().cpu()),
        "superset_pred_rate": float((((pred.view(-1, 1) & subset_id.view(-1, 1)) == subset_id.view(-1, 1)).float().mean()).detach().cpu()),
    }
    return loss, stats


def cardinality_loss(cardinality_logits: torch.Tensor, action_target: torch.Tensor) -> torch.Tensor:
    target_card = action_target.sum(-1).long().clamp(0, 4)
    return F.cross_entropy(cardinality_logits, target_card)


def calibration_loss(
    action_logits_cal: torch.Tensor,
    reason_logits_cal: torch.Tensor,
    action_targets: torch.Tensor,
    reason_targets: torch.Tensor,
) -> torch.Tensor:
    action_loss = F.binary_cross_entropy_with_logits(action_logits_cal, action_targets.float())
    reason_loss = F.binary_cross_entropy_with_logits(reason_logits_cal, reason_targets.float())
    return 0.5 * (action_loss + reason_loss)


def calibration_regularizer_only_small(temperature: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return ((temperature - 1.0) ** 2).mean() + 0.05 * (bias ** 2).mean()


def predicate_attention_compactness_loss(attention: torch.Tensor) -> torch.Tensor:
    entropy = -(attention.clamp_min(1e-9).log() * attention).sum(-1)
    return entropy.mean()
