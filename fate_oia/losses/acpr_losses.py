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
    max_hinge: float | None = 4.0,
    use_active_mask: bool = True,
    return_stats: bool = False,
):
    if isinstance(pair_pos_idx, dict):
        pairs = pair_pos_idx
        pos = pairs.get("pair_pos_indices")
        neg = pairs.get("pair_neg_indices")
        rid = pairs.get("pair_reason_ids")
        weights = pairs.get("pair_weights")
        is_memory = pairs.get("pair_neg_is_memory")
        neg_logits_detached = pairs.get("pair_neg_logits_detached")
        active = pairs.get("pair_active_mask")
    else:
        pos, neg, rid, weights = pair_pos_idx, pair_neg_idx, pair_reason_idx, pair_weights
        is_memory = None
        neg_logits_detached = None
        active = None
    if pos is None or neg is None or rid is None or pos.numel() == 0:
        zero = reason_logits.sum() * 0.0
        if return_stats:
            return zero, {"hinge_mean": 0.0, "hinge_positive_rate": 0.0, "active_pair_rate": 0.0, "zero_loss_rate": 1.0}
        return zero
    pos = pos.long().to(reason_logits.device)
    neg = neg.long().to(reason_logits.device)
    rid = rid.long().to(reason_logits.device)
    is_memory_t = torch.zeros_like(pos, dtype=torch.bool) if is_memory is None else is_memory.to(reason_logits.device).bool()
    valid = (pos >= 0) & (pos < reason_logits.shape[0]) & (rid >= 0) & (rid < reason_logits.shape[1])
    valid = valid & (is_memory_t | ((neg >= 0) & (neg < reason_logits.shape[0])))
    if neg_logits_detached is None and is_memory_t.any():
        valid = valid & (~is_memory_t)
    if not valid.any():
        zero = reason_logits.sum() * 0.0
        if return_stats:
            return zero, {"hinge_mean": 0.0, "hinge_positive_rate": 0.0, "active_pair_rate": 0.0, "zero_loss_rate": 1.0}
        return zero
    pos = pos[valid]
    neg = neg[valid]
    rid = rid[valid]
    is_memory_t = is_memory_t[valid]
    if weights is not None:
        weights = weights.to(reason_logits.device)[valid]
    if active is not None:
        active = active.to(reason_logits.device).bool()[valid]
    else:
        active = torch.ones_like(pos, dtype=torch.bool)
    z_pos = reason_logits[pos, rid]
    z_neg = torch.empty_like(z_pos)
    in_batch = ~is_memory_t
    if in_batch.any():
        z_neg[in_batch] = reason_logits[neg[in_batch], rid[in_batch]]
    if is_memory_t.any():
        z_neg[is_memory_t] = neg_logits_detached.to(reason_logits.device, reason_logits.dtype)[valid][is_memory_t].detach()  # type: ignore[union-attr]
    w = torch.ones_like(z_pos) if weights is None else weights.to(z_pos.device, z_pos.dtype)
    raw_hinge_live = margin - z_pos + z_neg
    hinge = F.relu(raw_hinge_live)
    if max_hinge is not None and max_hinge > 0:
        hinge = hinge.clamp(max=float(max_hinge))
    if use_active_mask:
        hinge = hinge * active.float()
        w = w * active.float()
    loss = (w * hinge).sum() / w.sum().clamp_min(1.0)
    if return_stats:
        raw_hinge = raw_hinge_live.detach()
        stats = {
            "hinge_mean": float(raw_hinge.mean().detach().cpu()) if raw_hinge.numel() else 0.0,
            "hinge_positive_rate": float((raw_hinge > 0).float().mean().detach().cpu()) if raw_hinge.numel() else 0.0,
            "active_pair_rate": float(active.float().mean().detach().cpu()) if active.numel() else 0.0,
            "zero_loss_rate": float((hinge.detach() <= 0).float().mean().detach().cpu()) if hinge.numel() else 1.0,
        }
        return loss, stats
    return loss


def matched_pair_embedding_loss(
    reason_embeddings: torch.Tensor,
    pair_pos_idx: torch.Tensor | dict,
    pair_neg_idx: torch.Tensor | None = None,
    pair_reason_idx: torch.Tensor | None = None,
    pair_weights: torch.Tensor | None = None,
    embed_margin: float = 0.2,
    use_active_mask: bool = True,
    return_stats: bool = False,
):
    if isinstance(pair_pos_idx, dict):
        pairs = pair_pos_idx
        pos = pairs.get("pair_pos_indices")
        neg = pairs.get("pair_neg_indices")
        rid = pairs.get("pair_reason_ids")
        weights = pairs.get("pair_weights")
        is_memory = pairs.get("pair_neg_is_memory")
        neg_embedding_detached = pairs.get("pair_neg_embedding_detached")
        active = pairs.get("pair_active_mask")
    else:
        pos, neg, rid, weights = pair_pos_idx, pair_neg_idx, pair_reason_idx, pair_weights
        is_memory = None
        neg_embedding_detached = None
        active = None
    if pos is None or neg is None or rid is None or pos.numel() == 0:
        zero = reason_embeddings.sum() * 0.0
        if return_stats:
            return zero, {"embed_cosine_mean": 0.0, "active_pair_rate": 0.0}
        return zero
    pos = pos.long().to(reason_embeddings.device)
    neg = neg.long().to(reason_embeddings.device)
    rid = rid.long().to(reason_embeddings.device)
    if reason_embeddings.dim() == 2:
        e_all = reason_embeddings
        valid = (pos >= 0) & (pos < e_all.shape[0]) & (neg >= 0) & (neg < e_all.shape[0])
        rid_valid = torch.ones_like(valid, dtype=torch.bool)
        is_memory_t = torch.zeros_like(pos, dtype=torch.bool)
    else:
        e_all = reason_embeddings
        is_memory_t = torch.zeros_like(pos, dtype=torch.bool) if is_memory is None else is_memory.to(reason_embeddings.device).bool()
        valid = (pos >= 0) & (pos < e_all.shape[0]) & (rid >= 0) & (rid < e_all.shape[1])
        valid = valid & (is_memory_t | ((neg >= 0) & (neg < e_all.shape[0])))
        rid_valid = valid
        if neg_embedding_detached is None and is_memory_t.any():
            valid = valid & (~is_memory_t)
    if not valid.any():
        zero = reason_embeddings.sum() * 0.0
        if return_stats:
            return zero, {"embed_cosine_mean": 0.0, "active_pair_rate": 0.0}
        return zero
    pos = pos[valid]
    neg = neg[valid]
    rid = rid[valid]
    is_memory_t = is_memory_t[valid] if reason_embeddings.dim() == 3 else torch.zeros_like(pos, dtype=torch.bool)
    if weights is not None:
        weights = weights.to(reason_embeddings.device)[valid]
    if active is not None:
        active = active.to(reason_embeddings.device).bool()[valid]
    else:
        active = torch.ones_like(pos, dtype=torch.bool)
    if reason_embeddings.dim() == 3 and rid_valid.any():
        e_pos = reason_embeddings[pos, rid]
        e_neg = torch.empty_like(e_pos)
        in_batch = ~is_memory_t
        if in_batch.any():
            e_neg[in_batch] = reason_embeddings[neg[in_batch], rid[in_batch]]
        if is_memory_t.any():
            e_neg[is_memory_t] = neg_embedding_detached.to(reason_embeddings.device, reason_embeddings.dtype)[valid][is_memory_t].detach()  # type: ignore[union-attr]
    else:
        e_pos = reason_embeddings[pos]
        e_neg = reason_embeddings[neg]
    sim_neg = F.cosine_similarity(e_pos, e_neg, dim=-1)
    w = torch.ones_like(sim_neg) if weights is None else weights.to(sim_neg.device, sim_neg.dtype)
    loss_vec = F.relu(embed_margin + sim_neg)
    if use_active_mask:
        loss_vec = loss_vec * active.float()
        w = w * active.float()
    loss = (w * loss_vec).sum() / w.sum().clamp_min(1.0)
    if return_stats:
        return loss, {
            "embed_cosine_mean": float(sim_neg.mean().detach().cpu()) if sim_neg.numel() else 0.0,
            "active_pair_rate": float(active.float().mean().detach().cpu()) if active.numel() else 0.0,
        }
    return loss


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
    pred_membership = membership[pred.long()]
    gt = action_target.float()
    stats = {
        "drop_margin_mean": float(torch.cat(drop_losses).mean().detach().cpu()) if drop_losses else 0.0,
        "add_margin_mean": float(torch.cat(add_losses).mean().detach().cpu()) if add_losses else 0.0,
        "combo_gt_single_pred_rate": float(((action_target.sum(-1) > 1) & (torch.tensor([bin(int(x)).count("1") for x in pred.detach().cpu()], device=action_target.device) <= 1)).float().mean().detach().cpu()),
        "superset_pred_rate": float((((pred_membership >= gt).all(dim=1)) & (pred_membership.sum(dim=1) > gt.sum(dim=1))).float().mean().detach().cpu()),
        "all_high_rate": float((pred_membership.sum(dim=1) >= 4).float().mean().detach().cpu()),
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
