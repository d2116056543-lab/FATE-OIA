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


def predicate_reason_alignment_loss(reason_logits: torch.Tensor, predicate_probs: torch.Tensor, reason_predicate_matrix: torch.Tensor) -> torch.Tensor:
    support = predicate_probs @ reason_predicate_matrix.t().to(predicate_probs.device, predicate_probs.dtype)
    support = support.clamp(0, 1)
    return F.binary_cross_entropy_with_logits(reason_logits, support.detach())


def matched_pair_logit_loss(reason_logits: torch.Tensor, pairs: dict, margin: float = 0.25) -> torch.Tensor:
    pos = pairs.get("pair_pos_indices")
    neg = pairs.get("pair_neg_indices")
    rid = pairs.get("pair_reason_ids")
    weights = pairs.get("pair_weights")
    if pos is None or neg is None or rid is None or pos.numel() == 0:
        return reason_logits.sum() * 0.0
    z_pos = reason_logits[pos.long(), rid.long()]
    z_neg = reason_logits[neg.long(), rid.long()]
    w = torch.ones_like(z_pos) if weights is None else weights.to(z_pos.device, z_pos.dtype)
    return (w * F.relu(margin - z_pos + z_neg)).sum() / w.sum().clamp_min(1.0)


def matched_pair_embedding_loss(embeddings: torch.Tensor, pairs: dict) -> torch.Tensor:
    pos = pairs.get("pair_pos_indices")
    neg = pairs.get("pair_neg_indices")
    weights = pairs.get("pair_weights")
    if pos is None or neg is None or pos.numel() == 0:
        return embeddings.sum() * 0.0
    sim_pos = (embeddings[pos.long()] * embeddings[pos.long()]).sum(-1)
    sim_neg = (embeddings[pos.long()] * embeddings[neg.long()]).sum(-1)
    w = torch.ones_like(sim_pos) if weights is None else weights.to(sim_pos.device, sim_pos.dtype)
    return (w * F.relu(0.2 - sim_pos + sim_neg)).sum() / w.sum().clamp_min(1.0)


def action_combo_ce_loss(action_set_logits: torch.Tensor, action_target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(action_set_logits, action_vectors_to_subset_id(action_target))


def action_combo_drop_add_loss(action_set_logits: torch.Tensor, action_target: torch.Tensor, margin: float = 0.05, return_stats: bool = False):
    subset_id = action_vectors_to_subset_id(action_target)
    true_score = action_set_logits.gather(1, subset_id.view(-1, 1)).squeeze(1)
    drop_losses = []
    add_losses = []
    for a, bit in enumerate([1, 2, 4, 8]):
        pos = action_target[:, a] > 0.5
        neg = ~pos
        if pos.any():
            drop_id = (subset_id & (~bit)).clamp(0, 15)
            drop_losses.append(F.relu(margin - true_score[pos] + action_set_logits[pos].gather(1, drop_id[pos].view(-1, 1)).squeeze(1)))
        if neg.any():
            add_id = (subset_id | bit).clamp(0, 15)
            add_losses.append(F.relu(margin - true_score[neg] + action_set_logits[neg].gather(1, add_id[neg].view(-1, 1)).squeeze(1)))
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


def cardinality_loss(action_set_logits: torch.Tensor, action_target: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(action_set_logits, dim=-1)
    cards = torch.tensor([bin(i).count("1") for i in range(16)], device=action_set_logits.device, dtype=action_set_logits.dtype)
    return F.smooth_l1_loss(probs @ cards, action_target.sum(-1))


def calibration_loss(calibrated_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(calibrated_logits, labels.float())


def calibration_regularizer_only_small(temperature: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return ((temperature - 1.0) ** 2).mean() + 0.05 * (bias ** 2).mean()


def predicate_attention_compactness_loss(attention: torch.Tensor) -> torch.Tensor:
    entropy = -(attention.clamp_min(1e-9).log() * attention).sum(-1)
    return entropy.mean()
