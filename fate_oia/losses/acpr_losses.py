from __future__ import annotations

import torch
import torch.nn.functional as F

from fate_oia.losses.asymmetric_loss import asymmetric_loss_with_logits
from fate_oia.utils.acpr_pair_mining import action_vectors_to_subset_id


def action_asl_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return asymmetric_loss_with_logits(logits, target, gamma_neg=4, gamma_pos=0, clip=0.05)


def partial_label_reason_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pos = target.float()
    neg_weight = torch.full_like(pos, 0.35)
    weights = torch.where(pos > 0.5, torch.ones_like(pos), neg_weight)
    return F.binary_cross_entropy_with_logits(logits, pos, weight=weights)


def reason_soft_f1_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    tp = (probs * target).sum(0)
    fp = (probs * (1 - target)).sum(0)
    fn = ((1 - probs) * target).sum(0)
    f1 = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)
    return 1.0 - f1.mean()


def predicate_weak_bce_mil_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.sum() <= 0:
        return logits.sum() * 0.0
    loss = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
    return (loss * mask.float()).sum() / mask.sum().clamp_min(1.0)


def predicate_reason_alignment_loss(reason_logits: torch.Tensor, predicate_probs: torch.Tensor, reason_predicate_matrix: torch.Tensor) -> torch.Tensor:
    support = predicate_probs @ reason_predicate_matrix.t().to(predicate_probs.device, predicate_probs.dtype)
    support = support.clamp(0, 1)
    return F.binary_cross_entropy_with_logits(reason_logits, support.detach())


def matched_pair_logit_loss(reason_logits: torch.Tensor, pairs: dict, tail_multiplier: float = 2.0) -> torch.Tensor:
    pos = pairs.get("positive_pairs")
    contrast = pairs.get("contrast_pairs")
    loss = reason_logits.sum() * 0.0
    if pos is not None and pos.numel() > 0:
        loss = loss + (reason_logits[pos[:, 0]] - reason_logits[pos[:, 1]]).abs().mean()
    if contrast is not None and contrast.numel() > 0:
        dist = (reason_logits[contrast[:, 0]] - reason_logits[contrast[:, 1]]).abs().mean(-1)
        loss = loss + F.relu(0.25 - dist).mean()
    return loss


def matched_pair_embedding_loss(embeddings: torch.Tensor, pairs: dict) -> torch.Tensor:
    pos = pairs.get("positive_pairs")
    contrast = pairs.get("contrast_pairs")
    loss = embeddings.sum() * 0.0
    if pos is not None and pos.numel() > 0:
        loss = loss + (1 - (embeddings[pos[:, 0]] * embeddings[pos[:, 1]]).sum(-1)).mean()
    if contrast is not None and contrast.numel() > 0:
        sim = (embeddings[contrast[:, 0]] * embeddings[contrast[:, 1]]).sum(-1)
        loss = loss + F.relu(sim - 0.3).mean()
    return loss


def action_combo_ce_loss(action_set_logits: torch.Tensor, action_target: torch.Tensor) -> torch.Tensor:
    subset_id = action_vectors_to_subset_id(action_target)
    return F.cross_entropy(action_set_logits, subset_id)


def action_combo_drop_add_loss(action_set_logits: torch.Tensor, action_target: torch.Tensor) -> torch.Tensor:
    subset_id = action_vectors_to_subset_id(action_target)
    true_score = action_set_logits.gather(1, subset_id.view(-1, 1)).squeeze(1)
    return F.relu(action_set_logits.max(1).values - true_score + 0.05).mean()


def cardinality_loss(action_set_logits: torch.Tensor, action_target: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(action_set_logits, dim=-1)
    cards = torch.tensor([bin(i).count("1") for i in range(16)], device=action_set_logits.device, dtype=action_set_logits.dtype)
    pred = probs @ cards
    return F.smooth_l1_loss(pred, action_target.sum(-1))


def calibration_loss(calibrated_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(calibrated_logits, labels.float())


def calibration_regularizer_only_small(temperature: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return ((temperature - 1.0) ** 2).mean() + 0.05 * (bias ** 2).mean()


def predicate_attention_compactness_loss(attention: torch.Tensor) -> torch.Tensor:
    entropy = -(attention.clamp_min(1e-9).log() * attention).sum(-1)
    return entropy.mean()
