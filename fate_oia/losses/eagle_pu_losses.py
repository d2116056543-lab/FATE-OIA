from __future__ import annotations

import torch
import torch.nn.functional as F

from fate_oia.losses.asymmetric_loss import asymmetric_loss_with_logits
from fate_oia.models.eagle_pu_action_set_aux import action_subset_targets


def action_direct_asl_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return asymmetric_loss_with_logits(logits, targets, gamma_pos=0.0, gamma_neg=4.0, clip=0.05)


def reason_direct_asl_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return asymmetric_loss_with_logits(logits, targets, gamma_pos=0.0, gamma_neg=4.0, clip=0.05)


def reason_soft_f1_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    tp = (probs * targets).sum(0)
    fp = (probs * (1 - targets)).sum(0)
    fn = ((1 - probs) * targets).sum(0)
    f1 = 2 * tp / (2 * tp + fp + fn + eps)
    return 1 - f1.mean()


def positive_unlabeled_reason_loss(logits: torch.Tensor, targets: torch.Tensor, reliability: torch.Tensor, contradiction: torch.Tensor | None = None) -> torch.Tensor:
    targets = targets.float()
    reliability = reliability.detach().clamp(0, 1)
    neg_w = 0.4 + 0.6 * reliability
    if contradiction is not None:
        neg_w = torch.maximum(neg_w, contradiction.detach().clamp(0, 1))
    pos_loss = F.binary_cross_entropy_with_logits(logits, torch.ones_like(targets), reduction="none")
    neg_loss = F.binary_cross_entropy_with_logits(logits, torch.zeros_like(targets), reduction="none")
    loss = targets * pos_loss + (1 - targets) * neg_w * neg_loss
    return loss.mean()


def state_weak_bag_loss(state_logits: torch.Tensor, targets: torch.Tensor | None = None) -> torch.Tensor:
    if targets is None:
        return state_logits.sigmoid().mean() * 0.0
    return F.binary_cross_entropy_with_logits(state_logits, targets.float())


def text_state_contrast_loss(state_tokens: torch.Tensor, label_text_prototypes: torch.Tensor) -> torch.Tensor:
    state = F.normalize(state_tokens.mean(1), dim=-1)
    text = F.normalize(label_text_prototypes.mean(0).unsqueeze(0).expand_as(state), dim=-1)
    return 1 - (state * text).sum(-1).mean()


def prototype_transport_loss(delta: torch.Tensor) -> torch.Tensor:
    return delta.pow(2).mean()


def state_label_graph_regularizer(edge_weights: torch.Tensor) -> torch.Tensor:
    entropy = -(edge_weights.clamp_min(1e-8).log() * edge_weights).sum(-1).mean()
    return entropy


def action_set_ce_loss(action_set_logits: torch.Tensor, subset_targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(action_set_logits, subset_targets.long())


def action_set_drop_add_loss(action_set_logits: torch.Tensor, action_targets: torch.Tensor, margin: float = 0.1) -> torch.Tensor:
    subset = action_subset_targets(action_targets)
    true_energy = action_set_logits.gather(1, subset.view(-1, 1)).squeeze(1)
    max_other = action_set_logits.masked_fill(F.one_hot(subset, 16).bool(), -1e4).max(1).values
    return F.relu(max_other - true_energy + margin).mean()


def cardinality_loss(cardinality_logits: torch.Tensor, action_targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(cardinality_logits, action_targets.sum(1).long().clamp(0, cardinality_logits.shape[1] - 1))


def tail_same_action_rank_loss(reason_logits: torch.Tensor, reason_targets: torch.Tensor, tail_indices: list[int] | tuple[int, ...] = (5, 6, 9, 12, 14)) -> torch.Tensor:
    if not tail_indices:
        return reason_logits.sum() * 0
    idx = torch.tensor(list(tail_indices), device=reason_logits.device, dtype=torch.long)
    tail_logits = reason_logits.index_select(1, idx)
    tail_targets = reason_targets.index_select(1, idx)
    pos = tail_logits[tail_targets > 0.5]
    neg = tail_logits[tail_targets <= 0.5]
    if pos.numel() == 0 or neg.numel() == 0:
        return reason_logits.sum() * 0
    return F.relu(0.1 - pos.mean() + neg.mean())


def calibration_regularizer(temperature: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return (temperature.log().pow(2).mean() + bias.pow(2).mean())


def evidence_margin_loss(selected_score: torch.Tensor, random_score: torch.Tensor, active: bool = False, margin: float = 0.02) -> torch.Tensor:
    if not active:
        return (selected_score.sum() + random_score.sum()) * 0
    return F.relu(random_score - selected_score + margin).mean()


LOSS_WEIGHTS = {
    "action_direct": 1.00,
    "reason_direct": 1.00,
    "reason_soft_f1": 0.06,
    "pu_reason": 0.15,
    "state_weak_bag": 0.05,
    "text_state_contrast": 0.03,
    "prototype_transport": 0.05,
    "state_label_graph": 0.02,
    "action_set_ce": 0.05,
    "action_set_drop_add": 0.03,
    "cardinality": 0.02,
    "tail_same_action_rank": 0.04,
    "calibration": 0.02,
    "evidence_margin_max": 0.002,
}
