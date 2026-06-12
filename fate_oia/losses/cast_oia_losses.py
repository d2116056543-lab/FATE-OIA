from __future__ import annotations

import torch
import torch.nn.functional as F

from fate_oia.models.cast_action_set_energy import PAIR_INDEX, action_targets_to_subset_ids, build_subset_membership


def action_multi_label_asl_loss(action_logits, action_targets):
    return F.binary_cross_entropy_with_logits(action_logits, action_targets.float())


def action_set_ce_loss(action_set_logits, action_targets):
    return F.cross_entropy(action_set_logits, action_targets_to_subset_ids(action_targets))


def cardinality_loss(action_set_probs, action_targets):
    card = action_targets.float().sum(-1).long().clamp(0, 4)
    subset = build_subset_membership(action_targets.shape[1]).to(action_set_probs.device)
    subset_card = subset.sum(-1).long()
    card_probs = torch.stack([action_set_probs[:, subset_card == c].sum(-1) for c in range(5)], dim=-1)
    return F.nll_loss(card_probs.clamp_min(1e-8).log(), card)


def drop_add_subset_margin_loss(action_set_logits, action_targets, margin: float = 0.25):
    gt_id = action_targets_to_subset_ids(action_targets)
    gt_score = action_set_logits.gather(1, gt_id.view(-1, 1)).squeeze(1)
    subset = build_subset_membership(action_targets.shape[1]).to(action_set_logits.device)
    gt = action_targets.float()
    losses = []
    for i in range(action_targets.shape[0]):
        mask_drop = ((subset <= gt[i].view(1, -1)).all(-1)) & (subset.sum(-1) < gt[i].sum())
        mask_add = ((subset >= gt[i].view(1, -1)).all(-1)) & (subset.sum(-1) > gt[i].sum())
        bad = mask_drop | mask_add
        if bool(bad.any()):
            losses.append(F.relu(margin + action_set_logits[i, bad] - gt_score[i]).mean())
    return torch.stack(losses).mean() if losses else action_set_logits.sum() * 0


def pair_compatibility_loss(pair_logits, action_targets):
    targets = []
    for a, b in PAIR_INDEX:
        targets.append(action_targets[:, a] * action_targets[:, b])
    return F.binary_cross_entropy_with_logits(pair_logits, torch.stack(targets, dim=-1).float())


def reason_reliability_asl_loss(reason_logits, reason_targets, reliability):
    y = reason_targets.float()
    bce = F.binary_cross_entropy_with_logits(reason_logits, y, reduction="none")
    weights = torch.where(y > 0.5, torch.ones_like(y), 0.4 + 0.6 * reliability.detach())
    return (bce * weights).mean()


def tail_same_action_set_ranking_loss(reason_logits, reason_targets, action_targets, margin: float = 0.1):
    pos = reason_targets > 0.5
    neg = ~pos
    losses = []
    for i in range(reason_logits.shape[0]):
        if bool(pos[i].any()) and bool(neg[i].any()):
            p = reason_logits[i, pos[i]].mean()
            n = reason_logits[i, neg[i]].topk(min(3, int(neg[i].sum().item()))).values.mean()
            losses.append(F.relu(margin - p + n))
    return torch.stack(losses).mean() if losses else reason_logits.sum() * 0


def reason_to_action_set_alignment_loss(reason_to_set_logits, reason_targets, action_targets):
    gt = action_targets_to_subset_ids(action_targets)
    score = reason_to_set_logits.gather(2, gt.view(-1, 1, 1).expand(-1, reason_to_set_logits.shape[1], 1)).squeeze(-1)
    return F.binary_cross_entropy_with_logits(score, reason_targets.float())


def text_evidence_contrast_loss(label_attention, text_similarity):
    attn_sim = torch.einsum("bln,bmn->blm", label_attention, label_attention).mean(0)
    target = text_similarity.to(attn_sim.device, attn_sim.dtype)
    return F.mse_loss(attn_sim, target)


def graph_sparsity_loss(edge_weights):
    entropy = -(edge_weights.clamp_min(1e-9) * edge_weights.clamp_min(1e-9).log()).sum(-1)
    return entropy.mean()


def calibration_regularizer(logits, targets):
    return torch.mean((torch.sigmoid(logits) - targets.float()) ** 2)


def evidence_compactness_loss(label_attention):
    entropy = -(label_attention.clamp_min(1e-9) * label_attention.clamp_min(1e-9).log()).sum(-1)
    return entropy.mean()
