from __future__ import annotations

import torch
import torch.nn.functional as F


def predicate_patch_alignment_loss(predicate_attention, predicate_patch_targets, predicate_patch_mask, predicate_patch_reliability, predicate_probs=None, entropy_weight=0.005):
    mask = predicate_patch_mask.float()
    reliability = predicate_patch_reliability.float()
    mass_inside = (predicate_attention * predicate_patch_targets.float()).sum(-1)
    if predicate_probs is None:
        prob_weight = torch.ones_like(mask)
    else:
        prob_weight = predicate_probs.detach().clamp(0.05, 1.0)
    weight = mask * reliability * prob_weight
    denom = weight.sum().clamp_min(1.0)
    mass_loss = ((1.0 - mass_inside) * weight).sum() / denom
    entropy = -(predicate_attention.clamp_min(1e-9).log() * predicate_attention).sum(-1)
    entropy_loss = (entropy * mask).sum() / mask.sum().clamp_min(1.0)
    loss = mass_loss + float(entropy_weight) * entropy_loss
    if mask.sum() <= 0:
        loss = predicate_attention.sum() * 0.0
    stats = {
        "mass_mean": float(((mass_inside * mask).sum() / mask.sum().clamp_min(1.0)).detach().cpu()) if mask.numel() else 0.0,
        "mass_pos_mean": float(((mass_inside * weight).sum() / denom).detach().cpu()),
        "random_region_baseline_estimate": float(predicate_patch_targets.float().mean().detach().cpu()),
        "entropy_mean": float(((entropy * mask).sum() / mask.sum().clamp_min(1.0)).detach().cpu()) if mask.numel() else 0.0,
        "valid_predicate_mask_rate": float(mask.mean().detach().cpu()) if mask.numel() else 0.0,
    }
    return loss, stats


def triadic_consistency_loss(triadic_reason_support, action_targets, reason_targets, contradiction_score=None, weight_positive=1.0, weight_negative=0.25):
    pos = (action_targets.unsqueeze(-1) > 0.5) & (reason_targets.unsqueeze(1) > 0.5)
    if pos.any():
        pos_loss = F.binary_cross_entropy(triadic_reason_support[pos].clamp(1e-5, 1-1e-5), torch.ones_like(triadic_reason_support[pos]))
    else:
        pos_loss = triadic_reason_support.sum() * 0.0
    if contradiction_score is not None:
        neg_weight = (1.0 - action_targets).unsqueeze(-1) * reason_targets.unsqueeze(1) * contradiction_score.unsqueeze(1).detach()
        neg_loss = (triadic_reason_support * neg_weight).sum() / neg_weight.sum().clamp_min(1.0)
    else:
        neg_weight = torch.zeros_like(triadic_reason_support)
        neg_loss = triadic_reason_support.sum() * 0.0
    loss = float(weight_positive) * pos_loss + float(weight_negative) * neg_loss
    return loss, {"positive_count": int(pos.sum().detach().cpu()), "negative_weight_mean": float(neg_weight.mean().detach().cpu())}


def predicate_conditioned_pu_reason_loss(reason_logits, reason_targets, predicate_contradiction, neg_min=0.4):
    targets = reason_targets.float()
    pos_weight = targets
    neg_weight = (1.0 - targets) * (float(neg_min) + (1.0 - float(neg_min)) * predicate_contradiction.detach().clamp(0, 1))
    loss_raw = F.binary_cross_entropy_with_logits(reason_logits, targets, reduction="none")
    weight = pos_weight + neg_weight
    loss = (loss_raw * weight).sum() / weight.sum().clamp_min(1.0)
    pos_mean = float(pos_weight[pos_weight > 0].mean().detach().cpu()) if (pos_weight > 0).any() else 1.0
    neg_vals = neg_weight[neg_weight > 0]
    return loss, {
        "positive_weight_mean": pos_mean,
        "negative_weight_mean": float(neg_vals.mean().detach().cpu()) if neg_vals.numel() else 0.0,
        "pu_neg_weight_mean": float(neg_vals.mean().detach().cpu()) if neg_vals.numel() else 0.0,
    }


def predicate_pair_weights(predicate_diff, threshold=0.05):
    strong = predicate_diff >= float(threshold)
    weights = torch.where(strong, torch.ones_like(predicate_diff), torch.full_like(predicate_diff, 0.25))
    return weights, {
        "predicate_filtered_pair_count": int(strong.sum().detach().cpu()),
        "weak_predicate_pair_count": int((~strong).sum().detach().cpu()),
    }


def capped_pair_loss(pair_loss_raw, main_prediction_loss, cap_ratio=0.10):
    cap = float(cap_ratio) * main_prediction_loss.detach().clamp_min(1e-8)
    capped = torch.minimum(pair_loss_raw, cap)
    return capped, {"pair_loss_raw": float(pair_loss_raw.detach().cpu()), "pair_loss_capped": float(capped.detach().cpu()), "pair_cap_active_rate": float((pair_loss_raw > cap).float().detach().cpu())}
