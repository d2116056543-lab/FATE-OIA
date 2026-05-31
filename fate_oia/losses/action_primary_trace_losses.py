from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from fate_oia.losses.asymmetric_loss import asymmetric_loss_with_logits
from fate_oia.losses.trace_losses import pairwise_rank_loss, prototype_diversity_loss
from fate_oia.utils.action_primary_conflict_gate import ActionPrimaryConflictGate

TAIL_LABELS = (12, 9, 5, 14, 6, 11, 10, 13)


def _w(args: Any, name: str, default: float) -> float:
    return float(getattr(args, name, default))


def _zero_like(t: torch.Tensor) -> torch.Tensor:
    return t.sum() * 0.0


def _agreement(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(torch.sigmoid(a), torch.sigmoid(b.detach())) + F.mse_loss(torch.sigmoid(b), torch.sigmoid(a.detach()))


def _tail_rank(logits: torch.Tensor, labels: torch.Tensor, margin: float, hard_k: int) -> torch.Tensor:
    terms = []
    for idx in TAIL_LABELS:
        if idx >= logits.shape[1]:
            continue
        pos_mask = labels[:, idx] > 0.5
        if not pos_mask.any():
            continue
        pos = logits[pos_mask, idx]
        neg_logits = logits[pos_mask]
        neg_mask = torch.ones_like(neg_logits, dtype=torch.bool)
        neg_mask[:, idx] = False
        neg = neg_logits.masked_select(neg_mask).view(pos.shape[0], -1)
        if neg.numel() == 0:
            continue
        top_neg = neg.topk(k=min(hard_k, neg.shape[1]), dim=1).values
        terms.append(F.relu(margin - (pos[:, None] - top_neg)).mean())
    return torch.stack(terms).mean() if terms else _zero_like(logits)


def _sentinel_params(out: dict[str, Any]) -> list[torch.nn.Parameter]:
    model = out.get("model_for_gate")
    if model is None:
        return []
    params = []
    max_params = int(getattr(out.get("args_for_gate", object()), "sentinel_max_params", 64))
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(key in name for key in ("label_head", "reason_to_action", "reason_alpha", "action_bias", "label_corr")):
            params.append(param)
        if len(params) >= max_params:
            break
    return params


def compute_action_primary_trace_loss(args: Any, out: dict[str, Any], labels: torch.Tensor, epoch: int, running_state: dict[str, Any] | None = None) -> tuple[torch.Tensor, dict[str, float]]:
    ad = int(getattr(args, "action_dim", 4))
    y_action = labels[:, :ad]
    y_reason = labels[:, ad:]
    gamma_pos = float(getattr(args, "asl_gamma_pos", 0.0))
    gamma_neg = float(getattr(args, "asl_gamma_neg", 4.0))
    clip = float(getattr(args, "asl_clip", 0.05))
    action_main_logits = out.get("action_logits_base_plus_bias", out["action_logits"])
    reason_logits = out["reason_logits"]
    evidence_logits = out.get("transport", {}).get("evidence_reason_logits", reason_logits)
    visual_logits = out.get("base_action_visual_logits", out.get("action_visual_logits", action_main_logits))
    reason_action_logits = out.get("action_logits_reason_to_action", out.get("base_action_reason_logits", out.get("reason_to_action_logits", action_main_logits)))
    action_main = asymmetric_loss_with_logits(action_main_logits, y_action, gamma_pos=gamma_pos, gamma_neg=gamma_neg, clip=clip)
    action_visual = asymmetric_loss_with_logits(visual_logits, y_action, gamma_pos=gamma_pos, gamma_neg=gamma_neg, clip=clip)
    action_reason = asymmetric_loss_with_logits(reason_action_logits, y_action, gamma_pos=gamma_pos, gamma_neg=gamma_neg, clip=clip)
    r2a = asymmetric_loss_with_logits(reason_action_logits, y_action, gamma_pos=gamma_pos, gamma_neg=gamma_neg, clip=clip)
    agree = _agreement(action_main_logits, reason_action_logits)
    bias_eff = out.get("action_bias_eff")
    bias_l2 = bias_eff.pow(2).mean() if torch.is_tensor(bias_eff) else _zero_like(action_main_logits)
    action_loss = (
        _w(args, "action_asl", 1.30) * action_main
        + _w(args, "action_visual_aux", 0.06) * action_visual
        + _w(args, "action_reason_aux", 0.12) * action_reason
        + _w(args, "reason_to_action_gt", 0.16) * r2a
        + _w(args, "action_agreement", 0.01) * agree
        + _w(args, "action_bias_l2", 0.002) * bias_l2
    )
    reason_asl = asymmetric_loss_with_logits(reason_logits, y_reason, gamma_pos=gamma_pos, gamma_neg=gamma_neg, clip=clip)
    evidence_asl = asymmetric_loss_with_logits(evidence_logits, y_reason, gamma_pos=0.0, gamma_neg=4.0, clip=0.05)
    evidence_rank = pairwise_rank_loss(evidence_logits, y_reason, margin=float(getattr(args, "evidence_rank_margin", 0.2)))
    distill = F.mse_loss(torch.sigmoid(evidence_logits), torch.sigmoid(out.get("base_reason_logits", reason_logits).detach()))
    reason_loss = _w(args, "reason_asl", 0.70) * reason_asl
    evidence_loss = _w(args, "evidence_reason_asl", 0.10) * evidence_asl + _w(args, "evidence_reason_rank", 0.020) * evidence_rank + _w(args, "evidence_base_distill", 0.010) * distill
    proto_div = prototype_diversity_loss(out["transport_module"].prototypes) if "transport_module" in out else _zero_like(reason_logits)
    entropy = out.get("transport", {}).get("transport_entropy", reason_logits.new_zeros(())).mean()
    reg_loss = _w(args, "prototype_diversity", 0.002) * proto_div + _w(args, "transport_entropy", 0.001) * entropy
    tail_proto = _zero_like(reason_logits)
    if int(epoch) >= int(getattr(args, "tail_proto_rank_start_epoch", 6)):
        tail_proto = _tail_rank(evidence_logits, y_reason, float(getattr(args, "tail_proto_rank_margin", 0.35)), int(getattr(args, "tail_proto_rank_hard_k", 5)))
        evidence_loss = evidence_loss + _w(args, "tail_proto_rank_weight_final", 0.015) * tail_proto
    tail_logit = _zero_like(reason_logits)
    if int(epoch) >= int(getattr(args, "tail_logit_rank_start_epoch", 5)):
        tail_logit = _tail_rank(reason_logits, y_reason, float(getattr(args, "tail_logit_rank_margin", 0.45)), int(getattr(args, "tail_logit_rank_hard_k", 5)))
        reason_loss = reason_loss + _w(args, "tail_logit_rank_weight_final", 0.020) * tail_logit
    cf_loss = _zero_like(reason_logits)
    cf_scale = 1.0
    cf = out.get("cf", {})
    if cf and int(epoch) >= int(getattr(args, "counterfactual_start_epoch", 6)):
        valid = cf.get("cf_valid_mask")
        if torch.is_tensor(valid) and valid.any():
            drop = cf.get("reason_logits_factual", reason_logits) - cf.get("reason_logits_target_deleted", reason_logits)
            cf_loss = F.relu(float(getattr(args, "cf_margin", 0.05)) - drop).masked_select(valid).mean()
    reason_scale = 1.0
    evidence_scale = 1.0
    gate_stats = {
        "grad_cos_action_reason": 0.0,
        "grad_cos_action_evidence": 0.0,
        "applied_reason_scale": 1.0,
        "applied_evidence_scale": 1.0,
        "applied_counterfactual_scale": 1.0,
        "action_floor_active": False,
    }
    if bool(getattr(args, "conflict_gate_enabled", False)) and bool(getattr(args, "_active_train", False)):
        gate = ActionPrimaryConflictGate.from_args(args)
        gate_stats = gate.compute(action_loss, reason_loss, evidence_loss, _sentinel_params(out), epoch, getattr(args, "latest_test_act_mF1", None))
        reason_scale = float(gate_stats["applied_reason_scale"])
        evidence_scale = float(gate_stats["applied_evidence_scale"])
        cf_scale = float(gate_stats["applied_counterfactual_scale"])
    total = action_loss + reason_scale * reason_loss + evidence_scale * evidence_loss + reg_loss + cf_scale * _w(args, "counterfactual_direct_max", 0.008) * cf_loss
    stats = {
        "loss_action_main": float(action_main.detach().cpu()),
        "loss_action_visual_aux": float(action_visual.detach().cpu()),
        "loss_action_reason_aux": float(action_reason.detach().cpu()),
        "loss_r2a_gt": float(r2a.detach().cpu()),
        "loss_action_agree": float(agree.detach().cpu()),
        "loss_action_bias_l2": float(bias_l2.detach().cpu()),
        "loss_reason_asl": float(reason_asl.detach().cpu()),
        "loss_evidence_reason_asl": float(evidence_asl.detach().cpu()),
        "loss_evidence_rank": float(evidence_rank.detach().cpu()),
        "loss_evidence_base_distill": float(distill.detach().cpu()),
        "loss_tail_proto_rank": float(tail_proto.detach().cpu()),
        "loss_tail_logit_rank": float(tail_logit.detach().cpu()),
        "loss_counterfactual": float(cf_loss.detach().cpu()),
        "action_loss_total": float(action_loss.detach().cpu()),
        "action_primary_total": float(action_loss.detach().cpu()),
        "reason_loss_total": float(reason_loss.detach().cpu()),
        "evidence_loss_total": float(evidence_loss.detach().cpu()),
        "regularization_loss_total": float(reg_loss.detach().cpu()),
        "total_loss": float(total.detach().cpu()),
    }
    stats.update({k: float(v) if isinstance(v, (int, float)) else bool(v) for k, v in gate_stats.items()})
    return total, stats
