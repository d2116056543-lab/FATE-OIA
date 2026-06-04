from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from fate_oia.losses.asymmetric_loss import asymmetric_loss_with_logits
from fate_oia.losses.gradient_budget import compute_gradient_budget_scale
from fate_oia.models.ceai_scene_state import masked_scene_state_bce


DEFAULT_WEAK_PAIR_GROUPS = {
    0: [0, 2, 7, 15, 18, 19],
    1: [0, 1, 2, 3, 15, 16, 18],
    2: [5, 6, 9, 12, 14],
    3: [5, 6, 10, 11, 13],
}


def _asl(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return asymmetric_loss_with_logits(logits, target, gamma_pos=0.0, gamma_neg=4.0, clip=0.05)


def pareto_safety_penalty(final_loss: torch.Tensor, base_loss: torch.Tensor, margin: float = 0.005) -> torch.Tensor:
    return F.relu(final_loss - base_loss + float(margin))


def build_pair_seed_targets(
    action_labels: torch.Tensor,
    reason_labels: torch.Tensor,
    *,
    q_ar: torch.Tensor | None = None,
    weak_groups: dict[int, list[int]] | None = None,
    negative_weight: float = 0.05,
    positive_weight: float = 0.20,
    evidence_confidence: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weak_groups = weak_groups or DEFAULT_WEAK_PAIR_GROUPS
    b, action_dim = action_labels.shape
    reason_dim = reason_labels.shape[1]
    device = action_labels.device
    target = torch.zeros(b, action_dim, reason_dim, device=device, dtype=action_labels.dtype)
    mask = torch.zeros_like(target)
    weight = torch.zeros_like(target)
    action_pos = action_labels.unsqueeze(2) > 0.5
    reason_pos = reason_labels.unsqueeze(1) > 0.5
    negative = (~action_pos) | (~reason_pos)
    mask = torch.where(negative, torch.ones_like(mask), mask)
    weight = torch.where(negative, torch.full_like(weight, float(negative_weight)), weight)
    for a, reasons in weak_groups.items():
        if not (0 <= int(a) < action_dim):
            continue
        for r in reasons:
            if not (0 <= int(r) < reason_dim):
                continue
            pos = (action_labels[:, int(a)] > 0.5) & (reason_labels[:, int(r)] > 0.5)
            target[:, int(a), int(r)] = torch.where(pos, torch.ones_like(target[:, int(a), int(r)]), target[:, int(a), int(r)])
            mask[:, int(a), int(r)] = torch.where(pos, torch.ones_like(mask[:, int(a), int(r)]), mask[:, int(a), int(r)])
            weight[:, int(a), int(r)] = torch.where(pos, torch.full_like(weight[:, int(a), int(r)], float(positive_weight)), weight[:, int(a), int(r)])
    if q_ar is not None:
        weight = weight * q_ar.detach().clamp(0.0, 1.0)
    if evidence_confidence is not None:
        weight = weight * evidence_confidence.detach().clamp(0.0, 1.0)
    weight = weight * mask
    return target, mask, weight


def ceai_main_loss(outputs: dict[str, torch.Tensor], labels: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    action_loss = _asl(outputs["final_action_logits"], labels["action"])
    reason_loss = _asl(outputs["final_reason_logits"], labels["reason"])
    return {"main_loss": action_loss + reason_loss, "action_main_loss": action_loss, "reason_main_loss": reason_loss}


def ceai_regularizer_losses(
    outputs: dict[str, Any],
    labels: dict[str, torch.Tensor],
    scene_state_targets: dict[str, torch.Tensor] | None = None,
    config: dict[str, float] | None = None,
) -> dict[str, torch.Tensor]:
    config = config or {}
    regs: dict[str, torch.Tensor] = {}
    base_action_loss = _asl(outputs["base_action_logits"], labels["action"])
    base_reason_loss = _asl(outputs["base_reason_logits"], labels["reason"])
    final_action_loss = _asl(outputs["final_action_logits"], labels["action"])
    final_reason_loss = _asl(outputs["final_reason_logits"], labels["reason"])
    regs["base_action_loss"] = base_action_loss
    regs["base_reason_loss"] = base_reason_loss
    regs["action_specialist_loss"] = _asl(outputs["action_specialist_logits"], labels["action"])
    regs["reason_specialist_loss"] = _asl(outputs["reason_specialist_logits"], labels["reason"])
    regs["action_set_loss"] = _asl(outputs["action_set_logits"], labels["action"])
    if scene_state_targets is not None and "target" in scene_state_targets and "mask" in scene_state_targets:
        regs["scene_state_loss"] = masked_scene_state_bce(outputs["scene_state_logits"], scene_state_targets["target"], scene_state_targets["mask"])
    else:
        regs["scene_state_loss"] = outputs["final_action_logits"].new_zeros(())
    if "pair_support" in outputs and "pair_reliability" in outputs:
        target, mask, weight = build_pair_seed_targets(labels["action"], labels["reason"], q_ar=outputs["pair_reliability"])
        pair_loss = F.binary_cross_entropy_with_logits(outputs["pair_support"], target, reduction="none")
        regs["pair_seed_loss"] = (pair_loss * mask * weight).sum() / weight.sum().clamp_min(1.0)
        ent = outputs.get("pair_attention_entropy")
        regs["pair_attention_focus_loss"] = ent.mean() if torch.is_tensor(ent) else outputs["pair_support"].new_zeros(())
    else:
        regs["pair_seed_loss"] = outputs["final_action_logits"].new_zeros(())
        regs["pair_attention_focus_loss"] = outputs["final_action_logits"].new_zeros(())
    regs["pareto_safety_loss"] = pareto_safety_penalty(final_action_loss, base_action_loss, margin=0.005) + pareto_safety_penalty(final_reason_loss, base_reason_loss, margin=0.005)
    return regs


DEFAULT_REG_WEIGHTS = {
    "base_action_loss": 0.05,
    "base_reason_loss": 0.05,
    "action_specialist_loss": 0.15,
    "reason_specialist_loss": 0.15,
    "action_set_loss": 0.10,
    "scene_state_loss": 0.08,
    "pair_seed_loss": 0.05,
    "pair_attention_focus_loss": 0.005,
    "pareto_safety_loss": 0.10,
}


def compute_total_loss_with_gradient_budget(
    main_losses: dict[str, torch.Tensor],
    regularizers: dict[str, torch.Tensor],
    weights: dict[str, float] | None = None,
    gradient_budget_rho: float = 0.15,
    shared_params: list[torch.nn.Parameter] | None = None,
) -> tuple[torch.Tensor, dict[str, float | bool]]:
    weights = {**DEFAULT_REG_WEIGHTS, **(weights or {})}
    aux = None
    stats: dict[str, float | bool] = {}
    for name, loss in regularizers.items():
        w = min(float(weights.get(name, 0.0)), 0.30)
        stats[name] = float(loss.detach().cpu())
        if w <= 0:
            continue
        term = loss * w
        aux = term if aux is None else aux + term
    main = main_losses["main_loss"]
    if aux is None:
        stats.update({"gradient_budget_rho": float(gradient_budget_rho), "aux_scale": 0.0, "budget_scale": 0.0, "used_true_grad_norm": True})
        return main, stats
    if shared_params is None:
        stats.update({"gradient_budget_rho": float(gradient_budget_rho), "aux_scale": 0.0, "budget_scale": 0.0, "norm_main": 0.0, "norm_aux": 0.0, "rho": float(gradient_budget_rho), "used_true_grad_norm": False})
        stats["main_loss"] = float(main.detach().cpu())
        return main + aux * 0.0, stats
    scale, budget_stats = compute_gradient_budget_scale(main, aux, shared_params, rho=gradient_budget_rho)
    stats.update(budget_stats)
    stats["main_loss"] = float(main.detach().cpu())
    return main + aux * scale, stats
