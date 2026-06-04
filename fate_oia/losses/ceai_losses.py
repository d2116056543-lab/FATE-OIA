from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from fate_oia.losses.asymmetric_loss import asymmetric_loss_with_logits
from fate_oia.losses.gradient_budget import gradient_budget_scale
from fate_oia.models.ceai_scene_state import masked_scene_state_bce


def _asl(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return asymmetric_loss_with_logits(logits, target, gamma_pos=0.0, gamma_neg=4.0, clip=0.05)


def ceai_main_loss(outputs: dict[str, torch.Tensor], labels: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    action_loss = _asl(outputs["final_action_logits"], labels["action"])
    reason_loss = _asl(outputs["final_reason_logits"], labels["reason"])
    return {"main_loss": action_loss + reason_loss, "action_main_loss": action_loss, "reason_main_loss": reason_loss}


def ceai_regularizer_losses(outputs: dict[str, Any], labels: dict[str, torch.Tensor], scene_state_targets: dict[str, torch.Tensor] | None = None, config: dict[str, float] | None = None) -> dict[str, torch.Tensor]:
    config = config or {}
    regs: dict[str, torch.Tensor] = {}
    regs["base_action_loss"] = _asl(outputs["base_action_logits"], labels["action"])
    regs["base_reason_loss"] = _asl(outputs["base_reason_logits"], labels["reason"])
    regs["action_specialist_loss"] = _asl(outputs["action_specialist_logits"], labels["action"])
    regs["reason_specialist_loss"] = _asl(outputs["reason_specialist_logits"], labels["reason"])
    regs["action_set_loss"] = _asl(outputs["action_set_logits"], labels["action"])
    if scene_state_targets is not None and "target" in scene_state_targets and "mask" in scene_state_targets:
        regs["scene_state_loss"] = masked_scene_state_bce(outputs["scene_state_logits"], scene_state_targets["target"], scene_state_targets["mask"])
    else:
        regs["scene_state_loss"] = outputs["final_action_logits"].new_zeros(())
    if "pair_support" in outputs and "pair_reliability" in outputs:
        pair_target = labels["action"].unsqueeze(2) * labels["reason"].unsqueeze(1)
        pair_weight = outputs["pair_reliability"].detach()
        regs["pair_seed_loss"] = (F.binary_cross_entropy_with_logits(outputs["pair_support"], pair_target, reduction="none") * pair_weight).mean()
        ent = outputs.get("pair_attention_entropy")
        regs["pair_attention_focus_loss"] = ent.mean() if torch.is_tensor(ent) else outputs["pair_support"].new_zeros(())
    else:
        regs["pair_seed_loss"] = outputs["final_action_logits"].new_zeros(())
        regs["pair_attention_focus_loss"] = outputs["final_action_logits"].new_zeros(())
    regs["pareto_safety_loss"] = F.relu(regs["base_action_loss"] - _asl(outputs["final_action_logits"], labels["action"]) + 0.005) + F.relu(regs["base_reason_loss"] - _asl(outputs["final_reason_logits"], labels["reason"]) + 0.005)
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
) -> tuple[torch.Tensor, dict[str, float]]:
    weights = {**DEFAULT_REG_WEIGHTS, **(weights or {})}
    aux = None
    stats: dict[str, float] = {}
    for name, loss in regularizers.items():
        w = min(float(weights.get(name, 0.0)), 0.30)
        stats[name] = float(loss.detach().cpu())
        if w <= 0:
            continue
        term = loss * w
        aux = term if aux is None else aux + term
    main = main_losses["main_loss"]
    if aux is None:
        stats["gradient_budget_rho"] = float(gradient_budget_rho)
        stats["aux_scale"] = 0.0
        return main, stats
    aux_budgeted, budget_stats = gradient_budget_scale(main, aux, rho=gradient_budget_rho)
    stats.update(budget_stats)
    stats["main_loss"] = float(main.detach().cpu())
    return main + aux_budgeted, stats
