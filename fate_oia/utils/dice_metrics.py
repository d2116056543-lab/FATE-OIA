from __future__ import annotations

import torch
from torch import Tensor

from fate_oia.utils.aie_metrics import aie_branch_metrics, spearman_correlation


def dice_branch_metrics(base_action: Tensor, dice_action: Tensor, reason: Tensor,
                        action_target: Tensor, reason_target: Tensor) -> dict:
    return {"base": aie_branch_metrics(base_action, reason, action_target, reason_target),
            "dice": aie_branch_metrics(dice_action, reason, action_target, reason_target)}


def mechanism_metrics(output: dict, effect: Tensor | None = None) -> dict:
    atom = output["atom_correction"].detach().float()
    result = {
        "reason_identity_max_abs": float(output["reason_identity_max_abs"]),
        "predicate_top2_count_max": int(output["predicate_top2_count"].max()),
        "predicate_fallback_rate": float(output["predicate_fallback"].float().mean()),
        "predicate_agreement_mean": float(output["predicate_agreement"].float().mean()),
        "predicate_strength_mean": float(output["predicate_strength"].float().mean()),
        "visual_map_entropy": float((-(output["visual_map"].float().clamp_min(1e-8)*output["visual_map"].float().clamp_min(1e-8).log()).sum(-1)).mean()),
        "coherent_map_entropy_mean": float((-(output["coherent_map"].float().clamp_min(1e-8)*output["coherent_map"].float().clamp_min(1e-8).log()).sum(-1)).mean()),
        "support_magnitude_mean": float(output["support_magnitude"].float().mean()),
        "counter_magnitude_mean": float(output["counter_magnitude"].float().mean()),
        "license_support_hat_mean": float(output["license_support_hat"].float().mean()),
        "license_counter_hat_mean": float(output["license_counter_hat"].float().mean()),
        "dice_delta_abs_mean": float(output["dice_action_delta"].float().abs().mean()),
        "dice_delta_abs_max": float(output["dice_action_delta"].float().abs().max()),
        "positive_correction_rate": float((atom > 0).float().mean()),
        "negative_correction_rate": float((atom < 0).float().mean()),
    }
    if effect is not None and atom.numel()==effect.numel():
        result["contribution_effect_spearman"] = spearman_correlation(atom.flatten(), effect.detach().float().flatten())
    return result
