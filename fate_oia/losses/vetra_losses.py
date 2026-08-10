from __future__ import annotations

import torch
from torch import Tensor

from .asymmetric_loss import asymmetric_loss_with_logits


def vetra_action_asl_loss(logits: Tensor, target: Tensor) -> Tensor:
    return asymmetric_loss_with_logits(logits, target)


def fixed_base_rank_protection_loss(final: Tensor, base: Tensor, target: Tensor,
                                    rho: float = .95, margin_floor: float = .10) -> Tensor:
    terms = []
    for action in range(target.shape[1]):
        positive, negative = target[:, action] > .5, target[:, action] <= .5
        if not bool(positive.any() and negative.any()):
            continue
        base_margin = base[positive, action, None] - base[negative, action][None]
        final_margin = final[positive, action, None] - final[negative, action][None]
        reliable = base_margin >= float(margin_floor)
        if reliable.any():
            terms.append(torch.relu(float(rho) * base_margin[reliable] - final_margin[reliable]).mean())
    return torch.stack(terms).mean() if terms else final.sum() * 0


def correction_bias_energy_loss(delta: Tensor) -> tuple[Tensor, Tensor]:
    mean, variance = delta.mean(0), delta.var(0, unbiased=False)
    bias = (mean.square() / (variance + 1e-6)).mean()
    return bias, delta.square().mean()


def route_health_loss(support_route: Tensor, counter_route: Tensor, reliability: Tensor,
                      null_max: float = .70, confidence_floor: float = .35) -> Tensor:
    reliable = reliability[..., :-1].amax(-1) >= float(confidence_floor)
    null_rate = .5 * (support_route[..., -1] + counter_route[..., -1])
    if not reliable.any():
        return null_rate.sum() * 0
    return torch.relu(null_rate[reliable] - float(null_max)).mean()


def total_vetra_loss(output: dict, action_target: Tensor, map_loss,
                     weights: dict[str, float], rho: float, margin_floor: float,
                     null_max: float, confidence_floor: float) -> tuple[Tensor, dict[str, Tensor]]:
    asl = vetra_action_asl_loss(output["action_logits_final"], action_target)
    direct_map = map_loss(output["action_logits_final"], action_target)
    protect = fixed_base_rank_protection_loss(output["action_logits_final"], output["action_logits_base"],
                                              action_target, rho, margin_floor)
    bias, energy = correction_bias_energy_loss(output["vetra_action_delta"])
    health = route_health_loss(output["support_route"], output["counter_route"],
                               output["support_reliability"], null_max, confidence_floor)
    components = {"action_asl": asl, "action_map": direct_map, "base_rank_protect": protect,
                  "correction_bias": bias, "correction_energy": energy, "route_health": health}
    total = (weights["action_asl"] * asl + weights["action_map"] * direct_map
             + weights["base_rank_protect"] * protect
             + weights["correction_bias_energy"] * .5 * (bias + energy)
             + weights["route_health"] * health)
    return total, components
