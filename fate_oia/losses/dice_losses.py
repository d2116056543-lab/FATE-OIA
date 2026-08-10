from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .asymmetric_loss import asymmetric_loss_with_logits


def action_asl_loss(logits: Tensor, target: Tensor) -> Tensor:
    return asymmetric_loss_with_logits(logits, target)


def route_directional_license_targets(support_target: Tensor, counter_target: Tensor,
                                      action_target: Tensor) -> tuple[Tensor,Tensor]:
    target=action_target.to(support_target)
    raw_support=target*support_target+(1-target)*counter_target
    raw_counter=(1-target)*support_target+target*counter_target
    return raw_support.detach(),raw_counter.detach()


def directional_license_loss(support_logits: Tensor, counter_logits: Tensor,
                             support_target: Tensor, counter_target: Tensor,
                             action_target: Tensor) -> Tensor:
    raw_support,raw_counter=route_directional_license_targets(support_target,counter_target,action_target)
    return F.binary_cross_entropy_with_logits(support_logits.float(),raw_support.float()) + F.binary_cross_entropy_with_logits(counter_logits.float(),raw_counter.float())


def directional_effect_loss(atom_correction: Tensor, action_target: Tensor, effect: Tensor) -> Tensor:
    if atom_correction.shape != action_target.shape or effect.shape != atom_correction.shape:
        raise ValueError("directional effect tensors must be eventwise and shape-identical")
    signed = (2 * action_target - 1) * atom_correction
    # A negative target-margin effect is retained as a counter-evidence audit,
    # but must not be duplicated as a new correction that further harms ranking.
    return F.smooth_l1_loss(signed, effect.detach().clamp_min(0))


def delta_regularizer(delta: Tensor) -> Tensor:
    return delta.square().mean()


def certificate_targets(selected_drop: Tensor, controls: Tensor, temperature: float = 0.05) -> dict[str, Tensor]:
    median = controls.median(-1).values
    mad = (controls - median[..., None]).abs().median(-1).values
    support = torch.sigmoid((selected_drop - median - mad) / float(temperature))
    counter = torch.sigmoid((median - mad - selected_drop) / float(temperature))
    return {"control_median": median, "control_mad": mad, "license_support_cf": support,
            "license_counter_cf": counter, "directional_effect": selected_drop - median}
