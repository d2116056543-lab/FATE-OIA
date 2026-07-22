from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn


OWNER_NAMES = ("action_foundation", "action_decoder", "reason_semantic", "evidence_core", "exchange_reread", "annotation_adapter", "threshold_head")


def parameter_ownership(model: nn.Module) -> dict[str, list[nn.Parameter]]:
    owned = model.owned_parameters()
    if set(owned) != set(OWNER_NAMES):
        raise ValueError("PRECISE parameter ownership is incomplete")
    return owned


def grad_norm(parameters: Iterable[nn.Parameter]) -> torch.Tensor:
    values = [parameter.grad.detach().norm() for parameter in parameters if parameter.grad is not None]
    if not values:
        return torch.tensor(0.0)
    return torch.stack(values).norm()


def project_target_credit_gradient(grounding_grad: torch.Tensor, target_credit_grad: torch.Tensor, max_ratio: float = 0.20) -> torch.Tensor:
    denominator = grounding_grad.square().sum().clamp_min(1e-12)
    projected = target_credit_grad
    dot = (projected * grounding_grad).sum()
    if dot < 0:
        projected = projected - dot / denominator * grounding_grad
    cap = max_ratio * grounding_grad.norm().clamp_min(1e-12)
    return projected * (cap / projected.norm().clamp_min(cap)).clamp(max=1.0)


def ownership_snapshot(model: nn.Module) -> dict[str, float]:
    return {owner: float(grad_norm(parameters).item()) for owner, parameters in parameter_ownership(model).items()}
