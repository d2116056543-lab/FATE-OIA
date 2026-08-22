from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class TIDALossRow:
    value: torch.Tensor
    weight: float
    available: bool
    unavailable_reason: str | None


class TIDALossRegistry:
    required_terms = (
        "terminal_hist", "terminal_no_history", "terminal_gain", "temporal_order", "repeated_last_contrast",
        "flow_transition_align",
        "action_asl", "action_smooth_ap", "action_base_protect", "action_delta", "action_route_sparse",
        "action_flow_credit", "action_flow_no_harm",
        "action_utility_calibration",
        "reason_partial", "reason_rank", "reason_soft_f1", "reason_delta",
        "reason_flow_credit", "reason_flow_no_harm",
        "reason_utility_calibration",
    )
    default_weights = {
        "terminal_hist": 0.25,
        # Kept as a diagnostic row, never optimized. Optimizing this branch
        # teaches a target reconstruction shortcut that competes with history.
        "terminal_no_history": 0.0,
        "terminal_gain": 0.25 * 0.20,
        "temporal_order": 0.25 * 0.10,
        "repeated_last_contrast": 0.25 * 0.10,
        "flow_transition_align": 0.05,
        "action_asl": 1.00,
        "action_smooth_ap": 0.15,
        "action_base_protect": 0.10,
        "action_delta": 0.005,
        "action_route_sparse": 0.005,
        # Paired flow credit must be large enough to compete with the direct
        # task loss, while no-harm keeps the frozen image fallback dominant
        # whenever ordered history is not useful.
        "action_flow_credit": 0.50,
        "action_flow_no_harm": 0.15,
        "action_utility_calibration": 0.05,
        "reason_partial": 1.00,
        "reason_rank": 0.08,
        "reason_soft_f1": 0.04,
        "reason_delta": 0.005,
        "reason_flow_credit": 0.30,
        "reason_flow_no_harm": 0.12,
        "reason_utility_calibration": 0.04,
    }

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = dict(self.default_weights)
        if weights:
            unknown = set(weights) - set(self.required_terms)
            if unknown:
                raise ValueError(f"unknown TIDA loss weights: {sorted(unknown)}")
            self.weights.update({name: float(value) for name, value in weights.items()})
        self.rows: dict[str, TIDALossRow] = {}

    def add(self, name: str, value: torch.Tensor, *, available: bool = True, unavailable_reason: str | None = None) -> None:
        if name not in self.required_terms:
            raise ValueError(f"unregistered loss term: {name}")
        if name in self.rows:
            raise ValueError(f"loss term added twice: {name}")
        if not torch.isfinite(value).all():
            raise ValueError(f"non-finite loss: {name}")
        if not available and not unavailable_reason:
            raise ValueError("unavailable loss requires a reason")
        self.rows[name] = TIDALossRow(value, self.weights[name], bool(available), unavailable_reason)

    def total(self) -> torch.Tensor:
        missing = set(self.required_terms) - set(self.rows)
        if missing:
            raise ValueError(f"missing TIDA loss terms: {sorted(missing)}")
        return sum(row.weight * row.value for row in self.rows.values())

    def artifact(self) -> dict[str, dict[str, float | bool | str | None]]:
        return {
            name: {
                "raw": float(row.value.detach().cpu()),
                "weight": row.weight,
                "weighted": float((row.weight * row.value).detach().cpu()),
                "available": row.available,
                "unavailable_reason": row.unavailable_reason,
            }
            for name, row in self.rows.items()
        }


def assert_owner_exact_cover(model: nn.Module, owners: dict[str, list[nn.Parameter]]) -> None:
    trainable = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    owner_ids = [id(parameter) for parameters in owners.values() for parameter in parameters if parameter.requires_grad]
    duplicates = {identifier for identifier in owner_ids if owner_ids.count(identifier) > 1}
    missing = trainable - set(owner_ids)
    extra = set(owner_ids) - trainable
    if duplicates or missing or extra:
        raise ValueError(
            f"owner exact-cover failure: duplicate={len(duplicates)}, missing={len(missing)}, extra={len(extra)}"
        )
