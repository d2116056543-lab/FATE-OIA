from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor


@dataclass(frozen=True)
class AIELossTerm:
    name: str
    owner: str
    weight: float
    value: Tensor


class AIELossRegistry:
    """A one-shot ledger preventing placeholder, duplicate, and double-weighted losses."""

    def __init__(self, configured_weights: dict[str, float]) -> None:
        self.configured_weights = {str(k): float(v) for k, v in configured_weights.items()}
        self._terms: dict[str, AIELossTerm] = {}

    def add(self, name: str, owner: str, value: Tensor) -> None:
        if name in self._terms:
            raise ValueError(f"AIE loss {name!r} was registered more than once")
        if name not in self.configured_weights:
            raise KeyError(f"AIE loss {name!r} has no configured weight")
        if value.ndim != 0 or not torch.isfinite(value):
            raise ValueError(f"AIE loss {name!r} must be a finite scalar")
        self._terms[name] = AIELossTerm(name, owner, self.configured_weights[name], value)

    def total(self) -> Tensor:
        missing = sorted(set(self.configured_weights) - set(self._terms))
        if missing:
            raise ValueError(f"Configured AIE losses were not computed: {missing}")
        return sum((term.weight * term.value for term in self._terms.values()), start=next(iter(self._terms.values())).value.new_zeros(()))

    def rows(self) -> list[dict[str, float | str]]:
        return [
            {"name": term.name, "owner": term.owner, "weight": term.weight, "raw": float(term.value.detach().cpu())}
            for term in self._terms.values()
        ]


def exact_owner_parameter_groups(model: torch.nn.Module) -> dict[str, list[torch.nn.Parameter]]:
    groups = {
        "primary": list(model.foundation.ego.parameters())
        + list(model.foundation.predicate_head.parameters())
        + list(model.foundation.trunk.parameters())
        + list(model.foundation.predicate_reason.parameters()),
        "action_evidence": list(model.action_evidence.parameters()) + list(model.predicate_naming.parameters()),
        "action_contribution": list(model.action_contribution.parameters()),
        "reason_private": list(model.reason_private.parameters()),
    }
    ids = [id(parameter) for values in groups.values() for parameter in values]
    if len(ids) != len(set(ids)):
        raise ValueError("AIE optimizer ownership overlaps")
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    actual = set(ids)
    if expected != actual:
        raise ValueError(f"AIE optimizer ownership is not exact: missing={len(expected-actual)} extra={len(actual-expected)}")
    return groups


