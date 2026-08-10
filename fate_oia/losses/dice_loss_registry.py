from __future__ import annotations

from torch import Tensor


class DICELossRegistry:
    weights = {"action_asl": .45, "rank_sketch": .25, "rank_protect": .15,
               "license": .08, "effect": .05, "delta": .02}

    def __init__(self) -> None:
        self.terms: dict[str, Tensor] = {}

    def add(self, name: str, value: Tensor) -> None:
        if name in self.terms:
            raise ValueError(f"duplicate DICE loss: {name}")
        self.terms[name] = value

    def total(self) -> Tensor:
        missing = set(self.weights) - set(self.terms)
        if missing:
            raise ValueError(f"missing DICE losses: {sorted(missing)}")
        return sum(self.weights[name] * self.terms[name] for name in self.weights)
