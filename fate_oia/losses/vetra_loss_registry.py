from __future__ import annotations

from torch import Tensor


class VETRALossRegistry:
    def __init__(self) -> None:
        self.values: dict[str, Tensor] = {}

    def add(self, name: str, value: Tensor) -> None:
        if name in self.values:
            raise KeyError(f"duplicate VETRA loss: {name}")
        self.values[name] = value
