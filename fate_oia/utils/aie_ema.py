from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager

import torch
from torch import nn


class ModelEMA:
    """Device-local exponential average of floating model state."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be in [0, 1)")
        self.decay = float(decay)
        self._state = OrderedDict(
            (name, value.detach().clone())
            for name, value in list(model.named_parameters()) + list(model.named_buffers())
        )

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        current = OrderedDict(list(model.named_parameters()) + list(model.named_buffers()))
        if current.keys() != self._state.keys():
            raise RuntimeError("EMA/model state keys differ")
        for name, value in current.items():
            if torch.is_floating_point(value):
                self._state[name].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
            else:
                self._state[name].copy_(value)

    def state_dict(self) -> OrderedDict[str, torch.Tensor]:
        return OrderedDict((name, value.detach().clone()) for name, value in self._state.items())

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        if state.keys() != self._state.keys():
            raise RuntimeError("EMA checkpoint keys differ")
        for name, value in state.items():
            self._state[name].copy_(value)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        target = dict(model.named_parameters())
        target.update(dict(model.named_buffers()))
        if target.keys() != self._state.keys():
            raise RuntimeError("EMA/model state keys differ")
        for name, value in self._state.items():
            target[name].copy_(value)

    @contextmanager
    @torch.no_grad()
    def average_parameters(self, model: nn.Module):
        """Temporarily swap EMA values into ``model`` without a model copy.

        During the context, the EMA storage holds the online values. Swapping
        one tensor at a time bounds temporary CUDA memory to the largest state
        tensor and the ``finally`` block restores both sides after exceptions.
        """
        target = dict(model.named_parameters())
        target.update(dict(model.named_buffers()))
        if target.keys() != self._state.keys():
            raise RuntimeError("EMA/model state keys differ")

        def swap() -> None:
            for name, average in self._state.items():
                online = target[name]
                temporary = online.detach().clone()
                online.copy_(average)
                average.copy_(temporary)

        swap()
        try:
            yield model
        finally:
            swap()
