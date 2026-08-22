from __future__ import annotations

from collections.abc import Iterator, Sized
from typing import Any

import torch
from torch.utils.data import Sampler


class TIDAStatefulRandomSampler(Sampler[tuple[int, int]]):
    """Deterministic epoch sampler whose checkpoint cursor ignores worker prefetch."""

    def __init__(self, data_source: Sized, *, seed: int) -> None:
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = 0
        self.consumed = 0
        self._permutation = self._make_permutation(self.epoch)

    def _make_permutation(self, epoch: int) -> list[int]:
        generator = torch.Generator().manual_seed(self.seed + int(epoch))
        return torch.randperm(len(self.data_source), generator=generator).tolist()

    def __iter__(self) -> Iterator[tuple[int, int]]:
        for position in range(self.consumed, len(self._permutation)):
            index = self._permutation[position]
            augmentation_seed = self.seed + self.epoch * max(len(self._permutation), 1) + position
            yield index, augmentation_seed

    def __len__(self) -> int:
        return len(self._permutation) - self.consumed

    @property
    def epoch_complete(self) -> bool:
        return self.consumed == len(self._permutation)

    def mark_consumed(self, count: int) -> None:
        next_position = self.consumed + int(count)
        if next_position > len(self._permutation):
            raise RuntimeError("sampler consumed beyond the current epoch permutation")
        self.consumed = next_position

    def advance_epoch(self) -> None:
        if not self.epoch_complete:
            raise RuntimeError("cannot advance an incompletely consumed sampler epoch")
        self.epoch += 1
        self.consumed = 0
        self._permutation = self._make_permutation(self.epoch)

    def state_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "epoch": self.epoch,
            "consumed": self.consumed,
            "dataset_size": len(self.data_source),
            "permutation": list(self._permutation),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state["seed"]) != self.seed or int(state["dataset_size"]) != len(self.data_source):
            raise RuntimeError("sampler identity differs from the checkpoint")
        permutation = [int(value) for value in state["permutation"]]
        if sorted(permutation) != list(range(len(self.data_source))):
            raise RuntimeError("checkpoint sampler permutation is invalid")
        consumed = int(state["consumed"])
        if not 0 <= consumed <= len(permutation):
            raise RuntimeError("checkpoint sampler cursor is invalid")
        self.epoch = int(state["epoch"])
        self.consumed = consumed
        self._permutation = permutation
