"""Batch-local DINO field reuse with no cross-batch persistence."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch


class BatchLocalDinoFieldReuse:
    """Memoize one extractor call for one exact image tensor identity.

    The cache is intentionally scoped to the object instance and cleared when
    a different tensor is requested. It is a branch-local computation reuse,
    not a feature cache and never persists across batches or checkpoints.
    """

    def __init__(self, extractor: Callable[[torch.Tensor], Mapping[str, Any]]) -> None:
        self.extractor = extractor
        self._image_identity: int | None = None
        self._field: Mapping[str, Any] | None = None

    def clear(self) -> None:
        self._image_identity = None
        self._field = None

    def __call__(self, images: torch.Tensor) -> Mapping[str, Any]:
        identity = id(images)
        if self._image_identity != identity or self._field is None:
            self._field = self.extractor(images)
            self._image_identity = identity
        return self._field
