from __future__ import annotations

import torch


class TIDAActionRankMemory:
    """Detached fixed-capacity memory for cross-update action ranking."""

    def __init__(
        self,
        *,
        capacity: int,
        num_actions: int,
        device: torch.device | str,
    ) -> None:
        if capacity < 0:
            raise ValueError("capacity must be non-negative")
        if num_actions <= 0:
            raise ValueError("num_actions must be positive")
        self.capacity = int(capacity)
        self.num_actions = int(num_actions)
        self.device = torch.device(device)
        self._logits = torch.empty(self.capacity, self.num_actions, device=self.device)
        self._targets = torch.empty(self.capacity, self.num_actions, device=self.device)
        self._write = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    @torch.no_grad()
    def enqueue(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        expected = f"[B,{self.num_actions}]"
        if logits.ndim != 2 or logits.shape[1] != self.num_actions:
            raise ValueError(f"action logits must have shape {expected}")
        if targets.shape != logits.shape:
            raise ValueError(f"action targets must have shape {expected}")
        if self.capacity == 0:
            return
        logits = logits.detach().to(device=self.device, dtype=self._logits.dtype)
        targets = targets.detach().to(device=self.device, dtype=self._targets.dtype)
        if logits.shape[0] >= self.capacity:
            logits = logits[-self.capacity :]
            targets = targets[-self.capacity :]
        count = int(logits.shape[0])
        first = min(count, self.capacity - self._write)
        self._logits[self._write : self._write + first].copy_(logits[:first])
        self._targets[self._write : self._write + first].copy_(targets[:first])
        remainder = count - first
        if remainder:
            self._logits[:remainder].copy_(logits[first:])
            self._targets[:remainder].copy_(targets[first:])
        self._write = (self._write + count) % self.capacity
        self._size = min(self.capacity, self._size + count)

    @torch.no_grad()
    def snapshot(self) -> dict[str, torch.Tensor] | None:
        if self._size == 0:
            return None
        if self._size < self.capacity:
            logits = self._logits[: self._size]
            targets = self._targets[: self._size]
        else:
            logits = torch.cat((self._logits[self._write :], self._logits[: self._write]))
            targets = torch.cat((self._targets[self._write :], self._targets[: self._write]))
        return {
            "action_logits": logits.detach(),
            "action_target": targets.detach(),
        }

    @torch.no_grad()
    def reset(self) -> None:
        self._write = 0
        self._size = 0
