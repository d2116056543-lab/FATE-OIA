from __future__ import annotations

import hashlib
from collections.abc import Sequence

import torch
from torch import nn


class MOSAICSoftRankQueue(nn.Module):
    def __init__(self, label_dim: int, *, capacity: int = 2048) -> None:
        super().__init__()
        if type(label_dim) is not int or label_dim <= 0 or type(capacity) is not int or capacity <= 0:
            raise ValueError("soft rank queue requires positive integer dimensions")
        self.label_dim = label_dim
        self.capacity = capacity
        self.register_buffer("logit_buffer", torch.zeros(capacity, label_dim), persistent=True)
        self.register_buffer("target_buffer", torch.zeros(capacity, label_dim), persistent=True)
        self.register_buffer("sample_hash_buffer", torch.zeros(capacity, dtype=torch.long), persistent=True)
        self.register_buffer("_count", torch.zeros((), dtype=torch.long), persistent=True)
        self.register_buffer("_write_position", torch.zeros((), dtype=torch.long), persistent=True)
        self._count_value = 0
        self._write_position_value = 0

    @property
    def count(self) -> int:
        return self._count_value

    def _load_from_state_dict(self, *args, **kwargs) -> None:
        super()._load_from_state_dict(*args, **kwargs)
        self._count_value = int(self._count.detach().cpu())
        self._write_position_value = int(self._write_position.detach().cpu())

    @staticmethod
    def hash_sample_ids(sample_ids: Sequence[str | int], *, device: torch.device) -> torch.Tensor:
        hashes: list[int] = []
        for sample_id in sample_ids:
            digest = hashlib.sha256(str(sample_id).encode("utf-8")).digest()
            hashes.append(int.from_bytes(digest[:8], byteorder="big", signed=False) & ((1 << 63) - 1))
        return torch.tensor(hashes, dtype=torch.long, device=device)

    @torch.no_grad()
    def enqueue(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_ids: Sequence[str | int],
    ) -> None:
        if logits.ndim != 2 or tuple(logits.shape) != tuple(targets.shape) or logits.shape[1] != self.label_dim:
            raise ValueError("queue logits/targets must have matching [B,label_dim] shapes")
        if len(sample_ids) != logits.shape[0]:
            raise ValueError("queue sample id count must match the batch")
        logits = logits.detach().to(device=self.logit_buffer.device, dtype=self.logit_buffer.dtype)
        targets = targets.detach().to(device=self.target_buffer.device, dtype=self.target_buffer.dtype)
        sample_hashes = self.hash_sample_ids(sample_ids, device=self.sample_hash_buffer.device)
        if logits.shape[0] >= self.capacity:
            self.logit_buffer.copy_(logits[-self.capacity :])
            self.target_buffer.copy_(targets[-self.capacity :])
            self.sample_hash_buffer.copy_(sample_hashes[-self.capacity :])
            self._count.fill_(self.capacity)
            self._write_position.zero_()
            self._count_value = self.capacity
            self._write_position_value = 0
            return
        indices = (
            torch.arange(logits.shape[0], device=self.logit_buffer.device) + self._write_position_value
        ) % self.capacity
        self.logit_buffer.index_copy_(0, indices, logits)
        self.target_buffer.index_copy_(0, indices, targets)
        self.sample_hash_buffer.index_copy_(0, indices, sample_hashes)
        self._write_position_value = (self._write_position_value + logits.shape[0]) % self.capacity
        self._count_value = min(self._count_value + logits.shape[0], self.capacity)
        self._write_position.fill_(self._write_position_value)
        self._count.fill_(self._count_value)

    def snapshot(self) -> dict[str, torch.Tensor]:
        count = self.count
        if count == 0:
            indices = torch.empty(0, dtype=torch.long, device=self.logit_buffer.device)
        elif count < self.capacity:
            indices = torch.arange(count, device=self.logit_buffer.device)
        else:
            indices = (
                torch.arange(self.capacity, device=self.logit_buffer.device) + self._write_position_value
            ) % self.capacity
        return {
            "logits": self.logit_buffer.index_select(0, indices).detach(),
            "targets": self.target_buffer.index_select(0, indices).detach(),
            "sample_hashes": self.sample_hash_buffer.index_select(0, indices).detach(),
        }
