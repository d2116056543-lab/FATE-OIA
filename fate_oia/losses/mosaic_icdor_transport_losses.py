from __future__ import annotations

import torch
from torch.nn import functional as F


class MOSAICDetachedPosteriorQueue:
    """Fixed-capacity ring queue for detached posterior ranking candidates."""

    def __init__(self, *, capacity: int, label_count: int = 21, device: str | torch.device = "cuda") -> None:
        if type(capacity) is not int or capacity <= 0 or type(label_count) is not int or label_count <= 0:
            raise ValueError("IC-DOR posterior queue capacity and label_count must be positive integers")
        self.capacity = capacity
        self.label_count = label_count
        self.logits = torch.zeros(capacity, label_count, device=device)
        self.posterior = torch.zeros(capacity, label_count, device=device)
        self.sample_ids = torch.full((capacity,), -1, dtype=torch.long, device=device)
        self.size = 0
        self.write_index = 0

    @torch.no_grad()
    def enqueue(self, logits: torch.Tensor, posterior: torch.Tensor, sample_ids: torch.Tensor) -> None:
        if logits.shape != posterior.shape or logits.ndim != 2 or logits.shape[1] != self.label_count:
            raise ValueError("IC-DOR posterior queue expects matching [B,R] logits/posteriors")
        if sample_ids.shape != (logits.shape[0],):
            raise ValueError("IC-DOR posterior queue sample IDs must be [B]")
        logits = logits.detach().to(device=self.logits.device, dtype=self.logits.dtype)
        posterior = posterior.detach().to(device=self.posterior.device, dtype=self.posterior.dtype)
        sample_ids = sample_ids.detach().to(device=self.sample_ids.device, dtype=torch.long)
        if logits.shape[0] >= self.capacity:
            logits = logits[-self.capacity :]
            posterior = posterior[-self.capacity :]
            sample_ids = sample_ids[-self.capacity :]
            self.logits.copy_(logits)
            self.posterior.copy_(posterior)
            self.sample_ids.copy_(sample_ids)
            self.size = self.capacity
            self.write_index = 0
            return
        for index in range(logits.shape[0]):
            slot = (self.write_index + index) % self.capacity
            self.logits[slot].copy_(logits[index])
            self.posterior[slot].copy_(posterior[index])
            self.sample_ids[slot].copy_(sample_ids[index])
        self.write_index = (self.write_index + logits.shape[0]) % self.capacity
        self.size = min(self.capacity, self.size + logits.shape[0])

    def posterior_rank_loss(self, logits: torch.Tensor, posterior: torch.Tensor, sample_ids: torch.Tensor, *, margin: float = 0.10) -> torch.Tensor:
        if self.size == 0:
            return logits.sum() * 0.0
        stored_logits = self.logits[: self.size]
        stored_posterior = self.posterior[: self.size]
        stored_ids = self.sample_ids[: self.size]
        valid = sample_ids[:, None] != stored_ids[None, :]
        if not valid.any():
            return logits.sum() * 0.0
        probability = torch.sigmoid(logits)
        stored_probability = torch.sigmoid(stored_logits)
        pair_weight = posterior.detach().unsqueeze(1) * (1.0 - stored_posterior).unsqueeze(0)
        pair_weight = pair_weight * valid.unsqueeze(-1).to(pair_weight.dtype)
        hinge = F.relu(margin - probability.unsqueeze(1) + stored_probability.unsqueeze(0))
        return (hinge * pair_weight).sum() / pair_weight.sum().clamp_min(1.0)
