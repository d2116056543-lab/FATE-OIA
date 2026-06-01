from __future__ import annotations

import torch
from torch import nn


class UncertaintyRelationMemory(nn.Module):
    def __init__(self, dim: int, slots: int = 8) -> None:
        super().__init__()
        self.memory = nn.Parameter(torch.randn(slots, dim) * 0.02)
        self.key = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim)

    @staticmethod
    def uncertainty_gate(logits: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits.detach())
        margin = (probs - 0.5).abs() * 2.0
        gate = 1.0 - margin.mean(dim=1, keepdim=True)
        return gate.clamp(0.0, 1.0)

    def forward(self, label_tokens: torch.Tensor, logits: torch.Tensor) -> dict[str, torch.Tensor]:
        query = self.key(label_tokens.mean(1))
        weights = torch.softmax(torch.matmul(query, self.memory.t()) / (label_tokens.shape[-1] ** 0.5), dim=-1)
        memory_context = torch.matmul(weights, self.memory)
        gate = self.uncertainty_gate(logits)
        context = self.out(memory_context).unsqueeze(1) * gate.unsqueeze(-1)
        return {"memory_context": context.expand_as(label_tokens), "memory_gate": gate, "memory_weights": weights}
