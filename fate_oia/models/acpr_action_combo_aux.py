from __future__ import annotations

import torch
from torch import nn


class ACPRActionComboAux(nn.Module):
    def __init__(self, dim: int = 384, action_dim: int = 4) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.register_buffer("subset_membership", self._make_membership(action_dim), persistent=False)
        self.head = nn.Sequential(nn.Linear(dim + action_dim, dim), nn.GELU(), nn.Linear(dim, 16))

    @staticmethod
    def _make_membership(action_dim: int) -> torch.Tensor:
        rows = []
        for i in range(16):
            rows.append([(i >> bit) & 1 for bit in range(action_dim)])
        return torch.tensor(rows, dtype=torch.float32)

    def forward(self, label_nodes: torch.Tensor, action_logits_direct: torch.Tensor) -> dict[str, torch.Tensor]:
        pooled = label_nodes.mean(1)
        action_set_logits = self.head(torch.cat([pooled, action_logits_direct], dim=-1))
        probs = torch.softmax(action_set_logits, dim=-1)
        return {"action_set_logits": action_set_logits, "action_set_probs": probs, "subset_membership": self.subset_membership}
