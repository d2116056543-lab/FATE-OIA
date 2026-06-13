from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ActionSetAuxiliaryHead(nn.Module):
    def __init__(self, dim: int = 384, action_dim: int = 4) -> None:
        super().__init__()
        self.action_dim = action_dim
        subsets = []
        for code in range(2 ** action_dim):
            subsets.append([(code >> i) & 1 for i in range(action_dim)])
        self.register_buffer("subset_membership", torch.tensor(subsets, dtype=torch.float32), persistent=False)
        self.context = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 16))
        self.cardinality_head = nn.Linear(dim, action_dim + 1)

    def forward(self, label_nodes: torch.Tensor, action_logits_direct: torch.Tensor) -> dict[str, torch.Tensor]:
        ctx = label_nodes.mean(1)
        logp = F.logsigmoid(action_logits_direct)
        log1mp = F.logsigmoid(-action_logits_direct)
        member = self.subset_membership.to(action_logits_direct.device, action_logits_direct.dtype)
        atomic_energy = member @ logp.transpose(0, 1)
        neg_energy = (1 - member) @ log1mp.transpose(0, 1)
        base = (atomic_energy + neg_energy).transpose(0, 1)
        action_set_logits = base + self.context(ctx)
        return {"action_set_logits": action_set_logits, "action_set_probs": torch.softmax(action_set_logits, dim=-1), "cardinality_logits": self.cardinality_head(ctx), "subset_membership": member}


def action_subset_targets(action_targets: torch.Tensor) -> torch.Tensor:
    bits = action_targets.long().clamp(0, 1)
    weights = torch.tensor([1, 2, 4, 8], device=bits.device, dtype=torch.long)
    return (bits * weights.view(1, -1)).sum(-1)
