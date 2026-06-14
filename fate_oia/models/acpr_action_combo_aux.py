from __future__ import annotations

import torch
from torch import nn


class ACPRActionComboAux(nn.Module):
    def __init__(self, dim: int = 384, action_dim: int = 4) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.register_buffer("subset_membership", self._make_membership(action_dim), persistent=False)
        self.head = nn.Sequential(nn.Linear(dim + action_dim, dim), nn.GELU(), nn.Linear(dim, 16))
        self.cardinality_head = nn.Sequential(nn.Linear(dim + action_dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, action_dim + 1))

    @staticmethod
    def _make_membership(action_dim: int) -> torch.Tensor:
        rows = []
        for i in range(16):
            rows.append([(i >> bit) & 1 for bit in range(action_dim)])
        return torch.tensor(rows, dtype=torch.float32)

    def forward(self, label_nodes: torch.Tensor, action_logits_direct: torch.Tensor) -> dict[str, torch.Tensor]:
        pooled = label_nodes.mean(1)
        fused = torch.cat([pooled, action_logits_direct], dim=-1)
        action_set_logits = self.head(fused)
        cardinality_logits = self.cardinality_head(fused)
        probs = torch.softmax(action_set_logits, dim=-1)
        pred_subset = probs.argmax(-1)
        pred_cardinality = self.subset_membership[pred_subset].sum(-1)
        combo_stats = {
            "pred_cardinality_mean": float(pred_cardinality.float().mean().detach().cpu()),
            "action_set_entropy": float((-(probs.clamp_min(1e-9).log() * probs).sum(-1)).mean().detach().cpu()),
        }
        return {
            "action_set_logits": action_set_logits,
            "action_set_probs": probs,
            "cardinality_logits": cardinality_logits,
            "subset_membership": self.subset_membership,
            "combo_stats": combo_stats,
        }
