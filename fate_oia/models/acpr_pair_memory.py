from __future__ import annotations

import torch
from torch import nn


class ACPRPairMemory(nn.Module):
    def __init__(self, dim: int = 384, memory_size: int = 8192, tail_multiplier: float = 2.0) -> None:
        super().__init__()
        self.memory_size = memory_size
        self.tail_multiplier = tail_multiplier
        self.proj = nn.Linear(dim, dim)

    def mine_pairs(self, embeddings: torch.Tensor, action: torch.Tensor, reason: torch.Tensor, tail_indices: list[int] | None = None) -> dict[str, torch.Tensor | int]:
        b = embeddings.shape[0]
        if b < 2:
            empty = torch.empty(0, 2, dtype=torch.long, device=embeddings.device)
            return {"positive_pairs": empty, "contrast_pairs": empty, "tail_pair_count": 0}
        same_action = (action[:, None, :] == action[None, :, :]).float().mean(-1) > 0.99
        reason_overlap = (reason[:, None, :] * reason[None, :, :]).sum(-1) > 0
        diff_reason = (reason[:, None, :] - reason[None, :, :]).abs().sum(-1) > 0
        eye = torch.eye(b, dtype=torch.bool, device=embeddings.device)
        pos_mask = same_action & reason_overlap & ~eye
        contrast_mask = same_action & diff_reason & ~reason_overlap & ~eye
        pos = pos_mask.nonzero(as_tuple=False)[: self.memory_size]
        contrast = contrast_mask.nonzero(as_tuple=False)[: self.memory_size]
        tail_count = 0
        if tail_indices:
            tail = reason[:, tail_indices].sum(-1) > 0
            tail_count = int((tail[pos[:, 0]] | tail[pos[:, 1]]).sum().item()) if pos.numel() else 0
        return {"positive_pairs": pos, "contrast_pairs": contrast, "tail_pair_count": tail_count}

    def forward(self, label_nodes: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(self.proj(label_nodes.mean(1)), dim=-1)
