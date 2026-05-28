from __future__ import annotations

import torch
from torch import nn


class EvidenceTeacherHead(nn.Module):
    """Diagnostic GT-evidence upper-bound head; not a fair patch-only model."""

    def __init__(self, dim: int = 384, action_dim: int = 4, reason_dim: int = 21) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.reason_dim = int(reason_dim)
        self.norm = nn.LayerNorm(dim)
        self.cls = nn.Linear(dim, action_dim + reason_dim)

    def forward(self, patch_summary: torch.Tensor, evidence_tokens: torch.Tensor | None = None, valid_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if evidence_tokens is not None and evidence_tokens.numel() > 0:
            if valid_mask is None:
                summary = evidence_tokens.mean(dim=1)
            else:
                weights = valid_mask.float().unsqueeze(-1)
                summary = (evidence_tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            summary = summary + patch_summary
        else:
            summary = patch_summary
        logits = self.cls(self.norm(summary))
        return {"logits": logits, "action_logits": logits[:, : self.action_dim], "reason_logits": logits[:, self.action_dim :]}
