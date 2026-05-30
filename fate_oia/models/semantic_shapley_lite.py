from __future__ import annotations

import torch
from torch import nn


class SemanticShapleyLite(nn.Module):
    """Cheap evidence contribution proxy used for diagnostics and gate targets."""

    def __init__(self, topk_evidence: int = 8) -> None:
        super().__init__()
        self.topk_evidence = int(topk_evidence)

    def forward(self, reason_logits: torch.Tensor, base_reason_logits: torch.Tensor, evidence_quality: torch.Tensor) -> dict[str, torch.Tensor]:
        effect = torch.relu(reason_logits - base_reason_logits)
        quality = evidence_quality.mean(1, keepdim=True) if evidence_quality.dim() == 2 else evidence_quality
        return {"shapley_proxy": effect * quality, "effect": effect}

