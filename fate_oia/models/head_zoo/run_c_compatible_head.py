from __future__ import annotations

import torch

from fate_oia.models.fate_oia_model import FATEOIAFeatureModel
from fate_oia.models.head_zoo.base import BaseOIAHead, HeadOutput


class RunCCompatibleHead(BaseOIAHead):
    """Run-C-like label-query/reason-to-action head for HeadZoo sanity checks."""

    def __init__(self, dim: int = 384, action_dim: int = 4, reason_dim: int = 21, dropout: float = 0.1) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.reason_dim = int(reason_dim)
        self.model = FATEOIAFeatureModel(
            dim=dim,
            action_dim=action_dim,
            reason_dim=reason_dim,
            use_label_query=True,
        )

    def forward(self, tokens: torch.Tensor, labels: torch.Tensor | None = None, **kwargs) -> HeadOutput:
        out = self.model(tokens)
        return {
            "logits": torch.cat([out["action_fused_logits"], out["reason_logits"]], dim=1),
            "action_logits": out["action_fused_logits"],
            "reason_logits": out["reason_logits"],
            "label_tokens": out.get("label_tokens"),
            "attention": out.get("attention"),
            "aux_losses": {},
            "action_visual_logits": out.get("action_visual_logits"),
            "action_reason_logits": out.get("action_reason_logits"),
            "fusion_gate": out.get("fusion_gate"),
        }
