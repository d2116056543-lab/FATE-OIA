from __future__ import annotations

import torch

from fate_oia.models.head_zoo.base import BaseOIAHead
from fate_oia.models.head_zoo.run_c_compatible_head import RunCCompatibleHead
from fate_oia.models.masked_reason_completion import MaskedReasonCompletion


class RunCMRCAuxHead(BaseOIAHead):
    def __init__(
        self,
        dim: int = 384,
        action_dim: int = 4,
        reason_dim: int = 21,
        mrc_mask_ratio: float = 0.35,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.reason_dim = int(reason_dim)
        self.mrc_mask_ratio = float(mrc_mask_ratio)
        self.base = RunCCompatibleHead(dim=dim, action_dim=action_dim, reason_dim=reason_dim, dropout=dropout)
        self.mrc = MaskedReasonCompletion(dim=dim, reason_dim=reason_dim, num_heads=4, dropout=dropout)

    def forward(self, tokens: torch.Tensor, labels: torch.Tensor | None = None, **kwargs):
        out = dict(self.base(tokens, labels=labels))
        aux = dict(out.get("aux_losses") or {})
        label_tokens = out.get("label_tokens")
        if label_tokens is not None:
            reason_labels = labels[:, self.action_dim :] if labels is not None else None
            mrc_out = self.mrc(label_tokens, reason_labels, mask_ratio=self.mrc_mask_ratio)
            if labels is not None:
                aux["mrc_loss"] = mrc_out["mrc_loss"]
            out["mrc_reason_logits"] = mrc_out["mrc_reason_logits"]
            out["mrc_mask"] = mrc_out["mrc_mask"]
        out["aux_losses"] = aux
        return out
