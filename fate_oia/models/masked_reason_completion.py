from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class MaskedReasonCompletion(nn.Module):
    """Predict masked reason labels from evidence-state tokens and partial labels."""

    def __init__(self, dim: int, reason_dim: int = 21, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.reason_dim = int(reason_dim)
        self.label_embed = nn.Embedding(reason_dim, dim)
        self.value_embed = nn.Embedding(3, dim)  # negative, positive, unknown
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.cls = nn.Linear(dim, 1)

    def build_mask(self, labels: torch.Tensor, mask_ratio: float = 0.30, positive_mask_keep_prob: float = 0.50) -> torch.Tensor:
        rand = torch.rand_like(labels.float())
        mask = rand < float(mask_ratio)
        pos = labels.float() > 0
        pos_rand = torch.rand_like(labels.float())
        mask = mask | (pos & (pos_rand < float(positive_mask_keep_prob) * float(mask_ratio)))
        # guarantee at least one masked label per sample
        if labels.numel() and not bool(mask.any(dim=1).all()):
            miss = torch.where(~mask.any(dim=1))[0]
            idx = torch.randint(0, labels.shape[1], (miss.numel(),), device=labels.device)
            mask[miss, idx] = True
        return mask

    def forward(
        self,
        state_tokens: torch.Tensor,
        reason_labels: torch.Tensor | None = None,
        *,
        mask_ratio: float = 0.30,
        positive_mask_keep_prob: float = 0.50,
        mrc_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        b = state_tokens.shape[0]
        label_ids = torch.arange(self.reason_dim, device=state_tokens.device).unsqueeze(0).expand(b, -1)
        if reason_labels is None:
            mrc_mask = torch.ones(b, self.reason_dim, dtype=torch.bool, device=state_tokens.device)
            values = torch.full((b, self.reason_dim), 2, dtype=torch.long, device=state_tokens.device)
        else:
            labels = reason_labels.float()
            if mrc_mask is None:
                mrc_mask = self.build_mask(labels, mask_ratio, positive_mask_keep_prob)
            values = torch.where(labels > 0, torch.ones_like(labels, dtype=torch.long), torch.zeros_like(labels, dtype=torch.long))
            values = torch.where(mrc_mask, torch.full_like(values, 2), values)
        queries = self.label_embed(label_ids) + self.value_embed(values)
        out, attn = self.attn(queries, state_tokens, state_tokens, need_weights=True, average_attn_weights=False)
        feats = self.norm(queries + out)
        logits = self.cls(feats).squeeze(-1)
        loss = logits.new_zeros(())
        if reason_labels is not None and bool(mrc_mask.any()):
            raw = F.binary_cross_entropy_with_logits(logits, reason_labels.float(), reduction="none")
            loss = raw[mrc_mask].mean()
        return {"mrc_reason_logits": logits, "mrc_mask": mrc_mask, "mrc_loss": loss, "mrc_attention": attn}
