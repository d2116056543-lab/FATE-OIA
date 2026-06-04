from __future__ import annotations

import torch
from torch import nn

from fate_oia.models.label_query_head import LabelQueryHead
from fate_oia.models.reason_to_action_bottleneck import ReasonToActionBottleneck


class P3LESharedLabelQueryEncoder(nn.Module):
    """Shared FATE-style label-query stem.

    The encoder preserves the main FATE-OIA action/reason semantics: visual
    action logits, reason logits, reason-to-action logits, and fused action
    logits remain available for diagnostics and safety checks.
    """

    def __init__(self, dim: int = 384, action_dim: int = 4, reason_dim: int = 21, num_heads: int = 4) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.reason_dim = int(reason_dim)
        self.label_head = LabelQueryHead(dim, action_dim + reason_dim, num_heads=num_heads, dropout=0.05)
        self.reason_to_action = ReasonToActionBottleneck(reason_dim=reason_dim, action_dim=action_dim, hidden_dim=dim)
        self.fusion_gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, action_dim), nn.Sigmoid())
        self.context_norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.label_head(tokens)
        label_tokens = out["label_tokens"]
        action_tokens = label_tokens[:, : self.action_dim]
        reason_tokens = label_tokens[:, self.action_dim :]
        logits = out["logits"]
        action_visual_logits = logits[:, : self.action_dim]
        reason_logits = logits[:, self.action_dim :]
        reason_to_action_logits = self.reason_to_action(reason_logits)
        action_summary = action_tokens.mean(dim=1)
        reason_summary = reason_tokens.mean(dim=1)
        fusion_gate = self.fusion_gate(torch.cat([action_summary, reason_summary], dim=-1))
        action_fused_logits = fusion_gate * action_visual_logits + (1.0 - fusion_gate) * reason_to_action_logits
        shared_context = self.context_norm(torch.cat([action_tokens, reason_tokens], dim=1).mean(dim=1))
        return {
            "action_tokens": action_tokens,
            "reason_tokens": reason_tokens,
            "shared_context": shared_context,
            "action_visual_logits": action_visual_logits,
            "action_reason_logits": reason_to_action_logits,
            "reason_to_action_logits": reason_to_action_logits,
            "action_fused_logits": action_fused_logits,
            "action_logits": action_fused_logits,
            "reason_logits": reason_logits,
            "fusion_gate": fusion_gate,
            "attention": out.get("attention"),
        }
