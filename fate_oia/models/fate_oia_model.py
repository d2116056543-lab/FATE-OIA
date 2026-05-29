from __future__ import annotations

import torch
from torch import nn

from fate_oia.models.label_query_head import LabelQueryHead
from fate_oia.models.label_correlation import LabelCorrelationBlock, LegacyLabelCorrelationBlock
from fate_oia.models.reason_to_action_bottleneck import ReasonToActionBottleneck


class FATEOIAFeatureModel(nn.Module):
    """Feature-level FATE-OIA head for SNNA/ViT token features.

    It expects precomputed or backbone-produced tokens [B,N,D]. This keeps the module
    compatible with SNNA checkpoints that are still being trained.
    """

    def __init__(
        self,
        dim: int = 384,
        action_dim: int = 4,
        reason_dim: int = 21,
        use_label_query: bool = True,
        label_correlation: str = "none",
        label_correlation_layers: int = 1,
        label_correlation_heads: int = 4,
        label_correlation_dropout: float = 0.1,
        label_correlation_bias: str = "none",
        label_correlation_bias_matrix: torch.Tensor | None = None,
        label_correlation_bias_weight: float = 0.0,
        label_correlation_residual_init: float = 1.0,
        label_correlation_residual_learnable: bool = True,
        fusion_mode: str = "learned_gate",
        fusion_fixed_alpha: float = 0.0,
        fusion_gate_floor: float = 0.0,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.reason_dim = reason_dim
        self.use_label_query = use_label_query
        self.label_correlation_mode = label_correlation
        if fusion_mode not in {"learned_gate", "gated_floor", "fixed_alpha", "reason_only", "visual_only"}:
            raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")
        self.fusion_mode = fusion_mode
        self.fusion_fixed_alpha = float(fusion_fixed_alpha)
        self.fusion_gate_floor = float(fusion_gate_floor)
        if use_label_query:
            self.label_head = LabelQueryHead(dim, action_dim + reason_dim)
            if label_correlation == "self_attn":
                self.label_correlation = LabelCorrelationBlock(
                    dim=dim,
                    num_labels=action_dim + reason_dim,
                    num_heads=label_correlation_heads,
                    num_layers=label_correlation_layers,
                    dropout=label_correlation_dropout,
                    bias_mode=label_correlation_bias,
                    bias_matrix=label_correlation_bias_matrix,
                    bias_weight=label_correlation_bias_weight,
                    residual_init=label_correlation_residual_init,
                    residual_learnable=label_correlation_residual_learnable,
                )
            elif label_correlation == "self_attn_legacy":
                self.label_correlation = LegacyLabelCorrelationBlock(
                    dim=dim,
                    num_labels=action_dim + reason_dim,
                    num_heads=label_correlation_heads,
                    num_layers=label_correlation_layers,
                    dropout=label_correlation_dropout,
                    bias_mode=label_correlation_bias,
                )
            elif label_correlation == "none":
                self.label_correlation = nn.Identity()
            else:
                raise ValueError(f"Unsupported label_correlation mode: {label_correlation}")
        else:
            self.pool = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU())
            self.action_head = nn.Linear(dim, action_dim)
            self.reason_head = nn.Linear(dim, reason_dim)
        self.reason_to_action = ReasonToActionBottleneck(reason_dim=reason_dim, action_dim=action_dim, hidden_dim=dim)
        self.fusion_gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, action_dim), nn.Sigmoid())

    def _fuse_action_logits(
        self,
        action_visual_logits: torch.Tensor,
        action_reason_logits: torch.Tensor,
        learned_gate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.fusion_mode == "reason_only":
            gate = torch.zeros_like(action_visual_logits)
        elif self.fusion_mode == "visual_only":
            gate = torch.ones_like(action_visual_logits)
        elif self.fusion_mode == "fixed_alpha":
            alpha = min(1.0, max(0.0, self.fusion_fixed_alpha))
            gate = torch.full_like(action_visual_logits, alpha)
        elif self.fusion_mode == "gated_floor":
            floor = min(0.49, max(0.0, self.fusion_gate_floor))
            gate = floor + (1.0 - 2.0 * floor) * learned_gate
        else:
            gate = learned_gate
        fused = gate * action_visual_logits + (1.0 - gate) * action_reason_logits
        return fused, gate

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.use_label_query:
            out = self.label_head(tokens)
            label_tokens = self.label_correlation(out["label_tokens"])
            logits = self.label_head.cls(label_tokens).squeeze(-1)
            action_visual_logits = logits[:, : self.action_dim]
            reason_logits = logits[:, self.action_dim :]
            action_summary = label_tokens[:, : self.action_dim].mean(1)
            reason_summary = label_tokens[:, self.action_dim :].mean(1)
            action_reason_logits = self.reason_to_action(reason_logits)
            learned_gate = self.fusion_gate(torch.cat([action_summary, reason_summary], dim=-1))
            action_fused_logits, gate = self._fuse_action_logits(
                action_visual_logits, action_reason_logits, learned_gate
            )
            return {
                **out,
                "logits": logits,
                "label_tokens": label_tokens,
                "action_logits": action_fused_logits,
                "action_visual_logits": action_visual_logits,
                "action_reason_logits": action_reason_logits,
                "action_fused_logits": action_fused_logits,
                "reason_logits": reason_logits,
                "reason_to_action_logits": action_reason_logits,
                "fusion_gate": gate,
                "fusion_gate_learned": learned_gate,
            }
        pooled = self.pool(tokens.mean(1))
        reason_logits = self.reason_head(pooled)
        action_visual_logits = self.action_head(pooled)
        action_reason_logits = self.reason_to_action(reason_logits)
        learned_gate = torch.sigmoid(action_visual_logits.new_zeros(action_visual_logits.shape))
        action_fused_logits, gate = self._fuse_action_logits(action_visual_logits, action_reason_logits, learned_gate)
        return {
            "action_logits": action_fused_logits,
            "action_visual_logits": action_visual_logits,
            "action_reason_logits": action_reason_logits,
            "action_fused_logits": action_fused_logits,
            "reason_logits": reason_logits,
            "reason_to_action_logits": action_reason_logits,
            "fusion_gate": gate,
            "fusion_gate_learned": learned_gate,
        }
