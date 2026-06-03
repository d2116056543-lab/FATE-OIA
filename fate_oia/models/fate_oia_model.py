from __future__ import annotations

import torch
from torch import nn

from fate_oia.models.label_query_head import LabelQueryHead
from fate_oia.models.label_correlation import LabelCorrelationBlock
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
        use_label_correlation: bool = False,
        label_correlation_layers: int = 1,
        label_correlation_heads: int = 4,
        label_correlation_dropout: float = 0.1,
        fusion_mode: str = "learned",
        fusion_fixed_alpha: float = 0.5,
        fusion_gate_floor: float = 0.0,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.reason_dim = reason_dim
        self.use_label_query = use_label_query
        self.use_label_correlation = bool(use_label_correlation)
        self.fusion_mode = fusion_mode
        self.fusion_fixed_alpha = float(fusion_fixed_alpha)
        self.fusion_gate_floor = float(fusion_gate_floor)
        if fusion_mode not in {"learned", "visual", "reason", "fixed_alpha", "gated_floor"}:
            raise ValueError(f"Unsupported fusion_mode={fusion_mode}")
        if use_label_query:
            self.label_head = LabelQueryHead(dim, action_dim + reason_dim)
            if self.use_label_correlation:
                self.label_correlation = LabelCorrelationBlock(
                    dim=dim,
                    num_layers=label_correlation_layers,
                    num_heads=label_correlation_heads,
                    dropout=label_correlation_dropout,
                )
                self.label_correlation_cls = nn.Linear(dim, 1)
            else:
                self.label_correlation = None
                self.label_correlation_cls = None
        else:
            self.pool = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU())
            self.action_head = nn.Linear(dim, action_dim)
            self.reason_head = nn.Linear(dim, reason_dim)
        self.reason_to_action = ReasonToActionBottleneck(reason_dim=reason_dim, action_dim=action_dim, hidden_dim=dim)
        self.fusion_gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, action_dim), nn.Sigmoid())

    def _fuse_actions(
        self,
        action_visual_logits: torch.Tensor,
        action_reason_logits: torch.Tensor,
        learned_gate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.fusion_mode == "visual":
            gate = torch.ones_like(learned_gate)
        elif self.fusion_mode == "reason":
            gate = torch.zeros_like(learned_gate)
        elif self.fusion_mode == "fixed_alpha":
            gate = torch.full_like(learned_gate, min(1.0, max(0.0, self.fusion_fixed_alpha)))
        elif self.fusion_mode == "gated_floor":
            floor = min(1.0, max(0.0, self.fusion_gate_floor))
            gate = floor + (1.0 - floor) * learned_gate
        else:
            gate = learned_gate
        return gate * action_visual_logits + (1.0 - gate) * action_reason_logits, gate

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.use_label_query:
            out = self.label_head(tokens)
            logits = out["logits"]
            label_tokens = out["label_tokens"]
            if self.label_correlation is not None and self.label_correlation_cls is not None:
                raw_label_tokens = label_tokens
                corr = self.label_correlation(raw_label_tokens)
                label_tokens = corr["label_tokens"]
                logits = self.label_correlation_cls(label_tokens).squeeze(-1)
                out = {
                    **out,
                    "logits": logits,
                    "label_tokens": label_tokens,
                    "label_tokens_raw": raw_label_tokens,
                    "label_correlation_attention": corr["attention"],
                    "label_correlation_enabled": torch.ones((), device=tokens.device, dtype=torch.bool),
                }
            else:
                out = {**out, "label_correlation_enabled": torch.zeros((), device=tokens.device, dtype=torch.bool)}
            action_visual_logits = logits[:, : self.action_dim]
            reason_logits = logits[:, self.action_dim :]
            action_summary = label_tokens[:, : self.action_dim].mean(1)
            reason_summary = label_tokens[:, self.action_dim :].mean(1)
            action_reason_logits = self.reason_to_action(reason_logits)
            learned_gate = self.fusion_gate(torch.cat([action_summary, reason_summary], dim=-1))
            action_fused_logits, gate = self._fuse_actions(action_visual_logits, action_reason_logits, learned_gate)
            return {
                **out,
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
        gate = torch.sigmoid(action_visual_logits.new_zeros(action_visual_logits.shape))
        action_fused_logits = gate * action_visual_logits + (1.0 - gate) * action_reason_logits
        return {
            "action_logits": action_fused_logits,
            "action_visual_logits": action_visual_logits,
            "action_reason_logits": action_reason_logits,
            "action_fused_logits": action_fused_logits,
            "reason_logits": reason_logits,
            "reason_to_action_logits": action_reason_logits,
            "fusion_gate": gate,
            "label_correlation_enabled": torch.zeros((), device=tokens.device, dtype=torch.bool),
        }
