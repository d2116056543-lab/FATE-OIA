from __future__ import annotations

from typing import Any

import torch
from torch import nn

from fate_oia.models.adaptive_calibration import AdaptiveCalibrationHead
from fate_oia.models.evidence_state_bottleneck import EvidenceStateBottleneck
from fate_oia.models.evidence_token_builder import EvidenceTokenBuilder
from fate_oia.models.masked_reason_completion import MaskedReasonCompletion
from fate_oia.models.reason_to_action_bottleneck import ReasonToActionBottleneck


class EviSOIAModel(nn.Module):
    """Evidence-state score branch for BDD-OIA action/reason prediction."""

    def __init__(
        self,
        dim: int,
        action_dim: int = 4,
        reason_dim: int = 21,
        num_state_queries: int = 8,
        evidence_mode: str = "patch_only",
        max_evidence_tokens: int = 32,
        adaptive_calibration: str = "global",
        calibration_delta_clip: float = 2.0,
        enable_mrc: bool = True,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.action_dim = int(action_dim)
        self.reason_dim = int(reason_dim)
        self.evidence_builder = EvidenceTokenBuilder(dim, max_evidence_tokens=max_evidence_tokens, evidence_mode=evidence_mode)
        self.state = EvidenceStateBottleneck(dim, num_state_queries=num_state_queries, num_heads=num_heads, dropout=dropout)
        self.action_queries = nn.Parameter(torch.randn(action_dim, dim) * 0.02)
        self.reason_queries = nn.Parameter(torch.randn(reason_dim, dim) * 0.02)
        self.action_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.reason_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.action_norm = nn.LayerNorm(dim)
        self.reason_norm = nn.LayerNorm(dim)
        self.action_cls = nn.Linear(dim, 1)
        self.reason_cls = nn.Linear(dim, 1)
        self.reason_to_action = ReasonToActionBottleneck(reason_dim, action_dim)
        self.calibration = AdaptiveCalibrationHead(dim, action_dim + reason_dim, mode=adaptive_calibration, delta_clip=calibration_delta_clip)
        self.mrc = MaskedReasonCompletion(dim, reason_dim, num_heads=num_heads, dropout=dropout) if enable_mrc else None

    @property
    def uses_gt_evidence_at_eval(self) -> bool:
        return self.evidence_builder.uses_gt_evidence_at_eval

    def _decode(self, queries: torch.Tensor, state_tokens: torch.Tensor, attn: nn.MultiheadAttention, norm: nn.LayerNorm) -> tuple[torch.Tensor, torch.Tensor]:
        b = state_tokens.shape[0]
        q = queries.unsqueeze(0).expand(b, -1, -1)
        out, weights = attn(q, state_tokens, state_tokens, need_weights=True, average_attn_weights=False)
        return norm(q + out), weights

    def forward(
        self,
        patch_tokens: torch.Tensor,
        patch_grid: tuple[int, int] | list[int] | None = None,
        evidence_metadata: list[dict[str, Any]] | None = None,
        reason_labels: torch.Tensor | None = None,
        *,
        train_mode: bool = True,
        mrc_mask_ratio: float = 0.30,
    ) -> dict[str, torch.Tensor | list[list[dict[str, Any]]]]:
        evidence = self.evidence_builder(patch_tokens, patch_grid=patch_grid, evidence_metadata=evidence_metadata, train=train_mode)
        state_out = self.state(patch_tokens, evidence.tokens, evidence.mask)
        state_tokens = state_out["state_tokens"]
        action_tokens, action_attention = self._decode(self.action_queries, state_tokens, self.action_attn, self.action_norm)
        reason_tokens, reason_attention = self._decode(self.reason_queries, state_tokens, self.reason_attn, self.reason_norm)
        action_visual_logits = self.action_cls(action_tokens).squeeze(-1)
        reason_logits_raw = self.reason_cls(reason_tokens).squeeze(-1)
        action_reason_logits = self.reason_to_action(reason_logits_raw)
        action_logits_raw = 0.5 * action_visual_logits + 0.5 * action_reason_logits
        raw_all = torch.cat([action_logits_raw, reason_logits_raw], dim=1)
        calib = self.calibration(raw_all, state_tokens)
        calibrated = calib["calibrated_logits"]
        action_logits_calibrated = calibrated[:, : self.action_dim]
        reason_logits_calibrated = calibrated[:, self.action_dim :]
        mrc_out: dict[str, torch.Tensor] = {}
        if self.mrc is not None:
            mrc_out = self.mrc(state_tokens, reason_labels, mask_ratio=mrc_mask_ratio)
        return {
            "action_logits_raw": action_logits_raw,
            "reason_logits_raw": reason_logits_raw,
            "action_visual_logits": action_visual_logits,
            "action_reason_logits": action_reason_logits,
            "action_logits_calibrated": action_logits_calibrated,
            "reason_logits_calibrated": reason_logits_calibrated,
            "action_logits": action_logits_calibrated if self.calibration.mode != "none" else action_logits_raw,
            "reason_logits": reason_logits_calibrated if self.calibration.mode != "none" else reason_logits_raw,
            "state_tokens": state_tokens,
            "evidence_tokens": evidence.tokens,
            "evidence_mask": evidence.mask,
            "evidence_info": evidence.info,
            "state_attention": state_out["state_attention"],
            "state_attention_entropy": state_out["state_attention_entropy"],
            "action_attention": action_attention,
            "reason_attention": reason_attention,
            "calibration_bias_global": calib["calibration_bias_global"],
            "calibration_delta_instance": calib["calibration_delta_instance"],
            "calibration_mean_abs_delta": calib["calibration_mean_abs_delta"],
            "calibration_mean_abs_global_bias": calib["calibration_mean_abs_global_bias"],
            **mrc_out,
        }
