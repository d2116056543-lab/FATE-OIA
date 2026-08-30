from __future__ import annotations

import math

import torch
from torch import nn

from .tida_temporal_encoder import TIDATemporalEncoder


def bounded_reason_delta(raw: torch.Tensor, cap: float) -> torch.Tensor:
    """Bound a residual without destroying its small-signal ranking."""
    if cap <= 0:
        raise ValueError("reason residual cap must be positive")
    return float(cap) * torch.tanh(raw)


class TIDAReasonLocalTemporalQuery(nn.Module):
    """Target-private reason evidence from ordered local visual queries."""

    def __init__(
        self,
        dim: int = 384,
        num_reasons: int = 21,
        num_heads: int = 4,
        cap: float = 0.08,
        utility_open_prior: float = 0.10,
        temporal_reason_indices: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        if dim <= 0 or num_reasons <= 0 or num_heads <= 0 or dim % num_heads:
            raise ValueError("invalid reason local-query dimensions")
        if cap <= 0 or not 0.0 < utility_open_prior < 0.5:
            raise ValueError("invalid reason local-query deployment settings")
        self.num_reasons = int(num_reasons)
        self.cap = float(cap)
        if temporal_reason_indices is None:
            temporal_reason_indices = tuple(range(self.num_reasons))
        if any(index < 0 or index >= self.num_reasons for index in temporal_reason_indices):
            raise ValueError("temporal reason index out of range")
        temporal_reason_mask = torch.zeros(self.num_reasons)
        temporal_reason_mask[list(dict.fromkeys(temporal_reason_indices))] = 1.0
        self.register_buffer(
            "temporal_reason_mask", temporal_reason_mask, persistent=True
        )
        self.temporal_encoder = TIDATemporalEncoder(
            dim=dim, num_layers=1, num_heads=num_heads, dropout=0.0
        )
        # The target reason token must query the complete ordered history.
        # Using only the encoder's last state discards short-lived events and
        # reduces this branch to a generic clip-summary residual.
        self.temporal_query_proj = nn.Linear(dim, dim, bias=False)
        self.temporal_key_proj = nn.Linear(dim, dim, bias=False)
        self.temporal_value_proj = nn.Linear(dim, dim, bias=False)
        self.temporal_output_proj = nn.Linear(dim, dim, bias=False)
        self.feature = nn.Sequential(
            nn.LayerNorm(4 * dim),
            nn.Linear(4 * dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
        )
        # Each reason owns its readout direction. Sharing one scalar readout
        # across all labels makes the branch learn generic clip novelty rather
        # than target-specific evidence, even when the query tokens differ.
        self.reason_readout_weight = nn.Parameter(
            torch.zeros(self.num_reasons, dim)
        )
        self.utility = nn.Sequential(
            nn.LayerNorm(dim + 6),
            nn.Linear(dim + 6, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )
        nn.init.zeros_(self.utility[-1].weight)
        nn.init.constant_(
            self.utility[-1].bias,
            math.log(float(utility_open_prior) / (1.0 - float(utility_open_prior))),
        )
        self.register_buffer(
            "deployment_label_gate", torch.zeros(self.num_reasons), persistent=True
        )
        self.register_buffer(
            "deployment_scale", torch.zeros(self.num_reasons), persistent=True
        )
        self.register_buffer(
            "deployment_cutoff", torch.full((self.num_reasons,), 0.5), persistent=True
        )
        self.register_buffer(
            "deployment_utility_inverted",
            torch.zeros(self.num_reasons),
            persistent=True,
        )
        self.register_buffer(
            "deployment_center", torch.zeros(self.num_reasons), persistent=True
        )
        self.deployment_policy_source = "strict_zero_fallback"

    @torch.no_grad()
    def set_deployment_policy(
        self,
        gate: torch.Tensor,
        scale: torch.Tensor,
        cutoff: torch.Tensor,
        *,
        utility_inverted: torch.Tensor | None = None,
        center: torch.Tensor | None = None,
        source: str,
    ) -> None:
        if utility_inverted is None:
            utility_inverted = torch.zeros_like(torch.as_tensor(gate))
        if center is None:
            center = torch.zeros_like(torch.as_tensor(gate))
        values = (gate, scale, cutoff, utility_inverted, center)
        if any(torch.as_tensor(value).shape != (self.num_reasons,) for value in values):
            raise ValueError("reason local deployment policy must be [R]")
        gate = torch.as_tensor(gate, device=self.deployment_label_gate.device).float()
        scale = torch.as_tensor(scale, device=self.deployment_scale.device).float()
        cutoff = torch.as_tensor(cutoff, device=self.deployment_cutoff.device).float()
        utility_inverted = torch.as_tensor(
            utility_inverted,
            device=self.deployment_utility_inverted.device,
        ).float()
        center = torch.as_tensor(
            center, device=self.deployment_center.device
        ).float()
        if not all(
            torch.isfinite(value).all()
            for value in (gate, scale, cutoff, utility_inverted, center)
        ):
            raise ValueError("reason local deployment policy must be finite")
        self.deployment_label_gate.copy_(gate.clamp(0.0, 1.0))
        self.deployment_scale.copy_(scale.clamp(-1.0, 1.0))
        self.deployment_cutoff.copy_(cutoff.clamp(0.0, 1.0))
        self.deployment_utility_inverted.copy_(utility_inverted.clamp(0.0, 1.0))
        self.deployment_center.copy_(
            center.clamp(-self.cap, self.cap) * self.temporal_reason_mask
        )
        self.deployment_policy_source = str(source)

    def forward(
        self,
        history_reason_tokens: torch.Tensor,
        target_reason_tokens: torch.Tensor,
        timestamps: torch.Tensor,
        frame_valid_mask: torch.Tensor,
        *,
        image_logits: torch.Tensor,
        temporal_scale: float | torch.Tensor = 1.0,
    ) -> dict[str, torch.Tensor]:
        if history_reason_tokens.ndim != 4:
            raise ValueError("history_reason_tokens must be [B,T,R,D]")
        batch, _, reasons, dim = history_reason_tokens.shape
        if target_reason_tokens.shape != (batch, reasons, dim):
            raise ValueError("target_reason_tokens shape mismatch")
        if reasons != self.num_reasons or image_logits.shape != (batch, reasons):
            raise ValueError("reason count or image logits shape mismatch")

        encoded = self.temporal_encoder(
            history_reason_tokens, timestamps, frame_valid_mask
        )
        target = target_reason_tokens.detach()
        history_states = encoded["history_states"]
        query = self.temporal_query_proj(target)
        key = self.temporal_key_proj(history_states)
        value = self.temporal_value_proj(history_states)
        attention_logits = torch.einsum("brd,brtd->brt", query, key) / math.sqrt(dim)
        history_valid = frame_valid_mask[:, : history_reason_tokens.shape[1]]
        safe_valid = history_valid.clone()
        no_history = ~safe_valid.any(-1)
        safe_valid[no_history, 0] = True
        attention_logits = attention_logits.masked_fill(
            ~safe_valid[:, None], torch.finfo(attention_logits.dtype).min
        )
        temporal_attention = attention_logits.softmax(-1)
        temporal_attention = temporal_attention.masked_fill(
            ~history_valid[:, None], 0.0
        )
        temporal_attention = temporal_attention / temporal_attention.sum(
            -1, keepdim=True
        ).clamp_min(1e-8)
        history = self.temporal_output_proj(
            torch.einsum("brt,brtd->brd", temporal_attention, value)
        )
        history = history * encoded["history_valid"][:, None, None].to(history.dtype)
        difference = target - history
        hidden = self.feature(
            torch.cat((target, history, difference, target * history), dim=-1)
        )
        # Paired target-only control prevents the residual from collapsing to
        # a constant per-label bias. Both paths share every parameter, so only
        # information contributed by ordered history survives the subtraction.
        control_hidden = self.feature(
            torch.cat(
                (target, target, torch.zeros_like(target), target * target), dim=-1
            )
        )
        scale = torch.as_tensor(
            temporal_scale, device=hidden.device, dtype=hidden.dtype
        )
        available = encoded["history_valid"][:, None].to(hidden.dtype)
        raw_candidate = torch.einsum(
            "brd,rd->br", hidden - control_hidden, self.reason_readout_weight
        )
        candidate = (
            available
            * scale
            * bounded_reason_delta(raw_candidate, self.cap)
            * self.temporal_reason_mask[None]
        )
        centered_candidate = available * (
            candidate - self.deployment_center[None]
        ) * self.temporal_reason_mask[None]

        cosine = torch.nn.functional.cosine_similarity(target, history, dim=-1)
        # Preserve target identity in the motion weighting used by the local
        # PU objectives. A shared clip-level motion score can otherwise train
        # every reason from activity that belongs to a different label.
        local_motion_energy = (
            0.5 * (1.0 - cosine).clamp(0.0, 2.0) * available
        )[:, None, :]
        utility_feature = torch.cat(
            (
                hidden.detach(),
                image_logits.detach()[..., None],
                image_logits.detach().abs()[..., None],
                candidate.detach()[..., None],
                candidate.detach().abs()[..., None],
                cosine.detach()[..., None],
                available[:, :, None].expand(-1, reasons, -1),
            ),
            dim=-1,
        )
        utility_logit = self.utility(utility_feature).squeeze(-1)
        utility_probability = utility_logit.sigmoid()
        attention_entropy = -(
            temporal_attention
            * temporal_attention.clamp_min(1e-8).log()
        ).sum(-1)
        valid_count = history_valid.sum(-1).clamp_min(2).to(attention_entropy.dtype)
        attention_entropy = attention_entropy / valid_count.log()[:, None]
        # Train-calib OOF policy is the only mechanism allowed to open this
        # route. Every unselected label remains an exact image fallback.
        selection_probability = torch.where(
            self.deployment_utility_inverted[None] > 0.5,
            1.0 - utility_probability,
            utility_probability,
        )
        selected = selection_probability >= self.deployment_cutoff[None]
        deploy_gate = (
            selected.to(candidate.dtype)
            * self.deployment_label_gate[None]
            * available
        )
        deploy_delta = (
            deploy_gate * self.deployment_scale[None] * centered_candidate
        ).clamp(-self.cap, self.cap)
        return {
            "reason_local_history_summary": history,
            "reason_local_temporal_attention": temporal_attention,
            "reason_local_temporal_attention_entropy": attention_entropy,
            "reason_local_target_query": target,
            "reason_local_candidate_delta": candidate,
            "reason_local_centered_candidate_delta": centered_candidate,
            "reason_local_centered_candidate_logits": image_logits + centered_candidate,
            "reason_local_motion_energy": local_motion_energy,
            "reason_local_candidate_logits": image_logits + candidate,
            "reason_local_utility_logit": utility_logit,
            "reason_local_utility_probability": utility_probability,
            "reason_local_deploy_gate": deploy_gate,
            "reason_local_deploy_scale": self.deployment_scale[None].expand_as(candidate),
            "reason_local_deploy_utility_inverted": (
                self.deployment_utility_inverted[None].expand_as(candidate)
            ),
            "reason_local_deploy_delta": deploy_delta,
            "reason_local_deploy_logits": image_logits + deploy_delta,
            "reason_local_deployment_center": self.deployment_center,
            "reason_local_temporal_reason_mask": self.temporal_reason_mask,
            "reason_local_history_available": encoded["history_valid"],
            "reason_local_policy_source": self.deployment_policy_source,
        }
