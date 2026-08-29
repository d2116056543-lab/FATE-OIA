from __future__ import annotations

import math

import torch
from torch import nn

from .tida_temporal_encoder import TIDATemporalEncoder


class TIDAReasonLocalTemporalQuery(nn.Module):
    """Target-private reason evidence from ordered local visual queries."""

    def __init__(
        self,
        dim: int = 384,
        num_reasons: int = 21,
        num_heads: int = 4,
        cap: float = 0.08,
        utility_open_prior: float = 0.10,
    ) -> None:
        super().__init__()
        if dim <= 0 or num_reasons <= 0 or num_heads <= 0 or dim % num_heads:
            raise ValueError("invalid reason local-query dimensions")
        if cap <= 0 or not 0.0 < utility_open_prior < 0.5:
            raise ValueError("invalid reason local-query deployment settings")
        self.num_reasons = int(num_reasons)
        self.cap = float(cap)
        self.temporal_encoder = TIDATemporalEncoder(
            dim=dim, num_layers=1, num_heads=num_heads, dropout=0.0
        )
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
        self.deployment_policy_source = "strict_zero_fallback"

    @torch.no_grad()
    def set_deployment_policy(
        self,
        gate: torch.Tensor,
        scale: torch.Tensor,
        cutoff: torch.Tensor,
        *,
        utility_inverted: torch.Tensor | None = None,
        source: str,
    ) -> None:
        if utility_inverted is None:
            utility_inverted = torch.zeros_like(torch.as_tensor(gate))
        values = (gate, scale, cutoff, utility_inverted)
        if any(torch.as_tensor(value).shape != (self.num_reasons,) for value in values):
            raise ValueError("reason local deployment policy must be [R]")
        gate = torch.as_tensor(gate, device=self.deployment_label_gate.device).float()
        scale = torch.as_tensor(scale, device=self.deployment_scale.device).float()
        cutoff = torch.as_tensor(cutoff, device=self.deployment_cutoff.device).float()
        utility_inverted = torch.as_tensor(
            utility_inverted,
            device=self.deployment_utility_inverted.device,
        ).float()
        if not all(
            torch.isfinite(value).all()
            for value in (gate, scale, cutoff, utility_inverted)
        ):
            raise ValueError("reason local deployment policy must be finite")
        self.deployment_label_gate.copy_(gate.clamp(0.0, 1.0))
        self.deployment_scale.copy_(scale.clamp(-1.0, 1.0))
        self.deployment_cutoff.copy_(cutoff.clamp(0.0, 1.0))
        self.deployment_utility_inverted.copy_(utility_inverted.clamp(0.0, 1.0))
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
        history = encoded["history_summary"]
        target = target_reason_tokens.detach()
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
            * self.cap
            * torch.tanh(raw_candidate / self.cap)
        )

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
            deploy_gate * self.deployment_scale[None] * candidate
        ).clamp(-self.cap, self.cap)
        return {
            "reason_local_history_summary": history,
            "reason_local_target_query": target,
            "reason_local_candidate_delta": candidate,
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
            "reason_local_history_available": encoded["history_valid"],
            "reason_local_policy_source": self.deployment_policy_source,
        }
