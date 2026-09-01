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
        self.event_feature = nn.Sequential(
            nn.LayerNorm(3 * dim),
            nn.Linear(3 * dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
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
        self.register_buffer(
            "deployment_center", torch.zeros(self.num_reasons), persistent=True
        )
        self.deployment_policy_source = "strict_zero_fallback"

    @staticmethod
    def _kinematics(
        states: torch.Tensor,
        timestamps: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute signed velocity and acceleration on an irregular timeline."""
        batch, reasons, frames, _ = states.shape
        velocity = torch.zeros_like(states)
        acceleration = torch.zeros_like(states)
        if frames < 2:
            return velocity, acceleration
        dt = (timestamps[:, 1:frames] - timestamps[:, : frames - 1]).clamp_min(1e-4)
        pair_valid = valid[:, 1:frames] & valid[:, : frames - 1]
        velocity[:, :, 1:] = (
            (states[:, :, 1:] - states[:, :, :-1])
            / dt[:, None, :, None]
            * pair_valid[:, None, :, None]
        )
        if frames >= 3:
            acceleration_dt = (
                0.5 * (dt[:, 1:] + dt[:, :-1])
            ).clamp_min(1e-4)
            triple_valid = pair_valid[:, 1:] & pair_valid[:, :-1]
            acceleration[:, :, 2:] = (
                (velocity[:, :, 2:] - velocity[:, :, 1:-1])
                / acceleration_dt[:, None, :, None]
                * triple_valid[:, None, :, None]
            )
        return velocity, acceleration

    def _attend(
        self,
        target: torch.Tensor,
        event_states: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, reasons, frames, dim = event_states.shape
        if valid.ndim == 2:
            valid = valid[:, None].expand(-1, reasons, -1)
        if valid.shape != (batch, reasons, frames):
            raise ValueError("reason event validity shape mismatch")
        query = self.temporal_query_proj(target)
        key = self.temporal_key_proj(event_states)
        value = self.temporal_value_proj(event_states)
        attention_logits = torch.einsum("brd,brtd->brt", query, key) / math.sqrt(dim)
        safe_valid = valid.clone()
        no_history = ~safe_valid.any(-1)
        safe_flat = safe_valid.reshape(batch * reasons, frames)
        no_history_flat = no_history.reshape(batch * reasons)
        safe_flat[no_history_flat, 0] = True
        safe_valid = safe_flat.reshape(batch, reasons, frames)
        attention_logits = attention_logits.masked_fill(
            ~safe_valid, torch.finfo(attention_logits.dtype).min
        )
        attention = attention_logits.softmax(-1).masked_fill(~valid, 0.0)
        attention = attention / attention.sum(-1, keepdim=True).clamp_min(1e-8)
        history = self.temporal_output_proj(
            torch.einsum("brt,brtd->brd", attention, value)
        )
        available = (~no_history).to(history.dtype)
        history = history * available[..., None]
        return history, attention, available

    def _candidate_from_history(
        self,
        target: torch.Tensor,
        history: torch.Tensor,
        available: torch.Tensor,
        temporal_scale: float | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        difference = target - history
        hidden = self.feature(
            torch.cat((target, history, difference, target * history), dim=-1)
        )
        control_hidden = self.feature(
            torch.cat(
                (target, target, torch.zeros_like(target), target * target), dim=-1
            )
        )
        scale = torch.as_tensor(
            temporal_scale, device=hidden.device, dtype=hidden.dtype
        )
        raw = torch.einsum(
            "brd,rd->br", hidden - control_hidden, self.reason_readout_weight
        )
        candidate = (
            available
            * scale
            * bounded_reason_delta(raw, self.cap)
            * self.temporal_reason_mask[None]
        )
        return candidate, hidden, difference

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
        history_valid = frame_valid_mask[:, : history_reason_tokens.shape[1]]
        velocity, acceleration = self._kinematics(
            history_states, timestamps[:, : history_reason_tokens.shape[1]], history_valid
        )
        bounded_velocity = velocity.tanh()
        bounded_acceleration = acceleration.tanh()
        target_relative_change = history_states - target[:, :, None]
        event_states = self.event_feature(
            torch.cat(
                (
                    target_relative_change,
                    bounded_velocity,
                    bounded_acceleration,
                ),
                dim=-1,
            )
        )
        history, temporal_attention, available = self._attend(
            target, event_states, history_valid
        )
        candidate, hidden, _ = self._candidate_from_history(
            target, history, available, temporal_scale
        )

        shuffled_encoded = self.temporal_encoder(
            history_reason_tokens.flip(1), timestamps, frame_valid_mask
        )
        shuffled_velocity, shuffled_acceleration = self._kinematics(
            shuffled_encoded["history_states"],
            timestamps[:, : history_reason_tokens.shape[1]],
            history_valid,
        )
        shuffled_change = shuffled_encoded["history_states"] - target[:, :, None]
        shuffled_states = self.event_feature(
            torch.cat(
                (
                    shuffled_change,
                    shuffled_velocity.tanh(),
                    shuffled_acceleration.tanh(),
                ),
                dim=-1,
            )
        )
        shuffled_valid = history_valid
        shuffled_history, _, shuffled_available = self._attend(
            target, shuffled_states, shuffled_valid
        )
        shuffled_candidate, _, _ = self._candidate_from_history(
            target, shuffled_history, shuffled_available, temporal_scale
        )

        selected_index = temporal_attention.argmax(-1)
        selected_valid = history_valid[:, None].expand(-1, reasons, -1).clone()
        selected_valid.scatter_(2, selected_index[..., None], False)
        reason_index = torch.arange(reasons, device=target.device)[None, :, None]
        frame_index = torch.arange(
            history_reason_tokens.shape[1], device=target.device
        )[None, None]
        batch_index = torch.arange(batch, device=target.device)[:, None, None]
        control_score = torch.sin(
            frame_index * 12.9898 + reason_index * 78.233 + batch_index * 37.719
        )
        control_score = control_score.masked_fill(
            ~history_valid[:, None], torch.finfo(control_score.dtype).min
        )
        control_score.scatter_(2, selected_index[..., None], torch.finfo(control_score.dtype).min)
        control_index = control_score.argmax(-1)
        random_valid = history_valid[:, None].expand(-1, reasons, -1).clone()
        random_valid.scatter_(2, control_index[..., None], False)
        selected_history, _, selected_available = self._attend(
            target, event_states, selected_valid
        )
        random_history, _, random_available = self._attend(
            target, event_states, random_valid
        )
        selected_deleted_candidate, _, _ = self._candidate_from_history(
            target, selected_history, selected_available, temporal_scale
        )
        random_deleted_candidate, _, _ = self._candidate_from_history(
            target, random_history, random_available, temporal_scale
        )
        selected_minus_random_gap = (
            (candidate - selected_deleted_candidate).abs()
            - (candidate - random_deleted_candidate).abs()
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
                available[..., None],
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
        dynamic_mask = history_valid[:, None, :, None].to(velocity.dtype)
        dynamic_denominator = (
            dynamic_mask.sum((2, 3)).clamp_min(1.0) * float(dim)
        )
        velocity_rms = (
            (velocity.square() * dynamic_mask).sum((2, 3))
            / dynamic_denominator
        ).sqrt()
        acceleration_rms = (
            (acceleration.square() * dynamic_mask).sum((2, 3))
            / dynamic_denominator
        ).sqrt()
        return {
            "reason_local_history_summary": history,
            "reason_local_temporal_attention": temporal_attention,
            "reason_local_temporal_attention_entropy": attention_entropy,
            "reason_local_target_query": target,
            "reason_local_candidate_delta": candidate,
            "reason_local_velocity_rms": velocity_rms,
            "reason_local_acceleration_rms": acceleration_rms,
            "reason_local_shuffled_delta": shuffled_candidate,
            "reason_local_selected_deleted_delta": selected_deleted_candidate,
            "reason_local_random_deleted_delta": random_deleted_candidate,
            "reason_local_selected_minus_random_gap": selected_minus_random_gap,
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
