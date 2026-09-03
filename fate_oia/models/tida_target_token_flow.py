from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class TIDATargetTokenFlow(nn.Module):
    """Predict a label-private terminal token from ordered visual history."""

    def __init__(
        self,
        num_labels: int,
        dim: int = 384,
        hidden_dim: int = 128,
        num_heads: int = 4,
        cap: float = 0.05,
        utility_open_prior: float = 0.10,
        innovation_weight: float = 1.0,
        motion_weight: float = 0.0,
        order_weight: float = 0.0,
        candidate_temperature: float | None = 1.0,
    ) -> None:
        super().__init__()
        if num_labels < 1 or dim < 1 or hidden_dim < 1:
            raise ValueError("target-token dimensions must be positive")
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if cap <= 0 or not 0.0 < utility_open_prior < 0.5:
            raise ValueError("invalid target-token deployment settings")
        self.num_labels = int(num_labels)
        self.dim = int(dim)
        self.hidden_dim = int(hidden_dim)
        self.cap = float(cap)
        branch_weights = (innovation_weight, motion_weight, order_weight)
        if any(float(value) < 0.0 for value in branch_weights) or not any(
            float(value) > 0.0 for value in branch_weights
        ):
            raise ValueError("target-token feature weights must be non-negative and non-zero")
        self.innovation_weight = float(innovation_weight)
        self.motion_weight = float(motion_weight)
        self.order_weight = float(order_weight)
        self.candidate_temperature = float(
            math.sqrt(dim) if candidate_temperature is None else candidate_temperature
        )
        if self.candidate_temperature <= 0.0:
            raise ValueError("target-token candidate temperature must be positive")

        self.input_projection = nn.Linear(dim, hidden_dim)
        self.label_embedding = nn.Parameter(torch.zeros(num_labels, hidden_dim))
        self.time_projection = nn.Sequential(
            nn.Linear(4, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.temporal_depthwise = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_attention = nn.TransformerEncoder(
            layer, num_layers=1, norm=nn.LayerNorm(hidden_dim)
        )
        self.terminal_projection = nn.Linear(hidden_dim, dim)
        self.innovation_norm = nn.LayerNorm(dim)
        # Each target owns its scalar direction. Zero initialization gives an
        # exact image fallback while the terminal predictor learns immediately.
        self.candidate_output_weight = nn.Parameter(torch.zeros(num_labels, dim))
        self.utility = nn.Sequential(
            nn.Linear(6, 16), nn.GELU(), nn.Linear(16, 1)
        )
        nn.init.zeros_(self.utility[-1].weight)
        nn.init.constant_(
            self.utility[-1].bias,
            math.log(utility_open_prior / (1.0 - utility_open_prior)),
        )
        self.register_buffer("deployment_label_gate", torch.zeros(num_labels))
        self.register_buffer("deployment_scale", torch.zeros(num_labels))
        self.register_buffer("deployment_cutoff", torch.full((num_labels,), 0.5))
        self.register_buffer("deployment_center", torch.zeros(num_labels))
        self.register_buffer("deployment_utility_inverted", torch.zeros(num_labels))
        self.deployment_policy_source = "strict_zero_fallback"

    @staticmethod
    def _time_features(timestamps: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            (
                timestamps,
                timestamps.square(),
                torch.sin(timestamps),
                torch.cos(timestamps),
            ),
            dim=-1,
        )

    @staticmethod
    def _last_valid_repeat(
        history: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        batch, frames, labels, dim = history.shape
        positions = torch.arange(frames, device=history.device)[None]
        last = positions.masked_fill(~valid, -1).amax(-1).clamp_min(0)
        selected = history[
            torch.arange(batch, device=history.device), last
        ]
        return selected[:, None].expand(batch, frames, labels, dim)

    def _predict(
        self,
        history: torch.Tensor,
        history_timestamps: torch.Tensor,
        history_valid: torch.Tensor,
    ) -> torch.Tensor:
        batch, frames, labels, _ = history.shape
        hidden = self.input_projection(history)
        hidden = hidden + self.label_embedding[None, None]
        hidden = hidden + self.time_projection(
            self._time_features(history_timestamps)
        )[:, :, None]
        flat = hidden.permute(0, 2, 3, 1).reshape(
            batch * labels, self.hidden_dim, frames
        )
        hidden = hidden + self.temporal_depthwise(flat).reshape(
            batch, labels, self.hidden_dim, frames
        ).permute(0, 3, 1, 2)
        sequence = hidden.transpose(1, 2).reshape(
            batch * labels, frames, self.hidden_dim
        )
        valid = history_valid[:, None].expand(-1, labels, -1).reshape(
            batch * labels, frames
        )
        safe_valid = valid.clone()
        unavailable = ~safe_valid.any(-1)
        safe_valid[unavailable, 0] = True
        sequence = sequence.clone()
        sequence[unavailable, 0] = 0
        causal = torch.triu(
            torch.ones(frames, frames, dtype=torch.bool, device=history.device),
            diagonal=1,
        )
        encoded = self.temporal_attention(
            sequence, mask=causal, src_key_padding_mask=~safe_valid
        )
        positions = torch.arange(frames, device=history.device)[None]
        last = positions.expand_as(safe_valid).masked_fill(~safe_valid, -1).amax(-1)
        last = last.clamp_min(0)
        pooled = encoded[
            torch.arange(batch * labels, device=history.device), last
        ]
        pooled[unavailable] = 0
        return self.terminal_projection(pooled).reshape(batch, labels, self.dim)

    @staticmethod
    def _prediction_error(
        prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        cosine = 1.0 - F.cosine_similarity(prediction, target, dim=-1)
        smooth_l1 = F.smooth_l1_loss(prediction, target, reduction="none").mean(-1)
        return cosine + smooth_l1

    @torch.no_grad()
    def set_deployment_policy(
        self,
        gate: torch.Tensor,
        scale: torch.Tensor,
        cutoff: torch.Tensor,
        *,
        center: torch.Tensor | None = None,
        utility_inverted: torch.Tensor | None = None,
        source: str,
    ) -> None:
        values = tuple(
            torch.as_tensor(value, dtype=torch.float32, device=self.deployment_scale.device)
            for value in (gate, scale, cutoff)
        )
        if any(value.shape != (self.num_labels,) for value in values):
            raise ValueError("target-token deployment policy must be [num_labels]")
        if not all(torch.isfinite(value).all() for value in values):
            raise ValueError("target-token deployment policy must be finite")
        self.deployment_label_gate.copy_(values[0].clamp(0.0, 1.0))
        self.deployment_scale.copy_(values[1].clamp(-1.0, 1.0))
        self.deployment_cutoff.copy_(values[2].clamp(0.0, 1.0))
        if center is not None:
            center_value = torch.as_tensor(
                center, dtype=torch.float32, device=self.deployment_center.device
            )
            if center_value.shape != (self.num_labels,) or not torch.isfinite(center_value).all():
                raise ValueError("target-token deployment center must be finite [num_labels]")
            self.deployment_center.copy_(center_value)
        if utility_inverted is not None:
            inverted_value = torch.as_tensor(
                utility_inverted,
                dtype=torch.float32,
                device=self.deployment_utility_inverted.device,
            )
            if inverted_value.shape != (self.num_labels,):
                raise ValueError("target-token utility inversion must be [num_labels]")
            self.deployment_utility_inverted.copy_(inverted_value.clamp(0.0, 1.0))
        self.deployment_policy_source = str(source)

    def forward(
        self,
        history_tokens: torch.Tensor,
        terminal_tokens: torch.Tensor,
        timestamps: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        base_logits: torch.Tensor,
        temporal_scale: float | torch.Tensor = 1.0,
    ) -> dict[str, torch.Tensor | str]:
        if history_tokens.ndim != 4:
            raise ValueError("history_tokens must be [B,T,L,D]")
        batch, frames, labels, dim = history_tokens.shape
        if labels != self.num_labels or dim != self.dim:
            raise ValueError("history target-token shape mismatch")
        if terminal_tokens.shape != (batch, labels, dim):
            raise ValueError("terminal_tokens shape mismatch")
        if timestamps.shape != (batch, frames + 1):
            raise ValueError("timestamps must include history and target")
        if valid_mask.shape != (batch, frames + 1):
            raise ValueError("valid_mask must include history and target")
        if base_logits.shape != (batch, labels):
            raise ValueError("base_logits shape mismatch")

        # The image/query stack is the frozen strong baseline. Temporal owners
        # learn to read its tokens but must never update or retain its graph.
        history_tokens = history_tokens.detach()
        history_valid = valid_mask[:, :frames]
        available = history_valid.any(-1)
        target = terminal_tokens.detach()
        base = base_logits.detach()
        ordered = self._predict(
            history_tokens, timestamps[:, :frames], history_valid
        )
        reversed_prediction = self._predict(
            history_tokens.flip(1), timestamps[:, :frames], history_valid.flip(1)
        )
        repeated_history = self._last_valid_repeat(history_tokens, history_valid)
        repeated_prediction = self._predict(
            repeated_history, timestamps[:, :frames], history_valid
        )
        shuffle_index = torch.cat(
            (
                torch.arange(1, frames, 2, device=history_tokens.device),
                torch.arange(0, frames, 2, device=history_tokens.device),
            )
        )
        shuffled_prediction = self._predict(
            history_tokens[:, shuffle_index],
            timestamps[:, :frames],
            history_valid[:, shuffle_index],
        )
        availability = available[:, None].to(ordered.dtype)
        innovation = self.innovation_norm(target - ordered)
        motion = F.layer_norm(ordered - repeated_prediction, (self.dim,))
        order = F.layer_norm(ordered - shuffled_prediction, (self.dim,))
        innovation_score = torch.einsum(
            "bld,ld->bl", innovation, self.candidate_output_weight
        ) / self.candidate_temperature
        motion_score = torch.einsum(
            "bld,ld->bl", motion, self.candidate_output_weight
        ) / self.candidate_temperature
        order_score = torch.einsum(
            "bld,ld->bl", order, self.candidate_output_weight
        ) / self.candidate_temperature
        raw_delta = (
            self.innovation_weight * innovation_score
            + self.motion_weight * motion_score
            + self.order_weight * order_score
        )
        scale = torch.as_tensor(
            temporal_scale, device=raw_delta.device, dtype=raw_delta.dtype
        )
        candidate_delta = (
            availability * scale * self.cap * torch.tanh(raw_delta)
        ).clamp(-self.cap, self.cap)
        centered_candidate_delta = candidate_delta - self.deployment_center[None]

        ordered_error = self._prediction_error(ordered, target)
        reversed_error = self._prediction_error(reversed_prediction, target)
        repeated_error = self._prediction_error(repeated_prediction, target)
        shuffled_error = self._prediction_error(shuffled_prediction, target)
        utility_features = torch.stack(
            (
                base,
                base.abs(),
                centered_candidate_delta.detach(),
                centered_candidate_delta.detach().abs(),
                ordered_error.detach(),
                (reversed_error - ordered_error).detach(),
            ),
            dim=-1,
        )
        utility_logit = self.utility(utility_features).squeeze(-1)
        utility_probability = utility_logit.sigmoid()
        deployment_utility = torch.where(
            self.deployment_utility_inverted[None] > 0.5,
            1.0 - utility_probability,
            utility_probability,
        )
        deploy_gate = (
            (deployment_utility >= self.deployment_cutoff[None]).to(raw_delta.dtype)
            * self.deployment_label_gate[None]
            * availability
        )
        deploy_delta = (
            deploy_gate * self.deployment_scale[None] * centered_candidate_delta
        ).clamp(-self.cap, self.cap)
        return {
            "predicted_terminal_token": ordered,
            "reversed_predicted_terminal_token": reversed_prediction,
            "repeated_predicted_terminal_token": repeated_prediction,
            "shuffled_predicted_terminal_token": shuffled_prediction,
            "terminal_target_token": target,
            "candidate_delta": candidate_delta,
            "candidate_pre_tanh": raw_delta,
            "candidate_saturation": (raw_delta.abs() >= 2.0).to(raw_delta.dtype),
            "candidate_innovation_score": innovation_score,
            "candidate_motion_score": motion_score,
            "candidate_order_score": order_score,
            "candidate_logits": base + candidate_delta,
            "centered_candidate_delta": centered_candidate_delta,
            "centered_candidate_logits": base + centered_candidate_delta,
            "deploy_gate": deploy_gate,
            "deploy_delta": deploy_delta,
            "deploy_logits": base + deploy_delta,
            "utility_logit": utility_logit,
            "utility_probability": utility_probability,
            "deployment_utility_probability": deployment_utility,
            "ordered_prediction_error": ordered_error,
            "reversed_prediction_error": reversed_error,
            "repeated_prediction_error": repeated_error,
            "shuffled_prediction_error": shuffled_error,
            "history_available": available,
            "policy_source": self.deployment_policy_source,
        }
