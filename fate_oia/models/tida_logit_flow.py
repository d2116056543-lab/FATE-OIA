from __future__ import annotations

import torch
from torch import nn


class TIDALogitFlowReader(nn.Module):
    """Read temporal innovation in the frozen image head's own logit space."""

    def __init__(
        self,
        *,
        num_labels: int,
        hidden_dim: int = 64,
        cap: float = 0.05,
    ) -> None:
        super().__init__()
        if num_labels < 1 or hidden_dim < 4 or cap <= 0:
            raise ValueError("invalid logit-flow dimensions")
        self.num_labels = int(num_labels)
        self.cap = float(cap)
        self.label_embedding = nn.Parameter(torch.randn(num_labels, hidden_dim) * 0.02)
        self.feature_projection = nn.Sequential(
            nn.LayerNorm(10),
            nn.Linear(10, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.candidate_output = nn.Linear(hidden_dim, 1, bias=False)
        self.utility_output = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.candidate_output.weight)
        nn.init.zeros_(self.utility_output.weight)
        nn.init.zeros_(self.utility_output.bias)
        self.register_buffer("deployment_label_gate", torch.ones(num_labels))
        self.register_buffer("deployment_scale", torch.ones(num_labels))
        self.register_buffer("deployment_cutoff", torch.zeros(num_labels))
        self.register_buffer("deployment_center", torch.zeros(num_labels))
        self.register_buffer("deployment_use_utility", torch.ones(num_labels))
        self.register_buffer("deployment_policy_fitted", torch.tensor(False))
        self.deployment_source = "learned_soft_utility"

    @torch.no_grad()
    def set_deployment_policy(
        self,
        *,
        gate: torch.Tensor,
        scale: torch.Tensor,
        cutoff: torch.Tensor,
        center: torch.Tensor,
        use_utility: torch.Tensor,
        source: str,
    ) -> None:
        provenance = str(source).lower()
        if "train_calib" not in provenance or "test" in provenance or "oracle" in provenance:
            raise ValueError("logit-flow deployment policy must come from train_calib only")
        values = tuple(
            torch.as_tensor(value, dtype=self.deployment_scale.dtype, device=self.deployment_scale.device)
            for value in (gate, scale, cutoff, center, use_utility)
        )
        if any(value.shape != (self.num_labels,) for value in values):
            raise ValueError("logit-flow deployment policy must contain one value per label")
        if not all(torch.isfinite(value).all() for value in values):
            raise ValueError("logit-flow deployment policy must be finite")
        gate, scale, cutoff, center, use_utility = values
        self.deployment_label_gate.copy_(gate.clamp(0.0, 1.0))
        self.deployment_scale.copy_(scale)
        self.deployment_cutoff.copy_(cutoff.clamp(0.0, 1.0))
        self.deployment_center.copy_(center.clamp(-self.cap, self.cap))
        self.deployment_use_utility.copy_(use_utility.clamp(0.0, 1.0))
        self.deployment_policy_fitted.fill_(True)
        self.deployment_source = str(source)

    @staticmethod
    def _masked_mean(value: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
        weight = mask.to(value.dtype)
        return (value * weight).sum(dim) / weight.sum(dim).clamp_min(1.0)

    def _temporal_features(
        self,
        history_logits: torch.Tensor,
        terminal_logits: torch.Tensor,
        timestamps: torch.Tensor,
        frame_valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, frames, labels = history_logits.shape
        history_valid = frame_valid_mask[:, :frames]
        history_available = history_valid.any(1)
        mask = history_valid[:, :, None]
        history_time = timestamps[:, :frames]
        terminal_time = timestamps[:, -1]

        mean = self._masked_mean(history_logits, mask, 1)
        centered = (history_logits - mean[:, None]) * mask
        variance = self._masked_mean(centered.square(), mask, 1)
        std = variance.clamp_min(0.0).sqrt()

        time_mean = self._masked_mean(history_time, history_valid, 1)
        centered_time = (history_time - time_mean[:, None]) * history_valid
        time_variance = (centered_time.square()).sum(1).clamp_min(1e-6)
        slope = (
            centered * centered_time[:, :, None]
        ).sum(1) / time_variance[:, None]

        positions = torch.arange(frames, device=history_logits.device)
        last_index = positions[None].masked_fill(~history_valid, -1).amax(1).clamp_min(0)
        first_index = positions[None].masked_fill(~history_valid, frames).amin(1).clamp_max(frames - 1)
        gather = lambda index: torch.gather(
            history_logits, 1, index[:, None, None].expand(-1, 1, labels)
        ).squeeze(1)
        last = gather(last_index)
        first = gather(first_index)
        last_time = torch.gather(history_time, 1, last_index[:, None]).squeeze(1)
        first_time = torch.gather(history_time, 1, first_index[:, None]).squeeze(1)
        recent_slope = (terminal_logits - last) / (
            terminal_time - last_time
        ).abs().clamp_min(1e-3)[:, None]
        span_slope = (terminal_logits - first) / (
            terminal_time - first_time
        ).abs().clamp_min(1e-3)[:, None]
        acceleration = recent_slope - slope
        value_range = (
            history_logits.masked_fill(~mask, -torch.inf).amax(1)
            - history_logits.masked_fill(~mask, torch.inf).amin(1)
        )
        value_range = torch.where(history_available[:, None], value_range, torch.zeros_like(value_range))
        uncertainty = torch.exp(-terminal_logits.detach().abs())
        support = history_valid.float().mean(1)[:, None].expand(-1, labels)
        features = torch.stack(
            (
                terminal_logits.detach(),
                last - terminal_logits.detach(),
                mean - terminal_logits.detach(),
                slope,
                recent_slope,
                span_slope,
                acceleration,
                std,
                value_range,
                uncertainty * support,
            ),
            dim=-1,
        )
        features = torch.where(
            history_available[:, None, None], features, torch.zeros_like(features)
        )
        return features, history_available

    def forward(
        self,
        history_logits: torch.Tensor,
        terminal_logits: torch.Tensor,
        timestamps: torch.Tensor,
        frame_valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if history_logits.ndim != 3:
            raise ValueError("history_logits must be [B,T,L]")
        batch, frames, labels = history_logits.shape
        if labels != self.num_labels or terminal_logits.shape != (batch, labels):
            raise ValueError("terminal logits do not match logit-flow labels")
        if timestamps.shape != (batch, frames + 1) or frame_valid_mask.shape != timestamps.shape:
            raise ValueError("timestamps and frame_valid_mask must include history plus terminal")

        features, history_available = self._temporal_features(
            history_logits, terminal_logits, timestamps, frame_valid_mask
        )
        hidden = self.feature_projection(features) + self.label_embedding[None]
        candidate = self.cap * torch.tanh(self.candidate_output(hidden).squeeze(-1))
        candidate = candidate * history_available[:, None].to(candidate.dtype)
        utility_logit = self.utility_output(hidden).squeeze(-1)
        utility = torch.sigmoid(utility_logit) * history_available[:, None].to(candidate.dtype)
        availability = history_available[:, None].to(candidate.dtype)
        centered_candidate = (
            candidate - self.deployment_center[None]
        ) * availability
        if bool(self.deployment_policy_fitted):
            utility_selected = utility >= self.deployment_cutoff[None]
            selected = torch.where(
                self.deployment_use_utility[None] > 0.5,
                utility_selected,
                torch.ones_like(utility_selected),
            )
            deploy_delta = (
                self.deployment_label_gate[None]
                * self.deployment_scale[None]
                * selected.to(candidate.dtype)
                * centered_candidate
            ).clamp(-self.cap, self.cap)
        else:
            deploy_delta = utility * candidate
        return {
            "temporal_features": features,
            "history_available": history_available,
            "candidate_delta": candidate,
            "candidate_logits": terminal_logits + candidate,
            "utility_logit": utility_logit,
            "utility_probability": utility,
            "centered_candidate_delta": centered_candidate,
            "deploy_delta": deploy_delta,
            "deploy_logits": terminal_logits + deploy_delta,
            "deployment_label_gate": self.deployment_label_gate,
            "deployment_scale": self.deployment_scale,
            "deployment_cutoff": self.deployment_cutoff,
            "deployment_use_utility": self.deployment_use_utility,
            "deployment_center": self.deployment_center,
            "deployment_source": self.deployment_source,
        }
