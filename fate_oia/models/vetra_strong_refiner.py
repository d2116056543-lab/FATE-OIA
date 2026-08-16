from __future__ import annotations

import copy

import torch
from torch import Tensor, nn


class SelectiveVisualActionRankRefiner(nn.Module):
    """Bounded action-only residual over frozen visual action evidence."""

    def __init__(
        self,
        dim: int = 384,
        rank: int = 64,
        action_dim: int = 4,
        max_delta: float = 0.12,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.max_delta = float(max_delta)
        self.norm = nn.LayerNorm(2 * dim)
        self.input_projection = nn.Linear(2 * dim, rank)
        self.output_weight = nn.Parameter(torch.zeros(action_dim, rank))
        self.register_buffer("deployment_gain", torch.ones(action_dim))

    def set_deployment_gain(self, gain: Tensor) -> None:
        gain = gain.detach().to(self.deployment_gain).flatten()
        if gain.shape != self.deployment_gain.shape:
            raise ValueError(f"expected gain shape {tuple(self.deployment_gain.shape)}, got {tuple(gain.shape)}")
        self.deployment_gain.copy_(gain.clamp(0.0, 1.0))

    def forward(
        self,
        action_logits_base: Tensor,
        reason_logits_base: Tensor,
        action_nodes: Tensor,
        evidence_tokens: Tensor,
        gain: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if evidence_tokens.ndim == 4:
            evidence_tokens = evidence_tokens.mean(dim=2)
        if action_nodes.shape != evidence_tokens.shape:
            raise ValueError(
                f"action/evidence shape mismatch: {tuple(action_nodes.shape)} vs {tuple(evidence_tokens.shape)}"
            )
        # Keep the small ranking residual in FP32. Under BF16, deltas below the
        # base-logit ULP can disappear in the forward pass while retaining a
        # misleading straight-through gradient.
        features = torch.cat((action_nodes.detach(), evidence_tokens.detach()), dim=-1).float()
        with torch.autocast(device_type=features.device.type, enabled=False):
            hidden = torch.nn.functional.gelu(self.input_projection(self.norm(features)))
            raw_delta = torch.einsum("bar,ar->ba", hidden, self.output_weight)
            bounded = self.max_delta * torch.tanh(raw_delta)
            effective_gain = self.deployment_gain if gain is None else gain
            delta = bounded * effective_gain.to(bounded).view(1, self.action_dim)
            final = action_logits_base.detach().float() + delta
        return {
            "action_logits_base": action_logits_base,
            "action_logits_final": final,
            "reason_logits_final": reason_logits_base,
            "action_delta": delta,
            "action_delta_unscaled": bounded,
            "deployment_gain": effective_gain,
        }


class SelectiveActionPathRefiner(nn.Module):
    """Clone and fine-tune only the source action evidence/contribution path."""

    def __init__(self, action_evidence: nn.Module, action_contribution: nn.Module,
                 action_dim: int = 4, max_delta: float = 0.12) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.max_delta = float(max_delta)
        self.action_evidence = copy.deepcopy(action_evidence)
        self.action_contribution = copy.deepcopy(action_contribution)
        for parameter in self.parameters():
            parameter.requires_grad_(True)
        self.register_buffer("deployment_gain", torch.ones(action_dim))

    def set_deployment_gain(self, gain: Tensor) -> None:
        gain = gain.detach().to(self.deployment_gain).flatten()
        if gain.shape != self.deployment_gain.shape:
            raise ValueError(f"expected gain shape {tuple(self.deployment_gain.shape)}, got {tuple(gain.shape)}")
        self.deployment_gain.copy_(gain.clamp(0.0, 1.0))

    def forward(self, source: dict[str, Tensor], action_scale: float, gain: Tensor | None = None) -> dict[str, Tensor]:
        evidence = self.action_evidence(
            source["action_nodes_primary"].detach(),
            source["patch_tokens_by_layer_raw"].detach(),
            source["predicate_attention"].detach(),
            source["predicate_probs"].detach(),
            predicate_bias_enabled=True,
            local_reread_enabled=True,
        )
        contribution = self.action_contribution(
            evidence["evidence_token"], source["action_logits_primary"].detach(), action_scale=action_scale
        )
        base = source["action_logits_final"].detach()
        raw_delta = contribution["action_logits_final"] - base
        bounded = self.max_delta * torch.tanh(raw_delta / self.max_delta)
        effective_gain = self.deployment_gain if gain is None else gain
        delta = bounded * effective_gain.to(bounded).view(1, self.action_dim)
        return {
            "action_logits_base": base,
            "action_logits_final": base + delta,
            "reason_logits_final": source["reason_logits_final"],
            "action_delta": delta,
            "action_delta_unscaled": bounded,
            "deployment_gain": effective_gain,
            "refined_evidence_token": evidence["evidence_token"],
        }
