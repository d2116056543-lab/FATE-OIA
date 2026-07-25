"""P8 action-semantic bridge with an explicit action-only formal projection."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
from torch import Tensor, nn


FORMAL_ACTION_COUNT = 4
FORMAL_REASON_COUNT = 21
FORMAL_DIM = 384
P16_OWNER_MATRIX_INTEGRATION_STATUS = "deferred_to_p16"
REQUIRED_P16_FIREWALL_CHECKS = (
    "P4 semantic_reason owner-gradient matrix on the assembled model for action, reason, and private losses",
    "P11 evidence-ledger owner-gradient matrix on the assembled model for action, reason, and private losses",
    "P16 must verify branch admission and firewall isolation using real P4/P11 owners rather than the P8 fixture",
)


@runtime_checkable
class ActionGlobalProjector(Protocol):
    """The narrow P7 surface that owns the only formal action projection."""

    def project_global(self, action_tokens: Tensor) -> Tensor:
        """Map post-bridge action tokens [B,4,384] to formal logits [B,4]."""


class RAELActionReasonBridge(nn.Module):
    """Bounded image-semantic cross-attention that remains action-safe at step zero."""

    parameter_owner = "action_reason_bridge"
    learning_rate = 2.0e-4

    def __init__(self, dim: int = FORMAL_DIM, num_heads: int = 6) -> None:
        super().__init__()
        if dim != FORMAL_DIM:
            raise ValueError(f"RAEL action-semantic bridge requires dim={FORMAL_DIM}, got {dim}")
        if num_heads <= 0 or dim % num_heads != 0:
            raise ValueError("num_heads must divide the formal action embedding dimension")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.dim,
            num_heads=self.num_heads,
            dropout=0.0,
            batch_first=True,
        )
        # The bridge starts as a strict identity while leaving a scalar route
        # that receives gradient on the first action backward pass.
        self.gamma_as_raw = nn.Parameter(torch.zeros(()))

    def owned_parameter_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.named_parameters())

    def _validate_inputs(self, action_visual_tokens: Tensor, semantic_reason_tokens: Tensor) -> None:
        if (
            not torch.is_tensor(action_visual_tokens)
            or action_visual_tokens.ndim != 3
            or action_visual_tokens.shape[1:] != (FORMAL_ACTION_COUNT, self.dim)
        ):
            raise ValueError("action_visual_tokens must be [B,4,384]")
        if (
            not torch.is_tensor(semantic_reason_tokens)
            or semantic_reason_tokens.ndim != 3
            or semantic_reason_tokens.shape[1:] != (FORMAL_REASON_COUNT, self.dim)
        ):
            raise ValueError("semantic_reason_tokens must be [B,21,384]")
        if action_visual_tokens.shape[0] <= 0 or semantic_reason_tokens.shape[0] != action_visual_tokens.shape[0]:
            raise ValueError("action_visual_tokens and semantic_reason_tokens must share a positive batch size")
        if action_visual_tokens.device != semantic_reason_tokens.device:
            raise ValueError("action_visual_tokens and semantic_reason_tokens must use the same device")
        if action_visual_tokens.dtype != semantic_reason_tokens.dtype:
            raise ValueError("action_visual_tokens and semantic_reason_tokens must use the same dtype")
        if not torch.is_floating_point(action_visual_tokens) or not torch.isfinite(action_visual_tokens).all():
            raise ValueError("action_visual_tokens must be finite floating point values")
        if not torch.is_floating_point(semantic_reason_tokens) or not torch.isfinite(semantic_reason_tokens).all():
            raise ValueError("semantic_reason_tokens must be finite floating point values")

    def forward(
        self,
        action_visual_tokens: Tensor,
        semantic_reason_tokens: Tensor,
        action_foundation: ActionGlobalProjector,
    ) -> dict[str, Tensor | dict[str, Tensor]]:
        """Return bridged action tokens and the sole P8 formal action-logit key."""

        self._validate_inputs(action_visual_tokens, semantic_reason_tokens)
        if not isinstance(action_foundation, ActionGlobalProjector):
            raise TypeError("action_foundation must provide the P7 project_global contract")
        if action_visual_tokens.dtype != self.gamma_as_raw.dtype:
            raise ValueError("bridge inputs must use the bridge parameter dtype")

        bridge_output, attention_weights = self.cross_attention(
            query=action_visual_tokens,
            key=semantic_reason_tokens,
            value=semantic_reason_tokens,
            need_weights=True,
            average_attn_weights=False,
        )
        gamma_as = 0.25 * torch.tanh(self.gamma_as_raw)
        action_bridged_tokens = action_visual_tokens + gamma_as * bridge_output
        z_a_global = action_foundation.project_global(action_bridged_tokens)
        if z_a_global.shape != (action_visual_tokens.shape[0], FORMAL_ACTION_COUNT):
            raise RuntimeError("P7 project_global must produce formal action logits [B,4]")
        if not torch.isfinite(action_bridged_tokens).all() or not torch.isfinite(z_a_global).all():
            raise FloatingPointError("P8 action-semantic bridge produced non-finite values")

        # Diagnostics must never retain a training graph when callers archive
        # them for hundreds of steps.  The formal tensors above stay attached.
        with torch.no_grad():
            diagnostic_attention = attention_weights.detach()
            attention_probabilities = diagnostic_attention.clamp_min(torch.finfo(diagnostic_attention.dtype).tiny)
            attention_entropy = -(attention_probabilities * attention_probabilities.log()).sum(dim=-1).mean(dim=1)
            bridge_delta = (action_bridged_tokens - action_visual_tokens).detach()
            diagnostics = {
                "bridge_rms": bridge_output.detach().square().mean(dim=(-2, -1)).sqrt(),
                "bridge_delta_rms": bridge_delta.square().mean(dim=(-2, -1)).sqrt(),
                "global_rms": z_a_global.detach().square().mean(dim=-1).sqrt(),
                "attention_entropy": attention_entropy,
                "attention_max": diagnostic_attention.amax(dim=-1).mean(dim=1),
            }
        return {
            "action_bridged_tokens": action_bridged_tokens,
            "z_A_global": z_a_global,
            "gamma_AS": gamma_as.detach(),
            "attention_weights": diagnostic_attention,
            "diagnostics": diagnostics,
        }


__all__ = [
    "ActionGlobalProjector",
    "P16_OWNER_MATRIX_INTEGRATION_STATUS",
    "RAELActionReasonBridge",
    "REQUIRED_P16_FIREWALL_CHECKS",
]
