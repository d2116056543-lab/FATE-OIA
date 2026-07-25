"""P11 action-safe reason-private adapter for RAEL-OIA.

This module is deliberately one-way: final action outputs provide a fully
detached context to reasons, while the private reason path never returns an
action value or action gradient.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


FORMAL_ACTION_COUNT = 4
FORMAL_REASON_COUNT = 21
FORMAL_DIM = 384
PRIVATE_RANK = 64
PRIVATE_NORM_CAP = 0.40
# A 0.5% headroom is small relative to the formal 0.40 budget, while leaving
# enough room for BF16 rounding before the final FP32 audit/reprojection.
PRIVATE_NORM_SAFETY = 0.995
GAMMA_RA_CAP = 0.25


class ReasonGlobalHead(nn.Module):
    """Action-wise reason head that preserves the formal [B,21] shape."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(FORMAL_REASON_COUNT, dim))
        self.bias = nn.Parameter(torch.zeros(FORMAL_REASON_COUNT))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, reason_tokens: Tensor) -> Tensor:
        if reason_tokens.ndim != 3 or reason_tokens.shape[1:] != (FORMAL_REASON_COUNT, self.weight.shape[1]):
            raise ValueError("reason global head requires formal reason tokens [B,21,384]")
        return torch.einsum("brd,rd->br", reason_tokens, self.weight) + self.bias


class RAELReasonPrivateAdapter(nn.Module):
    """Benchmark-private reason adaptation with an action-gradient firewall.

    ``action_context`` is constructed from final action outputs and then fully
    detached before it reaches any P11 parameter.  In contrast, semantic
    reason tokens remain attached so the future P13 admission mechanism can
    explicitly control that shared boundary.
    """

    parameter_owner = "reason_private"
    learning_rate = 3.0e-4

    def __init__(self, dim: int = FORMAL_DIM, rank: int = PRIVATE_RANK) -> None:
        super().__init__()
        if dim != FORMAL_DIM:
            raise ValueError(f"RAEL reason-private adapter requires dim={FORMAL_DIM}, got {dim}")
        if rank != PRIVATE_RANK:
            raise ValueError(f"RAEL reason-private adapter requires rank={PRIVATE_RANK}, got {rank}")
        self.dim = int(dim)
        self.rank = int(rank)
        self.private_down = nn.Linear(self.dim, self.rank)
        self.private_up = nn.Linear(self.rank, self.dim)
        self.action_context_projection = nn.Linear(self.dim, self.dim)
        self.gamma_ra_raw = nn.Parameter(torch.zeros(()))
        self.reason_global_head = ReasonGlobalHead(self.dim)

    def owned_parameter_names(self) -> tuple[str, ...]:
        """Expose the private optimizer/admission ownership boundary."""

        return tuple(name for name, _ in self.named_parameters())

    def _validate_inputs(
        self,
        semantic_reason_tokens: Tensor,
        action_bridged_tokens: Tensor,
        final_action_logits: Tensor,
    ) -> None:
        if (
            not torch.is_tensor(semantic_reason_tokens)
            or semantic_reason_tokens.ndim != 3
            or semantic_reason_tokens.shape[1:] != (FORMAL_REASON_COUNT, self.dim)
        ):
            raise ValueError("semantic_reason_tokens must be [B,21,384]")
        if (
            not torch.is_tensor(action_bridged_tokens)
            or action_bridged_tokens.ndim != 3
            or action_bridged_tokens.shape[1:] != (FORMAL_ACTION_COUNT, self.dim)
        ):
            raise ValueError("action_bridged_tokens must be [B,4,384]")
        if (
            not torch.is_tensor(final_action_logits)
            or final_action_logits.ndim != 2
            or final_action_logits.shape[1:] != (FORMAL_ACTION_COUNT,)
        ):
            raise ValueError("final_action_logits must be [B,4]")
        batch_size = semantic_reason_tokens.shape[0]
        if batch_size <= 0 or action_bridged_tokens.shape[0] != batch_size or final_action_logits.shape[0] != batch_size:
            raise ValueError("P11 inputs must share a positive batch size")
        values = (semantic_reason_tokens, action_bridged_tokens, final_action_logits)
        if any(not torch.is_floating_point(value) or not bool(torch.isfinite(value).all()) for value in values):
            raise ValueError("P11 inputs must be finite floating point tensors")
        if any(value.device != semantic_reason_tokens.device for value in values[1:]):
            raise ValueError("P11 inputs must share one device")
        if any(value.dtype != semantic_reason_tokens.dtype for value in values[1:]):
            raise ValueError("P11 inputs must share one dtype")
        parameter_dtype = self.private_down.weight.dtype
        if semantic_reason_tokens.dtype != parameter_dtype:
            raise ValueError("P11 inputs must use the adapter parameter dtype")

    @staticmethod
    def _cap_private_delta(private_raw: Tensor, semantic_reason_tokens: Tensor) -> Tensor:
        """Cap P in FP32, then re-audit after returning to the model dtype.

        BF16 rounding can move a vector just above its intended norm bound.  A
        small safety margin plus a second FP32 measurement/reprojection keeps
        the tensor that downstream consumers actually receive inside the hard
        0.40 budget.  Zero semantic rows are explicitly defined as P=0.
        """

        output_dtype = private_raw.dtype
        raw_fp32 = private_raw.float()
        semantic_fp32 = semantic_reason_tokens.float()
        raw_norm = raw_fp32.norm(dim=-1, keepdim=True)
        semantic_norm = semantic_fp32.norm(dim=-1, keepdim=True)
        nonzero_semantic = semantic_norm > 0.0
        safe_cap = PRIVATE_NORM_CAP * PRIVATE_NORM_SAFETY * semantic_norm
        first_scale = torch.minimum(
            torch.ones_like(raw_norm),
            safe_cap / raw_norm.clamp_min(torch.finfo(torch.float32).tiny),
        )
        first_scale = torch.where(nonzero_semantic, first_scale, torch.zeros_like(first_scale))
        provisional = (raw_fp32 * first_scale).to(dtype=output_dtype)

        # Re-measure the cast tensor in FP32.  This is deliberately not an
        # estimate from the pre-cast values because the exported BF16 tensor is
        # the formal P11 output and must satisfy the contract by itself.
        provisional_norm = provisional.float().norm(dim=-1, keepdim=True)
        second_scale = torch.minimum(
            torch.ones_like(provisional_norm),
            safe_cap / provisional_norm.clamp_min(torch.finfo(torch.float32).tiny),
        )
        second_scale = torch.where(nonzero_semantic, second_scale, torch.zeros_like(second_scale))
        private_delta = (provisional.float() * second_scale).to(dtype=output_dtype)
        return torch.where(nonzero_semantic.to(dtype=torch.bool), private_delta, torch.zeros_like(private_delta))

    def _detached_action_context(self, action_bridged_tokens: Tensor, final_action_logits: Tensor) -> Tensor:
        # The detach belongs after the exact sigmoid-weighted context equation.
        action_context = torch.einsum(
            "ba,bad->bd",
            torch.sigmoid(final_action_logits),
            action_bridged_tokens,
        )
        return action_context.detach()

    def forward(
        self,
        semantic_reason_tokens: Tensor,
        action_bridged_tokens: Tensor,
        final_action_logits: Tensor,
    ) -> dict[str, Tensor | dict[str, Tensor]]:
        """Return formal reason tokens/logits without exposing an action path."""

        self._validate_inputs(semantic_reason_tokens, action_bridged_tokens, final_action_logits)
        action_context = self._detached_action_context(action_bridged_tokens, final_action_logits)
        projected_action_context = self.action_context_projection(action_context).unsqueeze(1)
        gamma_ra = GAMMA_RA_CAP * torch.tanh(self.gamma_ra_raw)

        private_raw = self.private_up(F.gelu(self.private_down(semantic_reason_tokens)))
        private_delta = self._cap_private_delta(private_raw, semantic_reason_tokens)
        formal_reason_tokens = semantic_reason_tokens + gamma_ra * projected_action_context + private_delta
        z_r_global = self.reason_global_head(formal_reason_tokens)
        if not bool(torch.isfinite(formal_reason_tokens).all()) or not bool(torch.isfinite(z_r_global).all()):
            raise FloatingPointError("P11 reason-private adapter produced non-finite values")

        with torch.no_grad():
            semantic_norm = semantic_reason_tokens.detach().float().norm(dim=-1)
            private_norm = private_delta.detach().float().norm(dim=-1)
            private_norm_ratio = torch.where(
                semantic_norm > 0.0,
                private_norm / semantic_norm,
                torch.zeros_like(private_norm),
            )
            diagnostics = {
                "private_norm_ratio": private_norm_ratio,
                "gamma_RA": gamma_ra.detach().reshape(1),
                "h_A_norm": action_context.norm(dim=-1).detach(),
                "action_context_norm": projected_action_context.detach().norm(dim=-1).squeeze(1),
            }
        return {
            "formal_reason_tokens": formal_reason_tokens,
            "z_R_global": z_r_global,
            "private_delta": private_delta,
            "action_context": action_context,
            "diagnostics": diagnostics,
        }


__all__ = [
    "FORMAL_ACTION_COUNT",
    "FORMAL_DIM",
    "FORMAL_REASON_COUNT",
    "GAMMA_RA_CAP",
    "PRIVATE_NORM_CAP",
    "PRIVATE_NORM_SAFETY",
    "PRIVATE_RANK",
    "RAELReasonPrivateAdapter",
    "ReasonGlobalHead",
]
