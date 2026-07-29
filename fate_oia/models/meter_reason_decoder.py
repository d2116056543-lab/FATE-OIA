from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class METERPrivateReasonDecoder(nn.Module):
    """Global private reason predictor plus detached typed-evidence correction."""

    def __init__(self, dim: int = 384, reason_dim: int = 21, action_dim: int = 4) -> None:
        super().__init__()
        self.dim = int(dim)
        self.reason_dim = int(reason_dim)
        self.private_queries = nn.Parameter(torch.randn(reason_dim, dim) * 0.02)
        self.layer_router = nn.Parameter(torch.zeros(3))
        self.global_query = nn.Linear(dim, dim)
        self.global_key = nn.Linear(dim, dim)
        self.global_value = nn.Linear(dim, dim)
        self.global_norm = nn.LayerNorm(dim)
        self.reason_self_attention = nn.MultiheadAttention(
            dim, num_heads=4, batch_first=True
        )
        self.reason_self_norm = nn.LayerNorm(dim)
        self.global_head = nn.Linear(dim, 1)
        self.correction_vector = nn.Parameter(torch.randn(reason_dim, dim) * 0.02)
        self.correction_kappa_raw = nn.Parameter(
            torch.full((reason_dim,), -2.2521685)
        )

    def initialize_from_foundation(self, foundation: nn.Module) -> None:
        trunk = foundation.trunk
        with torch.no_grad():
            self.private_queries.copy_(trunk.label_queries[foundation.action_dim :])
            self.global_head.weight.copy_(trunk.logit_head.weight)
            self.global_head.bias.copy_(trunk.logit_head.bias)

    @staticmethod
    def _ramp(progress: float) -> float:
        return float(min(max(progress / 0.10, 0.0), 1.0))

    def forward(
        self,
        *,
        patch_tokens_by_layer: Tensor,
        reason_logits_calalign: Tensor,
        factor_typed_token: Tensor,
        factor_reliability: Tensor,
        factor_groundable_mask: Tensor,
        progress: float = 1.0,
    ) -> dict[str, Tensor]:
        detached_layers = patch_tokens_by_layer.detach()
        layer_weights = torch.softmax(self.layer_router, dim=0)
        patch = torch.einsum("s,bsnd->bnd", layer_weights, detached_layers)
        query = self.global_query(self.private_queries).unsqueeze(0).expand(
            patch.shape[0], -1, -1
        )
        key = self.global_key(patch)
        value = self.global_value(patch)
        attention = torch.softmax(
            torch.einsum("brd,bnd->brn", query, key) / math.sqrt(self.dim),
            dim=-1,
        )
        token = self.global_norm(torch.einsum("brn,bnd->brd", attention, value))
        context = self.reason_self_attention(token, token, token, need_weights=False)[0]
        token = self.reason_self_norm(token + context)
        learned_global = self.global_head(token).squeeze(-1)
        ramp = self._ramp(progress)
        global_logits = reason_logits_calalign.detach() + ramp * (
            learned_global - reason_logits_calalign.detach()
        )
        evidence = factor_typed_token.detach()
        reliability = factor_reliability.detach()
        groundable = factor_groundable_mask.to(evidence).view(1, -1)
        raw = torch.einsum("brd,rd->br", evidence, self.correction_vector)
        kappa = torch.nn.functional.softplus(self.correction_kappa_raw).clamp(
            0.02, 0.30
        )
        correction = (
            groundable
            * reliability
            * kappa.view(1, -1)
            * torch.tanh(raw / kappa.view(1, -1))
        )
        final = global_logits + ramp * correction
        return {
            "reason_global_tokens": token,
            "reason_global_attention": attention,
            "reason_logits_global": global_logits,
            "reason_evidence_delta": correction,
            "reason_logits_final": final,
            "reason_correction_kappa": kappa,
            "reason_groundable_mask": groundable.squeeze(0),
            "reason_layer_weights": layer_weights,
        }
