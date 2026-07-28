from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class METERPrivateReasonDecoder(nn.Module):
    """Private reason decoder that reads detached action/factor context only."""

    def __init__(self, dim: int = 384, reason_dim: int = 21, action_dim: int = 4) -> None:
        super().__init__()
        self.dim = int(dim)
        self.reason_dim = int(reason_dim)
        self.action_dim = int(action_dim)
        self.private_queries = nn.Parameter(torch.randn(reason_dim, dim) * 0.02)
        self.global_query = nn.Linear(dim, dim)
        self.global_key = nn.Linear(dim, dim)
        self.global_value = nn.Linear(dim, dim)
        self.global_norm = nn.LayerNorm(dim)
        self.local_proj = nn.Linear(dim, dim)
        self.local_norm = nn.LayerNorm(dim)
        self.factor_proj = nn.Linear(dim, dim)
        self.action_proj = nn.Linear(dim, dim)
        self.global_head = nn.Linear(dim, 1)
        self.local_head = nn.Linear(dim, 1)
        self.mix_gate = nn.Sequential(nn.Linear(dim * 2 + 5, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1))
        self.annotation_head = nn.Sequential(nn.Linear(dim * 2 + 2, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1))
        self.tail_gain = nn.Parameter(torch.zeros(reason_dim))

    def initialize_from_foundation(self, foundation: nn.Module) -> None:
        trunk = foundation.trunk
        with torch.no_grad():
            self.private_queries.copy_(trunk.label_queries[foundation.action_dim :])
            self.global_query.weight.copy_(trunk.query_proj.weight)
            self.global_query.bias.copy_(trunk.query_proj.bias)
            self.global_key.weight.copy_(trunk.key_proj.weight)
            self.global_key.bias.copy_(trunk.key_proj.bias)
            self.global_value.weight.copy_(trunk.value_proj.weight)
            self.global_value.bias.copy_(trunk.value_proj.bias)
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
        action_logits_final: Tensor,
        action_nodes: Tensor,
        factor_to_reason_tokens: Tensor,
        factor_support_map: Tensor,
        factor_counter_map: Tensor,
        factor_reliability: Tensor,
        factor_support_null: Tensor,
        progress: float = 1.0,
    ) -> dict[str, Tensor]:
        # Private reason views must not backpropagate into the foundation.
        patch = patch_tokens_by_layer.detach().mean(dim=1)
        query = self.global_query(self.private_queries).view(1, self.reason_dim, self.dim)
        key = self.global_key(patch)
        value = self.global_value(patch)
        global_attention = torch.softmax(torch.einsum("brd,bnd->brn", query, key) / math.sqrt(self.dim), dim=-1)
        global_token = self.global_norm(torch.einsum("brn,bnd->brd", global_attention, value))
        action_context = torch.einsum("ba,bad->bd", torch.sigmoid(action_logits_final).detach(), action_nodes.detach())
        factor_context = self.factor_proj(factor_to_reason_tokens)
        local_maps = (factor_support_map - factor_counter_map).detach()
        local_detail = torch.einsum("brn,bsnd->brsd", local_maps, patch_tokens_by_layer).mean(dim=2)
        tail = self.tail_gain.view(1, -1, 1) * (factor_support_map.detach().mean(-1) - factor_counter_map.detach().mean(-1)).unsqueeze(-1)
        local_token = self.local_norm(self.local_proj(local_detail) + factor_context + tail)
        action_term = self.action_proj(action_context).unsqueeze(1).expand(-1, self.reason_dim, -1)
        global_token = self.global_norm(global_token + action_term)
        local_token = self.local_norm(local_token + action_term)
        logits_global = self.global_head(global_token).squeeze(-1)
        logits_local = self.local_head(local_token).squeeze(-1)
        disagreement = (logits_global - logits_local).abs().unsqueeze(-1)
        gate_features = torch.cat(
            [
                global_token,
                local_token,
                factor_reliability.detach().unsqueeze(-1),
                factor_support_null.detach().unsqueeze(-1),
                disagreement,
                torch.sigmoid(action_logits_final).detach().mean(dim=-1, keepdim=True).unsqueeze(1).expand(-1, self.reason_dim, -1),
                local_maps.abs().mean(-1, keepdim=True),
            ],
            dim=-1,
        )
        mix_gate = torch.sigmoid(self.mix_gate(gate_features).squeeze(-1))
        logits_mix = mix_gate * logits_global + (1.0 - mix_gate) * logits_local
        annotation_features = torch.cat([global_token, local_token, factor_reliability.detach().unsqueeze(-1), disagreement], dim=-1)
        annotation_delta = 0.5 * torch.tanh(self.annotation_head(annotation_features).squeeze(-1))
        candidate = logits_mix + annotation_delta
        base_reason = reason_logits_calalign.detach()
        final = base_reason + self._ramp(progress) * (candidate - base_reason)
        return {
            "reason_global_tokens": global_token,
            "reason_local_tokens": local_token,
            "reason_logits_global": logits_global,
            "reason_logits_local": logits_local,
            "reason_logits_mix": logits_mix,
            "reason_annotation_delta": annotation_delta,
            "reason_logits_candidate": candidate,
            "reason_logits_final": final,
            "reason_mix_gate": mix_gate,
            "reason_tail_sensitivity": tail.squeeze(-1).abs(),
        }
