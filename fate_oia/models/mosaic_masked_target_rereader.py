from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .acpr_sparse_ops import entmax15_bisect


class MOSAICMaskedTargetRereader(nn.Module):
    """Re-read action-lane visual tokens through target-owned support/veto masks."""

    def __init__(
        self,
        *,
        dim: int = 384,
        action_count: int = 4,
        topk: int = 128,
        gate_init: float = 0.02,
        gate_max: float = 0.15,
    ) -> None:
        super().__init__()
        if action_count != 4 or not 0.0 < gate_init < gate_max <= 0.25:
            raise ValueError("IC-DOR rereader requires four actions and bounded route gates")
        self.dim = dim
        self.action_count = action_count
        self.topk = int(topk)
        self.key_proj = nn.Linear(dim, dim, bias=False)
        self.value_proj = nn.Linear(dim, dim, bias=False)
        self.support_query = nn.Linear(dim, dim, bias=False)
        self.veto_query = nn.Linear(dim, dim, bias=False)
        self.support_norm = nn.LayerNorm(dim)
        self.veto_norm = nn.LayerNorm(dim)
        self.support_head = nn.Linear(dim, 1, bias=False)
        self.veto_head = nn.Linear(dim, 1, bias=False)
        ratio = gate_init / gate_max
        initial_raw = math.log(ratio / (1.0 - ratio))
        self.support_gate_raw = nn.Parameter(torch.full((action_count,), initial_raw))
        self.veto_gate_raw = nn.Parameter(torch.full((action_count,), initial_raw))
        self.gate_max = float(gate_max)
        self.register_buffer("active_gate_cap", torch.tensor(float(gate_init)), persistent=True)

    @torch.no_grad()
    def set_gate_cap(self, cap: float) -> None:
        if not 0.0 < float(cap) <= self.gate_max:
            raise ValueError("IC-DOR active route gate cap must be in (0, gate_max]")
        self.active_gate_cap.fill_(float(cap))

    def _read(
        self,
        feature_map: torch.Tensor,
        action_queries: torch.Tensor,
        target_mask: torch.Tensor,
        query_proj: nn.Linear,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, dim, height, width = feature_map.shape
        tokens = feature_map.flatten(2).transpose(1, 2)
        keys = F.normalize(self.key_proj(tokens), dim=-1, eps=1e-6)
        queries = F.normalize(query_proj(action_queries), dim=-1, eps=1e-6)
        scores = torch.einsum("bad,bnd->ban", queries, keys) / math.sqrt(dim)
        flattened_mask = target_mask.flatten(2).clamp_min(0.0)
        if flattened_mask.shape != scores.shape:
            raise ValueError("IC-DOR rereader factor mask shape must match action-lane tokens")
        active = flattened_mask.sum(dim=-1) > 1e-8
        masked_scores = scores + flattened_mask.clamp_min(1e-8).log()
        count = min(self.topk, tokens.shape[1])
        top_scores, top_indices = masked_scores.topk(count, dim=-1)
        weights = entmax15_bisect(top_scores, dim=-1)
        gathered_mask = flattened_mask.gather(2, top_indices)
        weights = weights * (gathered_mask > 0).to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        values = self.value_proj(tokens)
        gathered_values = values.unsqueeze(1).expand(-1, self.action_count, -1, -1).gather(
            2, top_indices.unsqueeze(-1).expand(-1, -1, -1, dim)
        )
        nodes = torch.einsum("bak,bakd->bad", weights, gathered_values)
        nodes = nodes * active.unsqueeze(-1).to(nodes.dtype)
        full_attention = weights.new_zeros(scores.shape)
        full_attention.scatter_(2, top_indices, weights)
        return nodes, full_attention.reshape(batch_size, self.action_count, height, width), active

    def forward(
        self,
        action_pyramid: dict[str, torch.Tensor],
        action_queries: torch.Tensor,
        factor_masks: torch.Tensor,
        support_weights: torch.Tensor,
        veto_weights: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        high = action_pyramid.get("F_hi") if isinstance(action_pyramid, dict) else None
        if high is None or high.ndim != 4 or high.shape[1] != self.dim or tuple(high.shape[-2:]) != (45, 80):
            raise ValueError("IC-DOR rereader requires action F_hi [B,D,45,80]")
        if action_queries.shape != (high.shape[0], self.action_count, self.dim):
            raise ValueError("IC-DOR rereader action queries must be [B,4,D]")
        expected_weights = (high.shape[0], factor_masks.shape[1], self.action_count)
        if factor_masks.shape[:1] != high.shape[:1] or tuple(factor_masks.shape[-2:]) != (45, 80):
            raise ValueError("IC-DOR rereader factor masks must be [B,F,45,80]")
        if support_weights.shape != expected_weights or veto_weights.shape != expected_weights:
            raise ValueError("IC-DOR rereader route weights must be [B,F,4]")
        support_mask = torch.einsum("bfa,bfhw->bahw", support_weights.detach(), factor_masks.detach())
        veto_mask = torch.einsum("bfa,bfhw->bahw", veto_weights.detach(), factor_masks.detach())
        support_nodes, support_attention, support_active = self._read(high.detach(), action_queries.detach(), support_mask, self.support_query)
        veto_nodes, veto_attention, veto_active = self._read(high.detach(), action_queries.detach(), veto_mask, self.veto_query)
        support_gate = (self.gate_max * torch.sigmoid(self.support_gate_raw)).clamp_max(self.active_gate_cap).view(1, -1)
        veto_gate = (self.gate_max * torch.sigmoid(self.veto_gate_raw)).clamp_max(self.active_gate_cap).view(1, -1)
        support_logits = support_gate * F.softplus(self.support_head(self.support_norm(support_nodes)).squeeze(-1))
        veto_logits = veto_gate * F.softplus(self.veto_head(self.veto_norm(veto_nodes)).squeeze(-1))
        support_logits = support_logits * support_active.to(support_logits.dtype)
        veto_logits = veto_logits * veto_active.to(veto_logits.dtype)
        return {
            "action_support_nodes": support_nodes,
            "action_veto_nodes": veto_nodes,
            "action_support_attention": support_attention,
            "action_veto_attention": veto_attention,
            "action_support_mask": support_mask,
            "action_veto_mask": veto_mask,
            "action_support_logits": support_logits,
            "action_veto_logits": veto_logits,
            "action_route_strength": (support_logits + veto_logits).detach(),
            "action_support_gate": support_gate.expand(high.shape[0], -1),
            "action_veto_gate": veto_gate.expand(high.shape[0], -1),
            "action_route_gate_cap": self.active_gate_cap.clone(),
        }
