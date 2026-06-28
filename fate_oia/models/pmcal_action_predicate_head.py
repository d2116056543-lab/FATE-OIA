from __future__ import annotations

import torch
from torch import nn


class PMCalActionPredicateHead(nn.Module):
    def __init__(self, dim: int = 384, action_dim: int = 4, cap: float = 0.06, gate_max: float = 0.35) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.cap = float(cap)
        self.gate_max = float(gate_max)
        self.action_visual_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1))
        self.pred_proj = nn.Sequential(nn.LayerNorm(dim + 2), nn.Linear(dim + 2, dim), nn.GELU())
        self.action_predicate_head = nn.Linear(dim, action_dim)
        self.gate_head = nn.Sequential(nn.Linear(dim * 2, action_dim), nn.Sigmoid())

    def forward(
        self,
        action_visual_nodes: torch.Tensor,
        q_pred: torch.Tensor,
        rho_pred: torch.Tensor,
        predicate_tokens: torch.Tensor,
    ) -> dict[str, torch.Tensor | dict]:
        action_logits_visual = self.action_visual_head(action_visual_nodes).squeeze(-1)
        pred_features = torch.cat([predicate_tokens, q_pred.unsqueeze(-1), rho_pred.unsqueeze(-1)], dim=-1)
        pred_context = self.pred_proj(pred_features).mean(1)
        action_context = action_visual_nodes.mean(1)
        predicate_delta = self.action_predicate_head(pred_context).clamp(-self.cap, self.cap)
        gate = self.gate_head(torch.cat([action_context, pred_context], dim=-1)).clamp(0.0, self.gate_max)
        action_logits_final = action_logits_visual + gate * predicate_delta
        return {
            "action_logits_visual": action_logits_visual,
            "action_logits_predicate": predicate_delta,
            "action_predicate_gate": gate,
            "action_logits_final": action_logits_final,
            "action_predicate_stats": {
                "gate_mean": float(gate.detach().mean().cpu()),
                "delta_abs_mean": float(predicate_delta.detach().abs().mean().cpu()),
            },
        }
