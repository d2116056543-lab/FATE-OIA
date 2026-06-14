from __future__ import annotations

import torch
from torch import nn

from .acpr_reason_grammar import ACPRReasonGrammar


class ACPRPredicateReasoner(nn.Module):
    def __init__(self, dim: int = 384, reason_dim: int = 21, num_predicates: int = 32, predicate_names: list[str] | None = None, grammar_path: str = "configs/acpr_reason_predicate_grammar.yaml") -> None:
        super().__init__()
        self.reason_dim = reason_dim
        self.num_predicates = num_predicates
        self.mlp = nn.Sequential(nn.Linear(dim + 3, dim), nn.GELU(), nn.Linear(dim, 1))
        self.gate = nn.Parameter(torch.full((reason_dim,), -2.944))
        grammar = ACPRReasonGrammar(grammar_path)
        names = predicate_names or [str(i) for i in range(num_predicates)]
        pos, neg = grammar.reason_predicate_matrix(names)
        self.register_buffer("positive_mask", torch.tensor(pos, dtype=torch.float32), persistent=False)
        self.register_buffer("contradictory_mask", torch.tensor(neg, dtype=torch.float32), persistent=False)

    def forward(self, reason_nodes: torch.Tensor, predicate_probs: torch.Tensor, predicate_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        pos_count = self.positive_mask.sum(-1).clamp_min(1.0).view(1, -1)
        neg_count = self.contradictory_mask.sum(-1).clamp_min(1.0).view(1, -1)
        pos_score = predicate_probs @ self.positive_mask.t().to(predicate_probs.device, predicate_probs.dtype) / pos_count
        neg_score = predicate_probs @ self.contradictory_mask.t().to(predicate_probs.device, predicate_probs.dtype) / neg_count
        grammar_score = (pos_score - neg_score).clamp(-1.0, 1.0)
        contradiction = torch.sigmoid(neg_score - pos_score)
        aux = torch.stack([pos_score, neg_score, grammar_score], dim=-1)
        mlp_score = self.mlp(torch.cat([reason_nodes, aux], dim=-1)).squeeze(-1).tanh()
        gate = torch.sigmoid(self.gate).clamp(max=0.20).view(1, -1)
        delta = (gate * (grammar_score + 0.25 * mlp_score)).clamp(-0.20, 0.20)
        stats = {
            "required_support_mean": float(pos_score.detach().mean().cpu()),
            "contradiction_mean": float(contradiction.detach().mean().cpu()),
            "delta_abs_mean": float(delta.detach().abs().mean().cpu()),
            "gate_mean": float(gate.detach().mean().cpu()),
        }
        return {
            "predicate_reason_delta": delta,
            "required_support_score": pos_score,
            "contradiction_score": contradiction,
            "predicate_reason_gate": gate.expand_as(delta),
            "predicate_reason_positive_score_by_label": pos_score,
            "predicate_reason_contradiction_score_by_label": contradiction,
            "predicate_reason_delta_by_label": delta,
            "predicate_reason_stats": stats,
        }
