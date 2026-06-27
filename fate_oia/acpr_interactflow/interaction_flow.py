from __future__ import annotations

import time

import torch
from torch import nn
from fate_oia.models.acpr_sparse_ops import entmax15_bisect

from .interaction_grammar import InteractionGrammar
from .response_lag import ResponseLagEstimator
from .types import InteractionFlowState


class InteractionFlowReasoner(nn.Module):
    def __init__(self, grammar_path: str, dim: int = 384, num_predicates: int = 48, num_actions: int = 3) -> None:
        super().__init__()
        self.grammar = InteractionGrammar(grammar_path)
        self.num_factors = len(self.grammar.flow_factors)
        self.factor_queries = nn.Parameter(torch.randn(self.num_factors, dim) * 0.02)
        self.predicate_key = nn.Linear(dim, dim)
        self.factor_value = nn.Linear(dim, dim)
        self.edge_head = nn.Linear(dim, num_actions)
        self.factor_logit = nn.Linear(dim, 1)
        self.state_head = nn.Linear(dim, len(self.grammar.state_groups))
        self.lag = ResponseLagEstimator(dim=dim, max_lag=max(self.grammar.response_lags))

    def forward(
        self,
        predicate_tokens: torch.Tensor,
        predicate_probs: torch.Tensor,
        motion_token: torch.Tensor,
        predicate_corridor_mass: torch.Tensor | None = None,
        lag_disabled: bool = False,
        factor_mask: torch.Tensor | None = None,
    ) -> InteractionFlowState:
        if predicate_tokens.ndim == 3:
            predicate_tokens_t = predicate_tokens.unsqueeze(1).expand(-1, 15, -1, -1)
            predicate_probs_t = predicate_probs.unsqueeze(1).expand(-1, 15, -1)
        else:
            predicate_tokens_t = predicate_tokens
            predicate_probs_t = predicate_probs
        b, t, p, d = predicate_tokens_t.shape
        q = self.factor_queries.view(1, 1, self.num_factors, d)
        k = self.predicate_key(predicate_tokens_t)
        score = torch.einsum("btfd,btpd->btfp", q.expand(b, t, -1, -1), k) / (d ** 0.5)
        score = score + predicate_probs_t.clamp_min(1e-4).log().unsqueeze(2)
        attn = entmax15_bisect(score, dim=-1)
        factor_tokens_trajectory = torch.einsum("btfp,btpd->btfd", attn, self.factor_value(predicate_tokens_t))
        if predicate_corridor_mass is None:
            factor_to_corridor = factor_tokens_trajectory.new_zeros(b, t, self.num_factors, 4)
        else:
            factor_to_corridor = torch.einsum("btfp,btpc->btfc", attn, predicate_corridor_mass)
        factor_logits_trajectory = self.factor_logit(factor_tokens_trajectory).squeeze(-1)
        factor_probs_trajectory = torch.sigmoid(factor_logits_trajectory)
        lag_start = time.perf_counter()
        lag_weights, factor_tokens = self.lag(factor_tokens_trajectory, disabled=lag_disabled)
        response_lag_time = time.perf_counter() - lag_start
        if factor_mask is not None:
            mask = factor_mask.to(device=factor_tokens.device, dtype=factor_tokens.dtype).view(1, -1, 1)
            factor_tokens = factor_tokens * mask
            attn = attn * mask.view(1, 1, self.num_factors, 1)
            factor_tokens_trajectory = factor_tokens_trajectory * mask.view(1, 1, self.num_factors, 1)
        flow_edges = self.edge_head(factor_tokens)
        state_token = factor_tokens.mean(1)
        state_logits = self.state_head(state_token)
        stats = {
            "flow_factor_count": self.num_factors,
            "factor_attention_entropy": float((-(attn.clamp_min(1e-9).log() * attn).sum(-1)).mean().detach().cpu()),
            "lag_argmax_mean": float(lag_weights.argmax(-1).float().mean().detach().cpu()),
            "lag_disabled": bool(lag_disabled),
            "lag_context_norm": float(factor_tokens.norm(dim=-1).mean().detach().cpu()),
            "response_lag_time": float(response_lag_time),
        }
        return InteractionFlowState(
            state_tokens=state_token,
            state_logits=state_logits,
            state_attention=attn.mean(1),
            factor_tokens_trajectory=factor_tokens_trajectory,
            factor_logits_trajectory=factor_logits_trajectory,
            factor_probs_trajectory=factor_probs_trajectory,
            factor_to_predicate=attn,
            factor_to_corridor=factor_to_corridor,
            lag_weights=lag_weights,
            flow_edges=flow_edges,
            factor_tokens=factor_tokens,
            stats=stats,
        )
