from __future__ import annotations

import torch
from torch import nn
from fate_oia.models.acpr_sparse_ops import entmax15_bisect

from .interaction_grammar import InteractionGrammar
from .response_lag import ResponseLagEstimator
from .types import InteractionFlowState


class InteractionFlowReasoner(nn.Module):
    def __init__(self, grammar_path: str, dim: int = 384, num_predicates: int = 48, num_actions: int = 4) -> None:
        super().__init__()
        self.grammar = InteractionGrammar(grammar_path)
        self.num_factors = len(self.grammar.flow_factors)
        self.factor_queries = nn.Parameter(torch.randn(self.num_factors, dim) * 0.02)
        self.predicate_key = nn.Linear(dim, dim)
        self.factor_value = nn.Linear(dim, dim)
        self.edge_head = nn.Linear(dim, num_actions)
        self.state_head = nn.Linear(dim, len(self.grammar.state_groups))
        self.lag = ResponseLagEstimator(dim=dim, max_lag=max(self.grammar.response_lags))

    def forward(
        self,
        predicate_tokens: torch.Tensor,
        predicate_probs: torch.Tensor,
        motion_token: torch.Tensor,
        lag_disabled: bool = False,
        factor_mask: torch.Tensor | None = None,
    ) -> InteractionFlowState:
        d = predicate_tokens.shape[-1]
        q = self.factor_queries.view(1, self.num_factors, d)
        k = self.predicate_key(predicate_tokens)
        score = torch.einsum("bfd,bpd->bfp", q, k) / (d ** 0.5)
        score = score + predicate_probs.clamp_min(1e-4).log().unsqueeze(1)
        attn = entmax15_bisect(score, dim=-1)
        factor_tokens = torch.einsum("bfp,bpd->bfd", attn, self.factor_value(predicate_tokens))
        lag_weights, lag_context = self.lag(motion_token, disabled=lag_disabled)
        factor_tokens = factor_tokens + 0.1 * lag_context.unsqueeze(1)
        if factor_mask is not None:
            mask = factor_mask.to(device=factor_tokens.device, dtype=factor_tokens.dtype).view(1, -1, 1)
            factor_tokens = factor_tokens * mask
            attn = attn * mask
        flow_edges = self.edge_head(factor_tokens)
        state_token = factor_tokens.mean(1)
        state_logits = self.state_head(state_token)
        stats = {
            "flow_factor_count": self.num_factors,
            "factor_attention_entropy": float((-(attn.clamp_min(1e-9).log() * attn).sum(-1)).mean().detach().cpu()),
            "lag_argmax_mean": float(lag_weights.argmax(-1).float().mean().detach().cpu()),
            "lag_disabled": bool(lag_disabled),
            "lag_context_norm": float(lag_context.norm(dim=-1).mean().detach().cpu()),
        }
        return InteractionFlowState(
            state_tokens=state_token,
            state_logits=state_logits,
            state_attention=attn,
            lag_weights=lag_weights,
            flow_edges=flow_edges,
            factor_tokens=factor_tokens,
            stats=stats,
        )
