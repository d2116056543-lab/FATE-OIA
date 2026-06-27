from __future__ import annotations

import torch
from torch import nn

from .interaction_grammar import InteractionGrammar


class ObjectiveEnvironmentStateBank(nn.Module):
    def __init__(self, grammar_path: str, dim: int = 384, num_predicates: int = 48) -> None:
        super().__init__()
        self.grammar = InteractionGrammar(grammar_path)
        self.group_names = list(self.grammar.state_groups.keys())
        self.group_queries = nn.Parameter(torch.randn(len(self.group_names), dim) * 0.02)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.logit = nn.Linear(dim, 1)
        self.layer_weight = nn.Linear(dim, 3)
        self.num_predicates = num_predicates

    def forward(self, predicate_tokens: torch.Tensor, predicate_probs: torch.Tensor) -> dict[str, torch.Tensor | dict]:
        d = predicate_tokens.shape[-1]
        score = torch.einsum("gd,bpd->bgp", self.group_queries, self.key(predicate_tokens)) / (d ** 0.5)
        score = score + predicate_probs.clamp_min(1e-4).log().unsqueeze(1)
        attention = torch.softmax(score, dim=-1)
        state_tokens = torch.einsum("bgp,bpd->bgd", attention, self.value(predicate_tokens))
        state_logits = self.logit(state_tokens).squeeze(-1)
        state_layer_weights = torch.softmax(self.layer_weight(state_tokens.mean(1)), dim=-1)
        stats = {
            "state_group_count": len(self.group_names),
            "state_attention_entropy": float((-(attention.clamp_min(1e-9).log() * attention).sum(-1)).mean().detach().cpu()),
        }
        return {
            "state_tokens": state_tokens,
            "state_logits": state_logits,
            "state_attention": attention,
            "state_group_logits": state_logits,
            "state_layer_weights": state_layer_weights,
            "state_stats": stats,
        }

