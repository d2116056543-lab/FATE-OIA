from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .aie_cert_sparse import entmax15


class AIECertPredicateBank(nn.Module):
    def __init__(self, dim: int = 384, num_predicates: int = 32, key_dim: int = 64) -> None:
        super().__init__()
        self.predicate_keys = nn.Parameter(torch.randn(num_predicates, key_dim) * 0.02)
        self.probe_projection = nn.Linear(dim, key_dim)

    def forward(
        self,
        global_token: Tensor,
        predicate_probs: Tensor,
        predicate_attention: Tensor,
        presence_threshold: float = 0.30,
        prior_strength_max: float = 0.25,
        log_density_bound: float = 1.5,
    ) -> dict[str, Tensor]:
        logits = torch.einsum("bakd,pd->bakp", self.probe_projection(global_token), self.predicate_keys)
        logits = logits / math.sqrt(self.predicate_keys.shape[-1])
        logits = logits + predicate_probs[:, None, None, :].clamp_min(1e-8).log()
        mixture = entmax15(logits, dim=-1)
        available = predicate_probs.amax(-1)[:, None, None] >= presence_threshold
        mixture = torch.where(available[..., None], mixture, torch.zeros_like(mixture))
        mixture_map = torch.einsum("bakp,bpn->bakn", mixture, predicate_attention.clamp_min(0.0))
        mixture_map = mixture_map / mixture_map.sum(-1, keepdim=True).clamp_min(1e-8)
        log_ratio = (mixture_map.clamp_min(1e-8).log() + math.log(mixture_map.shape[-1])).clamp(
            -log_density_bound, log_density_bound
        )
        strength = prior_strength_max * available.to(global_token.dtype)
        return {
            "shared_predicate_keys": self.predicate_keys,
            "predicate_mixture": mixture,
            "predicate_mixture_map": mixture_map,
            "predicate_prior_available": available.expand_as(global_token[..., 0]),
            "predicate_prior_strength": strength.expand_as(global_token[..., 0]),
            "predicate_log_density_ratio": log_ratio,
        }
