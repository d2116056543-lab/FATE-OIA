from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .acpr_reason_grammar import ACPRReasonGrammar


def _bounded_log_prior(prior: Tensor, bound: float = 1.5) -> Tensor:
    """Log-density ratio against a uniform field, bounded as required."""
    uniform_log = -math.log(prior.shape[-1])
    return (prior.clamp_min(1e-8).log() - uniform_log).clamp(min=-float(bound), max=float(bound))


class AIEReasonRereader(nn.Module):
    def __init__(
        self,
        dim: int = 384,
        reason_dim: int = 21,
        action_dim: int = 4,
        probes_per_action: int = 4,
        num_predicates: int = 32,
        predicate_names: list[str] | None = None,
        grammar_path: str = "configs/acpr_reason_predicate_grammar.yaml",
        num_layers: int = 3,
        action_prior_max: float = 0.75,
        predicate_prior_max: float = 0.75,
        kappa: float = 4.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.reason_dim = reason_dim
        self.num_layers = num_layers
        self.action_prior_max = float(action_prior_max)
        self.predicate_prior_max = float(predicate_prior_max)
        self.kappa = float(kappa)
        self.action_query = nn.Linear(dim, dim)
        self.action_key = nn.Linear(dim, dim)
        self.reason_query = nn.Linear(dim, dim)
        self.field_keys = nn.ModuleList(nn.Linear(dim, dim) for _ in range(num_layers))
        self.field_values = nn.ModuleList(nn.Linear(dim, dim) for _ in range(num_layers))
        self.layer_logits = nn.Parameter(torch.zeros(reason_dim, num_layers))
        self.action_prior_strength = nn.Parameter(torch.zeros(reason_dim))
        self.predicate_prior_strength = nn.Parameter(torch.zeros(reason_dim))
        self.private_attention = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.delta_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)
        grammar = ACPRReasonGrammar(grammar_path)
        names = predicate_names or [str(i) for i in range(num_predicates)]
        positive, contradictory = grammar.reason_predicate_matrix(names)
        self.register_buffer("positive_predicate_mask", torch.tensor(positive, dtype=torch.float32), persistent=False)
        self.register_buffer("contradictory_predicate_mask", torch.tensor(contradictory, dtype=torch.float32), persistent=False)

    def forward(
        self,
        reason_nodes_primary: Tensor,
        patch_tokens_by_layer: Tensor,
        evidence_token: Tensor,
        evidence_map: Tensor,
        bounded_contribution: Tensor,
        predicate_attention: Tensor,
        predicate_probs: Tensor,
        reason_logits_primary: Tensor,
        *,
        reason_scale: float = 1.0,
        action_prior_enabled: bool = True,
        predicate_prior_enabled: bool = True,
    ) -> dict[str, Tensor]:
        reason_nodes = reason_nodes_primary.detach()
        evidence = evidence_token.detach()
        maps = evidence_map.detach()
        contributions = bounded_contribution.detach()
        pattn = predicate_attention.detach().clamp_min(1e-8)
        pprob = predicate_probs.detach().clamp(0, 1)
        action_score = torch.einsum(
            "brd,bakd->brak",
            self.action_query(reason_nodes),
            self.action_key(evidence),
        ) / math.sqrt(self.dim)
        gamma = torch.softmax(action_score.flatten(2), dim=-1).view_as(action_score)
        gamma = gamma * (contributions.abs()[:, None] + 1e-8).sqrt()
        gamma = gamma / gamma.sum((2, 3), keepdim=True).clamp_min(1e-8)
        action_prior = torch.einsum("brak,bakn->brn", gamma, maps)
        action_prior = action_prior / action_prior.sum(-1, keepdim=True).clamp_min(1e-8)
        pos = self.positive_predicate_mask.to(pprob.device, pprob.dtype)
        neg = self.contradictory_predicate_mask.to(pprob.device, pprob.dtype)
        signed = pprob[:, None, :] * (pos[None] - 0.5 * neg[None])
        predicate_prior = torch.einsum("brp,bpn->brn", signed, pattn).clamp_min(0)
        predicate_prior = predicate_prior / predicate_prior.sum(-1, keepdim=True).clamp_min(1e-8)
        field = patch_tokens_by_layer.detach()
        # The bounded spatial priors can only condition rereading when Q/K scale is
        # controlled. Parameter-free normalization preserves the planned dot product
        # while preventing raw field magnitude from saturating the softmax.
        q = F.layer_norm(self.reason_query(reason_nodes), (self.dim,))
        layer_mix = torch.softmax(self.layer_logits, dim=-1)
        layer_scores, layer_values = [], []
        for layer in range(self.num_layers):
            key = F.layer_norm(self.field_keys[layer](field[:, layer]), (self.dim,))
            value = self.field_values[layer](field[:, layer])
            layer_scores.append(torch.einsum("brd,bnd->brn", q, key) / math.sqrt(self.dim))
            layer_values.append(value)
        score = torch.stack(layer_scores, dim=2)
        action_bias = score.new_zeros(action_prior.shape[0], self.reason_dim, 1, action_prior.shape[-1])
        if action_prior_enabled:
            strength = self.action_prior_max * torch.sigmoid(self.action_prior_strength)[None, :, None, None]
            action_bias = strength * _bounded_log_prior(action_prior)[:, :, None, :]
            score = score + action_bias
        predicate_bias = score.new_zeros(predicate_prior.shape[0], self.reason_dim, 1, predicate_prior.shape[-1])
        if predicate_prior_enabled:
            strength = self.predicate_prior_max * torch.sigmoid(self.predicate_prior_strength)[None, :, None, None]
            predicate_bias = strength * _bounded_log_prior(predicate_prior)[:, :, None, :]
            score = score + predicate_bias
        score = score + layer_mix[None, :, :, None].clamp_min(1e-8).log()
        attention = torch.softmax(score.flatten(2), dim=-1).view_as(score)
        private = sum(torch.einsum("brn,bnd->brd", attention[:, :, layer], layer_values[layer]) for layer in range(self.num_layers))
        private = private + self.private_attention(private, private, private, need_weights=False)[0]
        raw_delta = self.delta_head(private).squeeze(-1)
        delta = float(reason_scale) * self.kappa * torch.tanh(raw_delta / self.kappa)
        final = reason_logits_primary + delta
        final_train = reason_logits_primary.detach() + delta
        return {
            "reason_action_evidence_attention": gamma,
            "reason_action_prior": action_prior,
            "reason_predicate_prior": predicate_prior,
            "reason_private_attention": attention,
            "reason_private_token": private,
            "reason_visual_score_rms": torch.stack(layer_scores, dim=2).float().square().mean().sqrt(),
            "reason_action_prior_bias_rms": action_bias.float().square().mean().sqrt(),
            "reason_predicate_prior_bias_rms": predicate_bias.float().square().mean().sqrt(),
            "reason_delta": delta,
            "reason_logits_final": final,
            "reason_logits_final_train": final_train,
        }
