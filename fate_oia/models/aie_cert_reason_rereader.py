from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .acpr_reason_grammar import ACPRReasonGrammar


def _normalize(value: Tensor) -> Tensor:
    value = value.clamp_min(0.0)
    return value / value.sum(-1, keepdim=True).clamp_min(1e-8)


def _log_ratio(value: Tensor, bound: float = 1.5) -> Tensor:
    return (value.clamp_min(1e-8).log() + math.log(value.shape[-1])).clamp(-bound, bound)


def _bc(left: Tensor, right: Tensor, valid: Tensor) -> Tensor:
    score = torch.sqrt((left.clamp_min(0.0) * right.clamp_min(0.0)).clamp_min(1e-12)).sum(-1)
    return torch.where(valid, score, torch.zeros_like(score))


class AIECertReasonRereader(nn.Module):
    def __init__(self, dim=384, reason_dim=21, action_dim=4, num_predicates=32,
                 predicate_names=None, grammar_path="configs/acpr_reason_predicate_grammar.yaml",
                 num_layers=3, kappa=4.0):
        super().__init__()
        self.dim, self.reason_dim, self.num_layers, self.kappa = dim, reason_dim, num_layers, float(kappa)
        self.reason_query = nn.Linear(dim, dim)
        self.atom_key = nn.Linear(dim, dim)
        self.field_keys = nn.ModuleList(nn.Linear(dim, dim) for _ in range(num_layers))
        self.field_values = nn.ModuleList(nn.Linear(dim, dim) for _ in range(num_layers))
        self.layer_logits = nn.Parameter(torch.zeros(reason_dim, num_layers))
        self.prior_strength = nn.Parameter(torch.zeros(reason_dim, 4))
        self.delta_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)
        grammar = ACPRReasonGrammar(grammar_path)
        names = predicate_names or [str(i) for i in range(num_predicates)]
        positive, counter = grammar.reason_predicate_matrix(names)
        self.register_buffer("positive_predicate_mask", torch.tensor(positive, dtype=torch.float32), persistent=False)
        self.register_buffer("contradictory_predicate_mask", torch.tensor(counter, dtype=torch.float32), persistent=False)

    def forward(self, reason_nodes: Tensor, field: Tensor, atom_token: Tensor, atom_map: Tensor,
                contribution: Tensor, predicate_attention: Tensor, predicate_probs: Tensor,
                primary_logits: Tensor, budget_max: float = 0.60, action_prior_max: float = 0.75,
                predicate_prior_max: float = 0.75, action_prior_enabled=True, predicate_prior_enabled=True,
                signed_priors=True, budget_enabled=True, delta_enabled=True) -> dict[str, Tensor]:
        reason_nodes, field = reason_nodes.detach(), field.detach()
        atom_token, atom_map, contribution = atom_token.detach(), atom_map.detach(), contribution.detach()
        predicate_attention, predicate_probs = predicate_attention.detach(), predicate_probs.detach()
        relevance = torch.einsum("brd,bakd->brak", self.reason_query(reason_nodes), self.atom_key(atom_token))
        relevance = relevance / math.sqrt(self.dim)
        support_value = contribution.abs() if not signed_priors else F.relu(contribution)
        inhibit_value = torch.zeros_like(contribution) if not signed_priors else F.relu(-contribution)
        support_weight = torch.softmax(relevance.flatten(2), -1).view_as(relevance) * support_value[:, None]
        inhibit_weight = torch.softmax(relevance.flatten(2), -1).view_as(relevance) * inhibit_value[:, None]
        action_support = _normalize(torch.einsum("brak,bakn->brn", support_weight, atom_map))
        action_inhibit = _normalize(torch.einsum("brak,bakn->brn", inhibit_weight, atom_map))
        pos = self.positive_predicate_mask.to(predicate_probs)
        neg = self.contradictory_predicate_mask.to(predicate_probs)
        pred_support_weight = predicate_probs[:, None] * pos[None]
        pred_counter_weight = predicate_probs[:, None] * neg[None]
        pred_support = _normalize(torch.einsum("brp,bpn->brn", pred_support_weight, predicate_attention))
        pred_counter = _normalize(torch.einsum("brp,bpn->brn", pred_counter_weight, predicate_attention))
        pair_valid_pos = (support_weight.sum((2, 3)) > 0) & (pred_support_weight.sum(-1) > 0)
        pair_valid_neg = (inhibit_weight.sum((2, 3)) > 0) & (pred_counter_weight.sum(-1) > 0)
        agreement = 0.5 * (_bc(action_support, pred_support, pair_valid_pos) +
                           _bc(action_inhibit, pred_counter, pair_valid_neg))
        uncertainty = 4.0 * torch.sigmoid(primary_logits).detach() * (1.0 - torch.sigmoid(primary_logits).detach())
        budget = 0.10 + (float(budget_max) - 0.10) * uncertainty * agreement if budget_enabled else torch.full_like(uncertainty, float(budget_max))
        q = F.layer_norm(self.reason_query(reason_nodes), (self.dim,))
        strengths = torch.sigmoid(self.prior_strength)
        layer_mix = torch.softmax(self.layer_logits, -1)
        scores, values = [], []
        for layer in range(self.num_layers):
            key = F.layer_norm(self.field_keys[layer](field[:, layer]), (self.dim,))
            score = torch.einsum("brd,bnd->brn", q, key) / math.sqrt(self.dim)
            if action_prior_enabled:
                score = score + action_prior_max * strengths[None, :, 0, None] * _log_ratio(action_support)
                score = score - action_prior_max * strengths[None, :, 1, None] * _log_ratio(action_inhibit)
            if predicate_prior_enabled:
                score = score + predicate_prior_max * strengths[None, :, 2, None] * _log_ratio(pred_support)
                score = score - predicate_prior_max * strengths[None, :, 3, None] * _log_ratio(pred_counter)
            scores.append(score + layer_mix[None, :, layer, None].clamp_min(1e-8).log())
            values.append(self.field_values[layer](field[:, layer]))
        score_stack = torch.stack(scores, 2)
        attention = torch.softmax(score_stack.flatten(2), -1).view_as(score_stack)
        private = sum(torch.einsum("brn,bnd->brd", attention[:, :, layer], values[layer])
                      for layer in range(self.num_layers))
        raw_delta = self.delta_head(private).squeeze(-1)
        delta = budget * self.kappa * torch.tanh(raw_delta / self.kappa) if delta_enabled else torch.zeros_like(raw_delta)
        final = primary_logits + delta
        return {"reason_action_support_prior": action_support, "reason_action_inhibit_prior": action_inhibit,
            "reason_predicate_support_prior": pred_support, "reason_predicate_counter_prior": pred_counter,
            "reason_private_attention": attention, "reason_uncertainty": uncertainty,
            "reason_evidence_agreement": agreement, "reason_budget": budget, "reason_delta": delta,
            "reason_logits_final": final, "reason_logits_final_train": primary_logits.detach() + delta}
