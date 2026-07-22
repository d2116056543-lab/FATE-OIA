from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from fate_oia.models.acpr_sparse_ops import entmax15_bisect


class PRECISESemanticExchange(nn.Module):
    def __init__(self, fields: list[dict[str, Any]], reason_schema: list[dict[str, Any]], dim: int = 384, overlap_tau: float = 0.08, overlap_slope: float = 12.0, reliability_eps: float = 1e-4, action_gamma_init: float = 0.05, reason_gamma_init: float = 0.05, gamma_max: float = 0.25) -> None:
        super().__init__()
        self.fields = fields
        self.reason_schema = reason_schema
        self.overlap_tau = overlap_tau
        self.overlap_slope = overlap_slope
        self.reliability_eps = reliability_eps
        self.gamma_max = gamma_max
        action_raw = math.log((action_gamma_init / gamma_max) / (1.0 - action_gamma_init / gamma_max))
        reason_raw = math.log((reason_gamma_init / gamma_max) / (1.0 - reason_gamma_init / gamma_max))
        self.action_gamma_raw = nn.Parameter(torch.full((4,), action_raw))
        self.reason_gamma_raw = nn.Parameter(torch.full((21,), reason_raw))
        self.action_query = nn.Linear(dim, dim, bias=False)
        self.reason_query = nn.Linear(dim, dim, bias=False)
        self.evidence_key = nn.Linear(dim, dim, bias=False)
        self.reason_value = nn.Linear(dim, dim, bias=False)
        self.action_value = nn.Linear(dim, dim, bias=False)
        self.action_pair = nn.Linear(dim, dim, bias=False)
        self.reason_pair = nn.Linear(dim, dim, bias=False)
        self.register_buffer("family_mask_action", self._action_mask())
        self.register_buffer("family_mask_reason", self._reason_mask())

    def _action_mask(self) -> torch.Tensor:
        action_families = [{"actor", "drivable", "boundary"}, {"traffic_control", "actor", "boundary"}, {"actor", "drivable", "boundary"}, {"actor", "drivable", "boundary"}]
        return torch.tensor([[field["family"] in allowed for field in self.fields] for allowed in action_families], dtype=torch.bool)

    def _reason_mask(self) -> torch.Tensor:
        return torch.tensor([[field["family"] in set(row["allowed_evidence_families"]) for field in self.fields] for row in self.reason_schema], dtype=torch.bool)

    def _attention(self, tokens: torch.Tensor, query: nn.Linear, mask: torch.Tensor, evidence: torch.Tensor, reliability: torch.Tensor) -> torch.Tensor:
        logits = torch.einsum("bcd,bed->bce", query(tokens), self.evidence_key(evidence)) / (tokens.shape[-1] ** 0.5)
        logits = logits + reliability.clamp_min(self.reliability_eps).log().unsqueeze(1)
        allowed = mask.view(1, *mask.shape)
        logits = logits.masked_fill(~allowed, -1e4)
        attention = entmax15_bisect(logits, dim=-1) * allowed
        return attention / attention.sum(-1, keepdim=True).clamp_min(1e-8)

    def forward(self, action_tokens: torch.Tensor, reason_tokens: torch.Tensor, explicit_evidence: torch.Tensor, reliability: torch.Tensor, mode: str = "certified", evidence_grad: bool = False) -> dict[str, torch.Tensor]:
        evidence = explicit_evidence if evidence_grad else explicit_evidence.detach()
        rho = reliability if evidence_grad else reliability.detach()
        reasons = reason_tokens
        if mode == "evidence_shuffled":
            evidence, rho = evidence.roll(1, 0), rho.roll(1, 0)
        if mode == "reason_tokens_shuffled":
            reasons = reasons.flip(1)
        action_attention = self._attention(action_tokens, self.action_query, self.family_mask_action, evidence, rho)
        reason_attention = self._attention(reasons, self.reason_query, self.family_mask_reason, evidence, rho)
        action_overlap = torch.einsum("bae,bre,be->bar", action_attention, reason_attention.detach(), rho)
        reason_overlap = torch.einsum("bae,bre,be->bar", action_attention.detach(), reason_attention, rho)
        action_gate = torch.sigmoid(self.overlap_slope * (action_overlap - self.overlap_tau))
        reason_gate = torch.sigmoid(self.overlap_slope * (reason_overlap - self.overlap_tau))
        action_certificate = action_overlap * (torch.ones_like(action_gate) if mode == "ungated" else action_gate)
        reason_certificate = reason_overlap * (torch.ones_like(reason_gate) if mode == "ungated" else reason_gate)
        if mode in {"off", "latent_only"}:
            action_certificate = torch.zeros_like(action_certificate)
            reason_certificate = torch.zeros_like(reason_certificate)
        reason_scores = torch.einsum("bad,brd->bar", self.action_pair(action_tokens), self.reason_pair(reasons.detach())) / (action_tokens.shape[-1] ** 0.5)
        action_weights = torch.softmax(reason_scores + action_certificate.clamp_min(1e-8).log(), dim=-1) * action_certificate
        action_message = torch.einsum("bar,brd->bad", action_weights, self.reason_value(reasons.detach()))
        action_delta = self.gamma_max * torch.sigmoid(self.action_gamma_raw).view(1, 4, 1) * action_message
        action_scores = torch.einsum("brd,bad->bra", self.reason_pair(reasons), self.action_pair(action_tokens.detach())) / (action_tokens.shape[-1] ** 0.5)
        reason_weights = torch.softmax(action_scores + reason_certificate.transpose(1, 2).clamp_min(1e-8).log(), dim=-1) * reason_certificate.transpose(1, 2)
        reason_message = torch.einsum("bra,bad->brd", reason_weights, self.action_value(action_tokens.detach()))
        reason_delta = self.gamma_max * torch.sigmoid(self.reason_gamma_raw).view(1, 21, 1) * reason_message
        return {
            "action_evidence_attention": action_attention,
            "reason_evidence_attention": reason_attention,
            "exchange_overlap": action_overlap,
            "exchange_gate": action_gate,
            "action_exchange_delta": action_delta,
            "reason_exchange_delta": reason_delta,
            "action_reason_message_norm": action_message.norm(dim=-1).mean(),
            "reason_action_message_norm": reason_message.norm(dim=-1).mean(),
            "wrong_target_message_ratio": action_weights.min(dim=-1).values.mean() / action_weights.max(dim=-1).values.mean().clamp_min(1e-8),
        }
