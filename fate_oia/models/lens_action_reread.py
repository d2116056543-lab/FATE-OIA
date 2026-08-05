from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .acpr_sparse_ops import entmax15_bisect


class LENSActionReread(nn.Module):
    """Full-field action reread. Named factors are soft biases, never cropped ROIs."""

    def __init__(self, dim: int = 384, action_dim: int = 4, reason_dim: int = 21, cap: float = 20.0) -> None:
        super().__init__()
        self.action_dim, self.reason_dim, self.cap = action_dim, reason_dim, cap
        self.factor_query = nn.Linear(dim, dim)
        self.factor_key = nn.Linear(dim, dim)
        self.patch_query = nn.Linear(dim, dim)
        self.patch_key = nn.Linear(dim, dim)
        self.patch_value = nn.Linear(dim, dim)
        self.named_weight = nn.Parameter(torch.randn(action_dim, reason_dim, dim * 3) * 0.02)
        self.named_bias = nn.Parameter(torch.zeros(action_dim, reason_dim))
        self.unnamed_head = nn.Linear(dim, 1)
        self.null_factor_bias = nn.Parameter(torch.zeros(action_dim))
        self.source_attention_bias = nn.Parameter(torch.tensor(0.25))
        self.map_bias = nn.Parameter(torch.tensor(0.25))

    def _bound_direction(self, logits: Tensor, alpha: float) -> Tensor:
        if alpha == 0.0:
            return logits
        scale=(self.cap/logits.norm(dim=-1,keepdim=True).clamp_min(1e-8)).clamp(max=1.0)
        return logits*scale

    def forward(
        self,
        *,
        action_nodes: Tensor,
        detail_tokens: Tensor,
        source_action_attention: Tensor,
        evidence_map: Tensor,
        evidence_token: Tensor,
        state_prob: Tensor,
        state_token: Tensor,
        state_embeddings: Tensor,
        action_logits_base: Tensor,
        progress: float,
        factor_chunk_size: int = 21,
    ) -> dict[str, Tensor]:
        b, actions, d = action_nodes.shape
        reasons, n = self.reason_dim, detail_tokens.shape[1]
        aq = self.factor_query(action_nodes)
        fk = self.factor_key(state_token)
        factor_score = torch.einsum("bad,brd->bar", aq, fk) / math.sqrt(d)
        factor_score = torch.cat([factor_score, self.null_factor_bias.view(1, actions, 1).expand(b, -1, -1)], dim=-1)
        factor_selection = entmax15_bisect(factor_score, dim=-1)
        pq = self.patch_query(action_nodes)
        pk = self.patch_key(detail_tokens)
        pv = self.patch_value(detail_tokens)
        action_patch_score = torch.einsum("bad,bnd->ban", pq, pk) / math.sqrt(d)
        action_patch_score = action_patch_score + self.source_attention_bias * torch.log(source_action_attention.clamp_min(1e-8))
        local_tokens: list[Tensor] = []
        local_attention: list[Tensor] = []
        for start in range(0, reasons, factor_chunk_size):
            end = min(reasons, start + factor_chunk_size)
            factor_map = evidence_map[:, start:end]
            score = action_patch_score.unsqueeze(2) + self.map_bias * torch.log(factor_map.clamp_min(1e-8)).unsqueeze(1)
            attention = torch.softmax(score, dim=-1)
            local_attention.append(attention)
            local_tokens.append(torch.einsum("barn,bnd->bard", attention, pv))
        factor_local_attention = torch.cat(local_attention, dim=2)
        local = torch.cat(local_tokens, dim=2)
        evidence = evidence_token.unsqueeze(1).expand(-1, actions, -1, -1)
        contributions = []
        for state_index in (0, 1):
            state_embed = state_embeddings[:, state_index].view(1, 1, reasons, d).expand(b, actions, -1, -1)
            features = torch.cat([local, evidence, state_embed], dim=-1)
            score = torch.einsum("bard,ard->bar", features, self.named_weight) + self.named_bias.unsqueeze(0)
            contributions.append(score * factor_selection[:, :, :reasons])
        # State-specific U_r+/U_r- are scored independently; unknown receives no named credit.
        unknown = torch.zeros_like(contributions[0])
        factor_contribution_state = torch.stack([contributions[0], contributions[1], unknown], dim=-1)
        factor_contribution_expected = (factor_contribution_state * state_prob.unsqueeze(1)).sum(-1)
        unnamed_attention = torch.softmax(action_patch_score, dim=-1)
        unnamed_local = torch.einsum("ban,bnd->bad", unnamed_attention, pv)
        unnamed = self.unnamed_head(unnamed_local).squeeze(-1) * factor_selection[:, :, reasons]
        raw_total = unnamed + factor_contribution_expected.sum(-1)
        alpha = float(max(0.0, min(1.0, progress)))
        bounded_delta = self.cap * torch.tanh(raw_total / self.cap)
        final = self._bound_direction(action_logits_base + alpha * bounded_delta,alpha)
        total_effect=final-action_logits_base
        safe_raw = torch.where(raw_total.abs() < 1e-8, torch.ones_like(raw_total), raw_total)
        scale = total_effect / safe_raw
        bounded_named = factor_contribution_expected * scale.unsqueeze(-1)
        bounded_unnamed = unnamed * scale
        bounded_unnamed = bounded_unnamed + (total_effect-bounded_named.sum(-1)-bounded_unnamed)
        factor_aux = action_logits_base.detach() + total_effect
        variants = []
        for state in range(3):
            variant_raw = raw_total.unsqueeze(2).expand(-1, -1, reasons).transpose(1, 2)
            variant_raw = variant_raw - factor_contribution_expected.transpose(1, 2) + factor_contribution_state[..., state].transpose(1, 2)
            variant_delta = self.cap * torch.tanh(variant_raw / self.cap)
            variants.append(self._bound_direction(action_logits_base.unsqueeze(1) + alpha * variant_delta,alpha))
        state_substitution = torch.stack(variants, dim=2)
        reconstructed = bounded_named.sum(-1) + bounded_unnamed
        return {
            "factor_selection": factor_selection,
            "factor_local_attention": factor_local_attention,
            "factor_contribution_state": factor_contribution_state,
            "factor_contribution_expected": factor_contribution_expected,
            "factor_contribution_bounded": bounded_named,
            "unnamed_contribution": bounded_unnamed,
            "action_logits_factor_aux": factor_aux,
            "action_logits_final": final,
            "action_logits_state_substitution": state_substitution,
            "contribution_reconstruction_error": (final - action_logits_base - reconstructed).abs().max(),
        }
