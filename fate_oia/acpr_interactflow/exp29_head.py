from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .cluster_semantics import load_exp29_names
from .types import Exp29Output


class Exp29Head(nn.Module):
    """Contribution-grounded weak explanation head for PSI Exp29 clusters.

    The deploy path is calibrated fixed-threshold: raw logits remain available
    for ranking/audit, while calibrated logits are used by primary fixed-F1
    evaluation. Cluster attention is anchored to exact ledger state
    contributions instead of a generic label-query over tokens.
    """

    def __init__(self, dim: int = 384, exp_dim: int = 29, label_names_path: str | None = None, max_factors: int = 16) -> None:
        super().__init__()
        self.label_names = load_exp29_names(label_names_path)
        self.exp_dim = exp_dim
        self.max_factors = max_factors
        self.queries = nn.Parameter(torch.randn(exp_dim, dim) * 0.02)
        self.factor_key = nn.Linear(dim, dim)
        self.factor_value = nn.Linear(dim + 3, dim)
        self.predicate_proj = nn.Linear(dim, dim)
        self.global_proj = nn.Linear(dim, dim)
        self.logit = nn.Linear(dim, 1)
        self.delta = nn.Sequential(nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1))
        self.theta = nn.Parameter(torch.zeros(exp_dim))
        self.delta_scale = nn.Parameter(torch.zeros(exp_dim))
        self.register_buffer("cluster_reliability_prior", torch.ones(exp_dim), persistent=True)
        self.register_buffer("theta_quality_mf1", torch.tensor(-1.0), persistent=True)
        self.register_buffer("theta_pred_positive_rate", torch.tensor(0.0), persistent=True)
        self.register_buffer("theta_last_epoch", torch.tensor(-1), persistent=True)
        prior = torch.full((exp_dim, max_factors), 1.0 / max(1, max_factors))
        self.register_buffer("cluster_to_state_prior_full", prior, persistent=True)

    def _state_prior(self, factors: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        prior = self.cluster_to_state_prior_full[:, :factors]
        if prior.shape[1] < factors:
            pad = torch.full((self.exp_dim, factors - prior.shape[1]), 1.0 / factors, device=prior.device, dtype=prior.dtype)
            prior = torch.cat([prior, pad], dim=1)
        prior = prior.to(device=device, dtype=dtype)
        return prior / prior.sum(-1, keepdim=True).clamp_min(1e-8)

    def forward(
        self,
        factor_tokens_lag: torch.Tensor,
        predicate_tokens_summary: torch.Tensor | None = None,
        gated_state_contributions: torch.Tensor | None = None,
        global_decision_hidden: torch.Tensor | None = None,
        action_logits: torch.Tensor | None = None,
        exp29_mask: torch.Tensor | None = None,
    ) -> Exp29Output:
        b, factors, d = factor_tokens_lag.shape
        if gated_state_contributions is None:
            gated_state_contributions = factor_tokens_lag.new_zeros(b, factors, 3)
        contrib_mag = gated_state_contributions.abs().sum(-1)
        contrib_norm = contrib_mag / contrib_mag.sum(-1, keepdim=True).clamp_min(1e-8)
        prior = self._state_prior(factors, factor_tokens_lag.device, factor_tokens_lag.dtype)

        q = self.queries.view(1, self.exp_dim, d)
        k = self.factor_key(factor_tokens_lag)
        score = torch.einsum("bkd,bfd->bkf", q.expand(b, -1, -1), k) / (d ** 0.5)
        score = score + 0.75 * contrib_norm.clamp_min(1e-8).log().unsqueeze(1)
        score = score + prior.clamp_min(1e-8).log().unsqueeze(0)
        if exp29_mask is not None:
            sample_reliability = exp29_mask.float().mean(-1, keepdim=True).unsqueeze(-1)
            score = score + 0.05 * sample_reliability
        attention = torch.softmax(score, dim=-1)

        source = torch.cat([factor_tokens_lag, gated_state_contributions], dim=-1)
        label_tokens = torch.einsum("bkf,bfd->bkd", attention, self.factor_value(source))
        if predicate_tokens_summary is not None:
            label_tokens = label_tokens + 0.15 * self.predicate_proj(predicate_tokens_summary.mean(1)).unsqueeze(1)
        if global_decision_hidden is not None:
            label_tokens = label_tokens + 0.10 * self.global_proj(global_decision_hidden).unsqueeze(1)
        logits_raw = self.logit(label_tokens).squeeze(-1)
        delta = self.delta(label_tokens).squeeze(-1).clamp(-0.50, 0.50)
        logits_calibrated = logits_raw - self.theta.view(1, -1) + torch.tanh(self.delta_scale).view(1, -1) * delta
        probs_raw = torch.sigmoid(logits_raw)
        probs_calibrated = torch.sigmoid(logits_calibrated)
        reliability = self.cluster_reliability_prior.to(device=label_tokens.device, dtype=label_tokens.dtype).clamp(0.0, 1.0)
        entropy = -(attention.clamp_min(1e-9).log() * attention).sum(-1)
        stats = {
            "exp29_attention_entropy": float(entropy.mean().detach().cpu()),
            "exp29_raw_positive_rate": float((probs_raw > 0.5).float().mean().detach().cpu()),
            "exp29_calibrated_positive_rate": float((probs_calibrated > 0.5).float().mean().detach().cpu()),
            "exp29_contribution_mass_mean": float(contrib_mag.mean().detach().cpu()),
            "exp29_theta_abs_mean": float(self.theta.detach().abs().mean().cpu()),
        }
        return Exp29Output(
            logits_raw=logits_raw,
            logits_calibrated=logits_calibrated,
            probs_raw=probs_raw,
            probs_calibrated=probs_calibrated,
            label_mask=torch.ones_like(logits_raw),
            label_names=self.label_names,
            cluster_attention_to_factors=attention,
            cluster_reliability=reliability,
            cluster_to_state_prior=prior,
            attention=attention,
            stats=stats,
        )
