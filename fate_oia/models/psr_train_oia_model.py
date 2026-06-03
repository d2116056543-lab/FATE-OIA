from __future__ import annotations

import torch
from torch import nn

from fate_oia.models.label_query_head import LabelQueryHead


DEFAULT_REASON_GROUPS = [
    0, 0, 0, 0, 0,
    3, 3, 0, 3, 3,
    3, 3, 3, 3, 3,
    1, 1, 1, 1, 2, 2,
]


class _SpecialistAdapter(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, label_tokens: torch.Tensor) -> torch.Tensor:
        return label_tokens + self.net(label_tokens)


class PSRTrainOIAFeatureModel(nn.Module):
    """End-to-end PSR-Train head over DINO/ViT image tokens.

    This is intentionally not an offline PSR/logit router. It owns action,
    explanation and calibration specialists inside one trainable model.
    """

    def __init__(
        self,
        dim: int = 384,
        action_dim: int = 4,
        reason_dim: int = 21,
        num_heads: int = 4,
        dropout: float = 0.1,
        action_delta_cap: float = 0.04,
        specialist_warmup_epochs: int = 4,
        router_warmup_epochs: int = 6,
        reason_group_count: int = 4,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.action_dim = action_dim
        self.reason_dim = reason_dim
        self.total_dim = action_dim + reason_dim
        self.action_delta_cap = float(action_delta_cap)
        self.specialist_warmup_epochs = int(specialist_warmup_epochs)
        self.router_warmup_epochs = int(router_warmup_epochs)
        self.shared_label_query = LabelQueryHead(dim, self.total_dim, num_heads=num_heads, dropout=dropout)
        self.action_expert = _SpecialistAdapter(dim, dropout=dropout)
        self.explanation_expert = _SpecialistAdapter(dim, dropout=dropout)
        self.shared_expert = _SpecialistAdapter(dim, dropout=dropout)

        self.a_action_head = nn.Linear(dim, 1)
        self.a_reason_head = nn.Linear(dim, 1)
        self.e_action_head = nn.Linear(dim, 1)
        self.e_reason_head = nn.Linear(dim, 1)

        self.reason_calibration_bias = nn.Parameter(torch.zeros(reason_dim))
        self.reason_calibration_log_temp = nn.Parameter(torch.zeros(reason_dim))
        self.action_router = nn.Sequential(
            nn.Linear(dim * 2 + action_dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, action_dim),
        )
        self.action_delta = nn.Linear(dim * 2, action_dim)
        self.reason_group_embed = nn.Embedding(reason_group_count, dim)
        groups = torch.tensor(DEFAULT_REASON_GROUPS[:reason_dim], dtype=torch.long)
        if groups.numel() < reason_dim:
            groups = torch.cat([groups, torch.full((reason_dim - groups.numel(),), 3, dtype=torch.long)])
        self.register_buffer("reason_group_ids", groups.clamp(min=0, max=reason_group_count - 1), persistent=False)
        self.reason_router = nn.Sequential(
            nn.LayerNorm(dim + 3),
            nn.Linear(dim + 3, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )

    def _router_scale(self, epoch: int | None) -> torch.Tensor | float:
        if epoch is None:
            return 1.0
        if epoch < self.specialist_warmup_epochs:
            return 0.0
        denom = max(1, self.router_warmup_epochs - self.specialist_warmup_epochs)
        return min(1.0, max(0.0, (float(epoch) - float(self.specialist_warmup_epochs) + 1.0) / float(denom)))

    @staticmethod
    def _margin_entropy(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        prob = torch.sigmoid(logits)
        margin = (prob - 0.5).abs()
        entropy = -(prob * prob.clamp_min(1e-6).log() + (1.0 - prob) * (1.0 - prob).clamp_min(1e-6).log())
        return margin, entropy

    def forward(self, tokens: torch.Tensor, epoch: int | None = None) -> dict[str, torch.Tensor]:
        shared = self.shared_label_query(tokens)
        raw_label_tokens = shared["label_tokens"]
        shared_tokens = self.shared_expert(raw_label_tokens)
        action_tokens = self.action_expert(shared_tokens)
        explanation_tokens = self.explanation_expert(shared_tokens)

        a_action_tokens = action_tokens[:, : self.action_dim]
        a_reason_tokens = action_tokens[:, self.action_dim :]
        e_action_tokens = explanation_tokens[:, : self.action_dim]
        e_reason_tokens = explanation_tokens[:, self.action_dim :]

        a_action_logits = self.a_action_head(a_action_tokens).squeeze(-1)
        a_reason_logits = self.a_reason_head(a_reason_tokens).squeeze(-1)
        e_action_logits = self.e_action_head(e_action_tokens).squeeze(-1)
        e_reason_logits = self.e_reason_head(e_reason_tokens).squeeze(-1)

        c_reason_logits = e_reason_logits * torch.exp(self.reason_calibration_log_temp).clamp(0.5, 2.0) + self.reason_calibration_bias

        a_summary = a_action_tokens.mean(1)
        e_summary = e_reason_tokens.mean(1)
        a_margin, a_entropy = self._margin_entropy(a_action_logits)
        router_in = torch.cat([a_summary, e_summary, a_margin, a_entropy], dim=-1)
        action_gate = torch.sigmoid(self.action_router(router_in))
        bounded_delta = torch.tanh(self.action_delta(torch.cat([a_summary, e_summary], dim=-1))) * self.action_delta_cap
        scale = self._router_scale(epoch)
        final_action = a_action_logits + float(scale) * action_gate * bounded_delta

        e_margin, e_entropy = self._margin_entropy(e_reason_logits)
        disagreement = torch.sigmoid(e_reason_logits) - torch.sigmoid(a_reason_logits)
        group_emb = self.reason_group_embed(self.reason_group_ids).unsqueeze(0).expand(tokens.shape[0], -1, -1)
        reason_router_in = torch.cat([e_reason_tokens + group_emb, e_margin.unsqueeze(-1), e_entropy.unsqueeze(-1), disagreement.unsqueeze(-1)], dim=-1)
        reason_gate = torch.sigmoid(self.reason_router(reason_router_in).squeeze(-1))
        if float(scale) <= 0.0:
            final_reason = e_reason_logits
            active_reason_gate = torch.ones_like(reason_gate)
        else:
            active_reason_gate = reason_gate
            final_reason = active_reason_gate * c_reason_logits + (1.0 - active_reason_gate) * a_reason_logits

        shared_conflict_proxy = (a_summary - e_summary).pow(2).mean()
        return {
            **shared,
            "label_tokens_raw": raw_label_tokens,
            "label_tokens_shared": shared_tokens,
            "action_logits": final_action,
            "reason_logits": final_reason,
            "final_action_logits": final_action,
            "final_reason_logits": final_reason,
            "a_action_logits": a_action_logits,
            "a_reason_logits": a_reason_logits,
            "e_action_logits": e_action_logits,
            "e_reason_logits": e_reason_logits,
            "c_reason_logits": c_reason_logits,
            "action_visual_logits": a_action_logits,
            "action_reason_logits": e_action_logits,
            "action_fused_logits": final_action,
            "reason_router_gate": active_reason_gate,
            "action_router_gate": action_gate,
            "action_delta": bounded_delta,
            "router_scale": torch.as_tensor(float(scale), device=tokens.device),
            "shared_conflict_proxy": shared_conflict_proxy,
            "psr_train_single_model": torch.ones((), device=tokens.device, dtype=torch.bool),
        }
