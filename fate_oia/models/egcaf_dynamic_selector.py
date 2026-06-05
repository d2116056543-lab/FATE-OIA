from __future__ import annotations

import torch
from torch import nn

from fate_oia.models.egcaf_factor_types import FactorBatch, gather_factors_by_indices
from fate_oia.models.egcaf_sparse_topk import entmax15, relaxed_topk, sparsemax


class ExplanationGuidedDynamicFactorSelector(nn.Module):
    def __init__(self, hidden_dim: int = 256, action_dim: int = 4, reason_dim: int = 21, k_max: int = 3, lambda_exp_max: float = 0.35, lambda_scene_max: float = 0.25, sparse_method: str = "entmax15") -> None:
        super().__init__()
        self.action_queries = nn.Parameter(torch.randn(action_dim, hidden_dim) * 0.02)
        self.score = nn.Sequential(nn.Linear(hidden_dim * 2 + 5, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.exp_compat = nn.Linear(reason_dim, action_dim)
        self.scene_compat = nn.Linear(6, action_dim)
        self.k_max = int(k_max)
        self.lambda_exp_max = float(lambda_exp_max)
        self.lambda_scene_max = float(lambda_scene_max)
        self.sparse_method = sparse_method
        self.register_buffer("help_ema", torch.zeros(action_dim, 6))
        self.register_buffer("faith_ema", torch.zeros(action_dim, 6))
        self.register_buffer("hurt_ema", torch.zeros(action_dim, 6))

    def forward(self, factors: FactorBatch, preliminary_reason_logits: torch.Tensor, reason_uncertainty: torch.Tensor | None, scene_state_logits: torch.Tensor) -> dict[str, torch.Tensor | FactorBatch]:
        b, m, d = factors.embeddings.shape
        aq = self.action_queries.view(1, -1, 1, d).expand(b, -1, m, -1)
        fe = factors.embeddings.view(b, 1, m, d).expand(-1, aq.shape[1], -1, -1)
        boxes = factors.boxes.view(b, 1, m, 4).expand(-1, aq.shape[1], -1, -1)
        rel = factors.reliability_init.view(b, 1, m, 1).expand(-1, aq.shape[1], -1, -1)
        visual_score = self.score(torch.cat([aq, fe, boxes, rel], -1)).squeeze(-1)
        exp_prior = self.exp_compat(torch.sigmoid(preliminary_reason_logits)).view(b, -1, 1)
        scene_prior = self.scene_compat(torch.sigmoid(scene_state_logits)).view(b, -1, 1)
        lambda_exp = torch.clamp(torch.sigmoid((self.help_ema + self.faith_ema - self.hurt_ema).mean(-1)) * self.lambda_exp_max, 0, self.lambda_exp_max)
        lambda_scene = torch.full_like(lambda_exp, self.lambda_scene_max)
        factor_type_conf = factors.type_logits.softmax(-1).max(-1).values.view(b, 1, m)
        exp_reliability = torch.sigmoid(factor_type_conf + factors.reliability_init.view(b, 1, m))
        scores = visual_score + lambda_exp.view(1,-1,1) * exp_reliability * exp_prior + lambda_scene.view(1,-1,1) * scene_prior
        scores = scores.masked_fill(~factors.valid_mask.view(b, 1, m), -1e4)
        weights = sparsemax(scores, -1) if self.sparse_method == "sparsemax" else entmax15(scores, -1)
        idx, vals = relaxed_topk(weights, self.k_max, True)
        selected = gather_factors_by_indices(factors, idx)
        entropy = -(weights.clamp_min(1e-9) * weights.clamp_min(1e-9).log()).sum(-1)
        return {
            "selected_factors": selected,
            "selected_indices": idx,
            "selected_weights": vals,
            "factor_scores": scores,
            "factor_weights": weights,
            "lambda_exp": lambda_exp,
            "selector_entropy": entropy,
            "factor_group_usage": torch.bincount(factors.source_ids.flatten(), minlength=4).float(),
        }

    @torch.no_grad()
    def update_reliability(self, faith_delta: torch.Tensor | float, help_delta: torch.Tensor | float, hurt_delta: torch.Tensor | float, decay: float = 0.90) -> None:
        fd = torch.as_tensor(faith_delta, device=self.faith_ema.device, dtype=self.faith_ema.dtype).mean()
        hd = torch.as_tensor(help_delta, device=self.help_ema.device, dtype=self.help_ema.dtype).mean()
        ud = torch.as_tensor(hurt_delta, device=self.hurt_ema.device, dtype=self.hurt_ema.dtype).mean()
        self.faith_ema.mul_(decay).add_((1 - decay) * fd)
        self.help_ema.mul_(decay).add_((1 - decay) * hd)
        self.hurt_ema.mul_(decay).add_((1 - decay) * ud)
