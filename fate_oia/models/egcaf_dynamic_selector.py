from __future__ import annotations

import torch
from torch import nn

from fate_oia.models.egcaf_factor_types import FactorBatch, gather_factors_by_indices
from fate_oia.models.egcaf_sparse_topk import entmax15, relaxed_topk, sparsemax


class ExplanationGuidedDynamicFactorSelector(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        action_dim: int = 4,
        reason_dim: int = 21,
        num_types: int = 11,
        k_max: int = 3,
        lambda_exp_max: float = 0.35,
        lambda_scene_max: float = 0.25,
        sparse_method: str = "entmax15",
    ) -> None:
        super().__init__()
        self.action_queries = nn.Parameter(torch.randn(action_dim, hidden_dim) * 0.02)
        self.score = nn.Sequential(nn.Linear(hidden_dim * 2 + 5, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.reason_type_compat = nn.Parameter(torch.zeros(reason_dim, num_types))
        self.action_type_compat = nn.Parameter(torch.zeros(action_dim, num_types))
        self.action_source_compat = nn.Parameter(torch.zeros(action_dim, 4))
        self.scene_compat = nn.Linear(6, action_dim)
        self.k_max = int(k_max)
        self.lambda_exp_max = float(lambda_exp_max)
        self.lambda_scene_max = float(lambda_scene_max)
        self.sparse_method = sparse_method
        self.register_buffer("help_ema", torch.zeros(action_dim, num_types))
        self.register_buffer("faith_ema", torch.zeros(action_dim, num_types))
        self.register_buffer("hurt_ema", torch.zeros(action_dim, num_types))
        self._init_semantic_compat()

    def _init_semantic_compat(self) -> None:
        # Weak semantics: traffic/control reasons prefer type 5, obstacle reasons type 6/7,
        # lane/drivable reasons type 2/3/4/9. It is a guide, not hard GT.
        with torch.no_grad():
            self.reason_type_compat[:, 10] = 0.10
            for idx in [0, 1, 2, 3, 4]:
                self.reason_type_compat[idx, 5] = 0.70
            for idx in [5, 6, 9, 10, 11, 12, 13, 14]:
                self.reason_type_compat[idx, 6] = 0.65
                self.reason_type_compat[idx, 7] = 0.45
            for idx in [7, 8, 15, 16, 17, 18, 19, 20]:
                self.reason_type_compat[idx, 2] = 0.55
                self.reason_type_compat[idx, 3] = 0.50
                self.reason_type_compat[idx, 4] = 0.50
                self.reason_type_compat[idx, 9] = 0.45
            self.action_type_compat[0, 2] = 0.45  # forward/drivable
            self.action_type_compat[1, 5] = 0.55  # stop/traffic control
            self.action_type_compat[1, 6] = 0.45
            self.action_type_compat[2, 3] = 0.50  # left/lane
            self.action_type_compat[3, 4] = 0.50  # right/lane

    def forward(
        self,
        factors: FactorBatch,
        preliminary_reason_logits: torch.Tensor,
        reason_uncertainty: torch.Tensor | None,
        scene_state_logits: torch.Tensor,
    ) -> dict[str, torch.Tensor | FactorBatch]:
        b, m, d = factors.embeddings.shape
        action_count = self.action_queries.shape[0]
        aq = self.action_queries.view(1, action_count, 1, d).expand(b, -1, m, -1)
        fe = factors.embeddings.view(b, 1, m, d).expand(-1, action_count, -1, -1)
        boxes = factors.boxes.view(b, 1, m, 4).expand(-1, action_count, -1, -1)
        rel = factors.reliability_init.view(b, 1, m, 1).expand(-1, action_count, -1, -1)
        visual_score = self.score(torch.cat([aq, fe, boxes, rel], -1)).squeeze(-1)
        type_probs = factors.type_logits.softmax(-1)
        reason_probs = torch.sigmoid(preliminary_reason_logits)
        reason_type_support = reason_probs @ self.reason_type_compat.clamp_min(0)
        exp_prior_factor = torch.einsum("bt,bmt->bm", reason_type_support, type_probs)
        action_type_prior = torch.einsum("at,bmt->bam", self.action_type_compat.clamp_min(0), type_probs)
        source_onehot = torch.nn.functional.one_hot(factors.source_ids.clamp_min(0), num_classes=4).float()
        action_source_prior = torch.einsum("as,bms->bam", self.action_source_compat.clamp_min(0), source_onehot)
        action_id_bonus = torch.zeros(b, action_count, m, device=factors.embeddings.device, dtype=factors.embeddings.dtype)
        if factors.action_ids is not None:
            for action_id in range(action_count):
                action_id_bonus[:, action_id] = (factors.action_ids == action_id).float() * 0.25
        exp_prior = exp_prior_factor.view(b, 1, m) + action_type_prior + action_source_prior + action_id_bonus
        scene_prior = self.scene_compat(torch.sigmoid(scene_state_logits)).view(b, action_count, 1)
        reliability_signal = torch.relu((self.help_ema + self.faith_ema - self.hurt_ema).mean(-1))
        lambda_exp = torch.clamp(reliability_signal * self.lambda_exp_max, 0, self.lambda_exp_max)
        lambda_scene = torch.full_like(lambda_exp, self.lambda_scene_max)
        scores = visual_score + lambda_exp.view(1, action_count, 1) * exp_prior + lambda_scene.view(1, action_count, 1) * scene_prior
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
            "factor_level_exp_prior": exp_prior,
            "factor_level_explanation_guidance": True,
            "factor_group_usage": torch.bincount(factors.source_ids.flatten(), minlength=4).float(),
        }

    @torch.no_grad()
    def update_reliability(
        self,
        faith_delta: torch.Tensor | float,
        help_delta: torch.Tensor | float,
        hurt_delta: torch.Tensor | float,
        selected_type_ids: torch.Tensor | None = None,
        decay: float = 0.90,
    ) -> None:
        fd = torch.as_tensor(faith_delta, device=self.faith_ema.device, dtype=self.faith_ema.dtype)
        hd = torch.as_tensor(help_delta, device=self.help_ema.device, dtype=self.help_ema.dtype)
        ud = torch.as_tensor(hurt_delta, device=self.hurt_ema.device, dtype=self.hurt_ema.dtype)
        self.faith_ema.mul_(decay)
        self.help_ema.mul_(decay)
        self.hurt_ema.mul_(decay)
        if selected_type_ids is None:
            self.faith_ema.add_((1 - decay) * fd.mean())
            self.help_ema.add_((1 - decay) * hd.mean())
            self.hurt_ema.add_((1 - decay) * ud.mean())
            return
        type_ids = selected_type_ids.to(self.faith_ema.device).long().clamp(0, self.faith_ema.shape[1] - 1)
        action_count = self.faith_ema.shape[0]
        if fd.ndim == 0:
            fd = fd.repeat(action_count)
            hd = hd.repeat(action_count)
            ud = ud.repeat(action_count)
        fd = fd.flatten()[:action_count]
        hd = hd.flatten()[:action_count]
        ud = ud.flatten()[:action_count]
        for action_id in range(action_count):
            ids = type_ids[:, action_id].reshape(-1)
            for type_id in ids.unique():
                mask = ids == type_id
                if mask.any():
                    self.faith_ema[action_id, type_id] += (1 - decay) * fd[action_id]
                    self.help_ema[action_id, type_id] += (1 - decay) * hd[action_id]
                    self.hurt_ema[action_id, type_id] += (1 - decay) * ud[action_id]

