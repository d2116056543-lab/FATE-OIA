from __future__ import annotations

from pathlib import Path

import torch
import yaml
from torch import nn

from .acpr_grounded_evidence_memory import ACPREvidenceMemoryAugmenter
from .acpr_sparse_ops import entmax15_bisect


class ACPRScenePredicateHead(nn.Module):
    def __init__(self, scene_config: str | Path, dim: int = 384, num_layers: int = 3) -> None:
        super().__init__()
        data = yaml.safe_load(Path(scene_config).read_text(encoding="utf-8")) or {}
        self.predicates = list(data.get("predicates", []))
        self.num_predicates = len(self.predicates)
        if self.num_predicates < 32:
            raise ValueError("ACPRScenePredicateHead requires at least 32 predicates")
        self.predicate_queries = nn.Parameter(torch.randn(self.num_predicates, dim) * 0.02)
        self.layer_logits = nn.Parameter(torch.zeros(self.num_predicates, num_layers))
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.logit_head = nn.Linear(dim, 1)
        self.temperature = nn.Parameter(torch.ones(self.num_predicates))
        self.evidence_augmenter = ACPREvidenceMemoryAugmenter(dim=dim, max_delta=0.20, num_heads=4)
        self.evidence_out_proj = self.evidence_augmenter.evidence_out_proj

    @property
    def names(self) -> list[str]:
        return [str(p["name"]) for p in self.predicates]

    def forward(
        self,
        patch_tokens_by_layer: torch.Tensor,
        region_masks: dict[str, torch.Tensor] | None = None,
        evidence_tokens: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | dict]:
        b, s, n, d = patch_tokens_by_layer.shape
        weights = torch.softmax(self.layer_logits, dim=-1)
        tokens = torch.einsum("ls,bsnd->blnd", weights, patch_tokens_by_layer)
        q = self.query_proj(self.predicate_queries).view(1, self.num_predicates, 1, d)
        k = self.key_proj(tokens)
        v = self.value_proj(tokens)
        score = (q * k).sum(-1) / (d ** 0.5)
        if region_masks is not None:
            for j, pred in enumerate(self.predicates):
                region = str(pred.get("region", ""))
                if region in region_masks:
                    prior = region_masks[region].to(score.device, score.dtype).clamp_min(1e-4)
                    score[:, j] = score[:, j] + prior.log().view(1, -1)
        tau = self.temperature.clamp(0.2, 5.0).view(1, -1, 1)
        attn = entmax15_bisect(score / tau, dim=-1)
        predicate_tokens = torch.einsum("bln,blnd->bld", attn, v)
        predicate_tokens_patch = predicate_tokens
        if evidence_tokens is not None:
            predicate_tokens, predicate_evidence_context, predicate_evidence_attention = self.evidence_augmenter(predicate_tokens_patch, evidence_tokens)
            predicate_evidence_delta = predicate_tokens - predicate_tokens_patch
        else:
            predicate_evidence_context = torch.zeros_like(predicate_tokens)
            predicate_evidence_delta = torch.zeros_like(predicate_tokens)
            predicate_evidence_attention = predicate_tokens.new_zeros(b, self.num_predicates, 0)
        logits = self.logit_head(predicate_tokens).squeeze(-1)
        probs = torch.sigmoid(logits)
        support = (attn > 1e-4).float().sum(-1).mean().detach()
        entropy = (-(attn.clamp_min(1e-9).log() * attn).sum(-1)).mean().detach()
        stats = {"predicate_support_size": float(support.cpu()), "predicate_attention_entropy": float(entropy.cpu())}
        return {
            "predicate_tokens": predicate_tokens,
            "predicate_tokens_patch": predicate_tokens_patch,
            "predicate_tokens_evidence_context": predicate_evidence_context,
            "predicate_evidence_attention": predicate_evidence_attention,
            "predicate_evidence_delta_norm": predicate_evidence_delta.norm(dim=-1).mean(),
            "predicate_logits": logits,
            "predicate_probs": probs,
            "predicate_attention": attn,
            "predicate_layer_weights": weights,
            "predicate_stats": stats,
        }
