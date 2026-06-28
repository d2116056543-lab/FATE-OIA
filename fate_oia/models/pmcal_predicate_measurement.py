from __future__ import annotations

from pathlib import Path
import hashlib
import torch
from torch import nn
import yaml

from .acpr_sparse_ops import entmax15_bisect


def _hash_embedding(text: str, dim: int, device: torch.device | None = None) -> torch.Tensor:
    h = hashlib.sha256(text.encode("utf-8")).digest()
    vals = torch.tensor([(h[i % len(h)] / 255.0) * 2.0 - 1.0 for i in range(dim)], dtype=torch.float32, device=device)
    return vals / vals.norm().clamp_min(1e-6)


class PMCalPredicateMeasurementLayer(nn.Module):
    def __init__(
        self,
        scene_config: str = "configs/acpr_scene_predicates.yaml",
        text_prompt_config: str | None = None,
        dim: int = 384,
        num_predicates: int = 32,
        num_layers: int = 3,
        text_dim: int = 384,
        entmax_alpha: float = 1.5,
        text_prior_max_weight: float = 0.25,
    ) -> None:
        super().__init__()
        data = yaml.safe_load(Path(scene_config).read_text(encoding="utf-8")) or {}
        self.predicates = list(data.get("predicates", []))
        self.predicate_names = [str(p["name"]) for p in self.predicates]
        self.num_predicates = max(num_predicates, len(self.predicate_names))
        if self.num_predicates != len(self.predicate_names):
            self.predicate_names += [f"unused_predicate_{i}" for i in range(len(self.predicate_names), self.num_predicates)]
        self.dim = dim
        self.text_prior_max_weight = float(text_prior_max_weight)
        text = torch.stack([_hash_embedding(name, dim) for name in self.predicate_names], 0)
        self.register_buffer("text_embeddings", text)
        self.predicate_queries = nn.Parameter(text.clone() + 0.01 * torch.randn_like(text))
        self.layer_logits = nn.Parameter(torch.zeros(self.num_predicates, num_layers))
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.visual_head = nn.Linear(dim, 1)
        self.rho_head = nn.Sequential(nn.LayerNorm(dim + 3), nn.Linear(dim + 3, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1))
        self.text_scale = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        patch_tokens_by_layer: torch.Tensor,
        region_masks: dict[str, torch.Tensor],
        text_prompt_embeddings: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor | dict]:
        b, s, n, d = patch_tokens_by_layer.shape
        layer_weights = torch.softmax(self.layer_logits, dim=-1)
        tokens = torch.einsum("ms,bsnd->bmnd", layer_weights, patch_tokens_by_layer)
        q = self.query_proj(self.predicate_queries).view(1, self.num_predicates, 1, d)
        k = self.key_proj(tokens)
        v = self.value_proj(tokens)
        score = (q * k).sum(-1) / (d ** 0.5)
        for j, pred in enumerate(self.predicates):
            region = str(pred.get("region", ""))
            if region in region_masks:
                prior = region_masks[region].to(score.device, score.dtype).clamp_min(1e-4)
                score[:, j] = score[:, j] + prior.log().view(1, -1)
        attn = entmax15_bisect(score, dim=-1)
        predicate_tokens = torch.einsum("bmn,bmnd->bmd", attn, v)
        predicate_logits_visual = self.visual_head(predicate_tokens).squeeze(-1)
        q_pred_visual = torch.sigmoid(predicate_logits_visual)
        text_sim = torch.einsum("bmd,md->bm", torch.nn.functional.normalize(predicate_tokens, dim=-1), self.text_embeddings.to(predicate_tokens.device))
        text_logit = self.text_scale.clamp(0.0, 4.0) * text_sim
        q_pred_text = torch.sigmoid(text_logit)
        text_weight = min(self.text_prior_max_weight, 0.25)
        q_pred = torch.sigmoid(predicate_logits_visual + text_weight * text_logit)
        entropy = -(attn.clamp_min(1e-8).log() * attn).sum(-1)
        margin = (q_pred - 0.5).abs()
        support = (attn > 1e-4).float().mean(-1)
        rho_in = torch.cat([predicate_tokens, entropy.unsqueeze(-1), margin.unsqueeze(-1), support.unsqueeze(-1)], dim=-1)
        rho_pred = torch.sigmoid(self.rho_head(rho_in).squeeze(-1))
        region_mass = []
        for name in ["front_center", "left_corridor", "right_corridor", "upper_traffic_region", "bottom_drivable_region"]:
            mask = region_masks.get(name)
            if mask is None:
                region_mass.append(torch.zeros(b, self.num_predicates, device=attn.device, dtype=attn.dtype))
            else:
                region_mass.append((attn * mask.to(attn.device, attn.dtype).view(1, 1, -1)).sum(-1))
        return {
            "predicate_tokens": predicate_tokens,
            "predicate_attention": attn,
            "predicate_logits_visual": predicate_logits_visual,
            "predicate_logits": predicate_logits_visual,
            "q_pred_visual": q_pred_visual,
            "q_pred_text": q_pred_text,
            "q_pred": q_pred,
            "q_pred_fair": q_pred,
            "rho_pred": rho_pred,
            "layer_weights": layer_weights,
            "predicate_layer_weights": layer_weights,
            "region_mass": torch.stack(region_mass, dim=-1),
            "attention_entropy": entropy,
            "predicate_measurement_stats": {
                "q_pred_mean": float(q_pred.detach().mean().cpu()),
                "rho_pred_mean": float(rho_pred.detach().mean().cpu()),
                "attention_entropy_mean": float(entropy.detach().mean().cpu()),
                "text_weight": text_weight,
            },
        }
