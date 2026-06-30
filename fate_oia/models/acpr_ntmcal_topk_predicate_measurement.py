from __future__ import annotations

import math

import torch
from torch import nn

from .acpr_sparse_ops import entmax15_bisect
from .acpr_ntmcal_predicate_bank import NativePredicateBank
from .acpr_ntmcal_text_atoms import NativeTextAtomEncoder


class NativeTextTopKPredicateMeasurement(nn.Module):
    def __init__(
        self,
        predicate_bank: NativePredicateBank,
        atom_encoder: NativeTextAtomEncoder,
        dim: int = 384,
        selected_layers: tuple[int, ...] = (3, 7, 11),
        topk: int = 64,
        temperature_min: float = 0.2,
        temperature_max: float = 5.0,
    ) -> None:
        super().__init__()
        self.predicate_bank = predicate_bank
        self.atom_encoder = atom_encoder
        self.dim = dim
        self.selected_layers = selected_layers
        self.topk = int(topk)
        self.temperature_min = float(temperature_min)
        self.temperature_max = float(temperature_max)
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.logit_head = nn.Linear(dim, 1)
        self.rho_head = nn.Sequential(nn.Linear(dim + 4, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1))
        self.layer_bias = nn.Parameter(torch.zeros(len(predicate_bank.specs), len(selected_layers)))
        self.log_temperature = nn.Parameter(torch.zeros(len(predicate_bank.specs)))

    def _region_prior(self, region_masks: dict[str, torch.Tensor] | None, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        n = 3600
        rows = []
        for spec in self.predicate_bank.specs:
            if region_masks and spec.region in region_masks:
                mask = region_masks[spec.region].to(device=device, dtype=dtype).flatten()
            elif region_masks and spec.region == "global":
                mask = torch.ones(n, device=device, dtype=dtype)
            else:
                mask = torch.ones(n, device=device, dtype=dtype)
            rows.append(mask.clamp_min(1e-4).log())
        return torch.stack(rows, dim=0)

    def forward(self, patch_tokens_by_layer: torch.Tensor, region_masks: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor | dict]:
        b, l, n, d = patch_tokens_by_layer.shape
        p = len(self.predicate_bank.specs)
        if n != 3600:
            raise ValueError(f"expected 3600 patches, got {n}")
        queries = self.query_proj(self.atom_encoder.encode_predicates(self.predicate_bank.specs))
        keys = self.key_proj(patch_tokens_by_layer)
        values = self.value_proj(patch_tokens_by_layer)
        score = torch.einsum("pd,blnd->bpln", queries, keys) / math.sqrt(d)
        score = score + self.layer_bias.view(1, p, l, 1)
        score = score + self._region_prior(region_masks, score.device, score.dtype).view(1, p, 1, n)
        flat_score = score.flatten(start_dim=2)
        k = min(self.topk, flat_score.shape[-1])
        topk_score, topk_idx = flat_score.topk(k=k, dim=-1)

        # Gather only selected predicate evidence tokens. Do not expand values to [B,P,L*N,D].
        flat_values = values.reshape(b, l * n, d)
        batch_idx = torch.arange(b, device=flat_values.device).view(b, 1, 1)
        topk_value = flat_values[batch_idx, topk_idx]

        tau = self.log_temperature.exp().clamp(self.temperature_min, self.temperature_max).view(1, p, 1)
        attn = entmax15_bisect(topk_score / tau, dim=-1)
        h_pred = torch.einsum("bpk,bpkd->bpd", attn, topk_value)
        q_logit = self.logit_head(h_pred).squeeze(-1)
        q_pred = torch.sigmoid(q_logit)
        entropy = -(attn.clamp_min(1e-8).log() * attn).sum(-1)
        layer_idx = torch.div(topk_idx, n, rounding_mode="floor")
        layer_consistency = torch.nn.functional.one_hot(layer_idx.clamp_max(l - 1), num_classes=l).float().mean(-2).max(-1).values
        region_mass = attn.sum(-1)
        rho_in = torch.cat(
            [h_pred, q_logit.abs().unsqueeze(-1), entropy.unsqueeze(-1), region_mass.unsqueeze(-1), layer_consistency.unsqueeze(-1)],
            dim=-1,
        )
        rho_pred = torch.sigmoid(self.rho_head(rho_in).squeeze(-1))
        stats = {
            "predicate_support_size": float((attn > 1e-5).float().sum(-1).mean().detach().cpu()),
            "predicate_attention_entropy": float(entropy.mean().detach().cpu()),
            "predicate_topk": int(k),
            "predicate_topk_unique_fraction": float(torch.unique(topk_idx.detach()).numel() / max(topk_idx.numel(), 1)),
            "predicate_region_mass_mean": float(region_mass.mean().detach().cpu()),
            "predicate_q_mean": float(q_pred.mean().detach().cpu()),
            "predicate_q_min": float(q_pred.min().detach().cpu()),
            "predicate_q_max": float(q_pred.max().detach().cpu()),
            "predicate_rho_mean": float(rho_pred.mean().detach().cpu()),
            "predicate_rho_min": float(rho_pred.min().detach().cpu()),
            "predicate_rho_max": float(rho_pred.max().detach().cpu()),
            "dense_bpnd_materialized": False,
        }
        return {
            "predicate_q": q_pred,
            "predicate_rho": rho_pred,
            "predicate_tokens": h_pred,
            "predicate_topk_indices": topk_idx,
            "predicate_topk_attention": attn,
            "predicate_region_mass": region_mass,
            "predicate_stats": stats,
        }
