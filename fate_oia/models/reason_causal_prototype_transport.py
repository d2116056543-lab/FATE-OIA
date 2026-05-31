from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


def masked_sparsemax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    if mask.dtype != torch.bool:
        mask = mask.bool()
    z = logits.masked_fill(~mask, -1e9)
    z_sorted, _ = torch.sort(z, descending=True, dim=dim)
    z_cumsum = z_sorted.cumsum(dim) - 1.0
    rhos = torch.arange(1, z.shape[dim] + 1, device=z.device, dtype=z.dtype)
    view = [1] * z.ndim
    view[dim] = -1
    rhos = rhos.view(*view)
    support = (rhos * z_sorted) > z_cumsum
    support_size = support.sum(dim=dim, keepdim=True).clamp_min(1)
    tau = z_cumsum.gather(dim, support_size.long() - 1) / support_size.to(z.dtype)
    out = torch.clamp(z - tau, min=0.0).masked_fill(~mask, 0.0)
    has_any = mask.any(dim=dim, keepdim=True)
    return torch.where(has_any, out / out.sum(dim=dim, keepdim=True).clamp_min(eps), torch.zeros_like(out))


@dataclass
class TransportConfig:
    dim: int = 384
    reason_dim: int = 21
    prototypes_per_common_reason: int = 4
    prototypes_per_tail_reason: int = 6
    tail_labels: tuple[int, ...] = (12, 9, 5, 14, 6, 11, 10, 13)
    temperature: float = 0.07
    reason_evidence_bias: float = 0.25
    real_source_bias: float = 0.35
    fallback_penalty: float = -0.25


class ReasonCausalPrototypeTransport(nn.Module):
    """Reason-causal prototype transport bottleneck T[b,r,k,m]."""

    def __init__(self, config: TransportConfig | None = None, **kwargs: Any) -> None:
        super().__init__()
        self.config = config or TransportConfig(**kwargs)
        self.dim = int(self.config.dim)
        self.reason_dim = int(self.config.reason_dim)
        self.kmax = int(max(self.config.prototypes_per_common_reason, self.config.prototypes_per_tail_reason))
        self.tail_labels = tuple(int(x) for x in self.config.tail_labels)
        self.prototypes = nn.Parameter(torch.randn(self.reason_dim, self.kmax, self.dim) * 0.02)
        mask = torch.zeros(self.reason_dim, self.kmax, dtype=torch.bool)
        for r in range(self.reason_dim):
            n = self.config.prototypes_per_tail_reason if r in self.tail_labels else self.config.prototypes_per_common_reason
            mask[r, : int(n)] = True
        self.register_buffer("prototype_mask", mask, persistent=False)
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(float(self.config.temperature))))
        self.source_bias = nn.Embedding(5, 1)
        with torch.no_grad():
            vals = torch.tensor([self.config.real_source_bias, self.config.real_source_bias * 0.85, self.config.real_source_bias * 0.75, self.config.fallback_penalty, 0.0])
            self.source_bias.weight[:, 0].copy_(vals)
        self.logit_mlp = nn.Sequential(nn.Linear(self.kmax * 2 + 5 + 2, self.dim // 2), nn.GELU(), nn.Linear(self.dim // 2, 1))
        self.reason_bias = nn.Parameter(torch.zeros(self.reason_dim))

    def _source_mass(self, T: torch.Tensor, source_type: torch.Tensor) -> torch.Tensor:
        return torch.stack([(T * (source_type == i).to(T.dtype)[:, None, None, :]).sum(dim=(2, 3)) for i in range(5)], dim=-1)

    def recompute_from_T(self, T: torch.Tensor, sim: torch.Tensor, source_type: torch.Tensor) -> dict[str, torch.Tensor]:
        proto_scores = (T * sim).sum(dim=-1)
        proto_mass = T.sum(dim=-1)
        source_mass = self._source_mass(T, source_type)
        entropy = -(T.clamp_min(1e-8) * T.clamp_min(1e-8).log()).sum(dim=(2, 3))
        reason_mass = T.sum(dim=(2, 3))
        feat = torch.cat([proto_scores, proto_mass, source_mass, entropy.unsqueeze(-1), reason_mass.unsqueeze(-1)], dim=-1)
        return {
            "proto_scores": proto_scores,
            "prototype_mass": proto_mass,
            "source_mass_by_reason": source_mass,
            "transport_entropy": entropy,
            "reason_transport_mass": reason_mass,
            "evidence_reason_logits": self.logit_mlp(feat).squeeze(-1) + self.reason_bias,
        }

    def forward(self, reason_tokens: torch.Tensor, base_reason_logits: torch.Tensor, evidence: dict[str, Any], tail_labels: tuple[int, ...] | None = None) -> dict[str, Any]:
        ev = evidence["evidence_tokens"]
        ev_mask = evidence.get("evidence_mask", torch.ones(ev.shape[:2], dtype=torch.bool, device=ev.device)).bool()
        source_type = evidence.get("evidence_source_type", torch.zeros(ev.shape[:2], dtype=torch.long, device=ev.device)).long().clamp(0, 4)
        reason_ev = evidence.get("reason_evidence_mask", torch.zeros(ev.shape[0], self.reason_dim, ev.shape[1], dtype=torch.bool, device=ev.device)).bool()
        sim = torch.einsum("bmd,rkd->brkm", F.normalize(ev, dim=-1), F.normalize(self.prototypes, dim=-1)) / self.log_temperature.exp().clamp(0.02, 1.0)
        valid = ev_mask[:, None, None, :] & self.prototype_mask[None, :, :, None]
        sim = sim + self.source_bias(source_type).squeeze(-1)[:, None, None, :] + reason_ev[:, :, None, :].to(sim.dtype) * float(self.config.reason_evidence_bias)
        T = masked_sparsemax(sim.flatten(2), valid.flatten(2), dim=-1).view(ev.shape[0], self.reason_dim, self.kmax, ev.shape[1])
        feats = self.recompute_from_T(T, sim, source_type)
        top_mass, top_flat = T.flatten(2).topk(k=min(8, T.shape[2] * T.shape[3]), dim=-1)
        return {
            "T": T,
            "sim": sim,
            **feats,
            "reliability": torch.sigmoid(feats["source_mass_by_reason"][..., :3].sum(-1) - feats["source_mass_by_reason"][..., 3]),
            "T_sparse_fraction": (T <= 1e-6).to(T.dtype).mean(),
            "top_transport_mass": top_mass,
            "top_evidence_indices": top_flat % T.shape[3],
            "top_prototype_indices": top_flat // T.shape[3],
            "transport_mass_sum_mean": feats["reason_transport_mass"].mean(),
            "transport_mass_error_max": (feats["reason_transport_mass"] - 1.0).abs().max(),
            "evidence_source_type": source_type,
        }
