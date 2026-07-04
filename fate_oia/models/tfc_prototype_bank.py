from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class TFCPrototypeBank(nn.Module):
    def __init__(self, num_factors: int, dim: int = 384, num_prototypes: int = 4) -> None:
        super().__init__()
        self.num_factors = int(num_factors)
        self.num_prototypes = int(num_prototypes)
        self.dim = int(dim)
        self.prototypes = nn.Parameter(torch.randn(self.num_factors, self.num_prototypes, self.dim) * 0.02)
        self.factor_query_delta = nn.Parameter(torch.zeros(self.num_factors, self.dim))

    def forward(self) -> dict[str, torch.Tensor]:
        prototypes = F.normalize(self.prototypes, dim=-1)
        factor_queries = F.normalize(prototypes.mean(dim=1) + self.factor_query_delta, dim=-1)
        return {"prototypes": prototypes, "factor_queries": factor_queries}


def prototype_consistency_loss(
    h_factor: torch.Tensor,
    q_factor: torch.Tensor,
    prototypes: torch.Tensor,
    native_similarity: torch.Tensor,
    conflict_matrix: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    h = F.normalize(h_factor.float(), dim=-1)
    q = F.normalize(q_factor.float(), dim=-1)
    p = F.normalize(prototypes.float(), dim=-1)
    assign = torch.softmax(torch.einsum("bfd,fkd->bfk", h, p), dim=-1)
    entropy = -(assign * (assign.clamp_min(1e-8).log())).sum(-1).mean()
    compact = (1.0 - torch.einsum("bfd,fd->bf", h, q)).mean()
    sim_pred = torch.matmul(q, q.t()).clamp(-1, 1)
    sim_loss = F.mse_loss(sim_pred, native_similarity.to(sim_pred.device, sim_pred.dtype).clamp(-1, 1))
    conflict = conflict_matrix.to(sim_pred.device, sim_pred.dtype) > 0
    if bool(conflict.any()):
        conflict_sep = F.relu(sim_pred[conflict] + 0.2).mean()
    else:
        conflict_sep = sim_pred.new_tensor(0.0)
    loss = compact + 0.05 * entropy + 0.05 * sim_loss + 0.10 * conflict_sep
    stats = {
        "assignment_entropy": float(entropy.detach().cpu()),
        "prototype_compactness": float(compact.detach().cpu()),
        "native_similarity_consistency": float(sim_loss.detach().cpu()),
        "conflict_separation_loss": float(conflict_sep.detach().cpu()),
        "left_right_mirror_score": float((sim_pred.mean()).detach().cpu()),
    }
    return loss, stats
