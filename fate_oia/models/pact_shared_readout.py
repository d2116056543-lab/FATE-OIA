from __future__ import annotations

import torch
from torch import Tensor, nn

from .acpr_sparse_ops import entmax15_bisect


class PACTSharedVisualReadout(nn.Module):
    """The source trunk's visual label readout, separated before role-specific interaction."""

    def __init__(self, dim: int = 384, action_dim: int = 4, reason_dim: int = 21) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.reason_dim = int(reason_dim)
        self.num_labels = self.action_dim + self.reason_dim
        self.label_queries = nn.Parameter(torch.randn(self.num_labels, dim) * 0.02)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.query_proj = nn.Linear(dim, dim)

    def forward(self, patch_tokens_by_layer: Tensor) -> dict[str, Tensor]:
        patch = patch_tokens_by_layer.mean(1)
        batch, patches, dim = patch.shape
        query = self.query_proj(self.label_queries).view(1, self.num_labels, 1, dim)
        key = self.key_proj(patch).view(batch, 1, patches, dim)
        score = (query * key).sum(-1) / (dim ** 0.5)
        attention = entmax15_bisect(score, dim=-1)
        nodes = torch.einsum("bln,bnd->bld", attention, self.value_proj(patch))
        return {"shared_label_nodes": nodes, "shared_label_attention": attention}


def licensed_gradient(value: Tensor, license_value: float | Tensor) -> Tensor:
    """Keep the forward value exact while scaling only the backward path."""
    license_tensor = torch.as_tensor(license_value, dtype=value.dtype, device=value.device)
    return value.detach() + license_tensor * (value - value.detach())
