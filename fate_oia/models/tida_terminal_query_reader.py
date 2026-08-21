from __future__ import annotations

import math

import torch
from torch import nn

from .acpr_sparse_ops import entmax15_bisect


class TIDATerminalQueryReader(nn.Module):
    """Read a frozen DINO field with target-frame action/predicate queries."""

    def __init__(
        self,
        dim: int = 384,
        num_actions: int = 4,
        num_predicates: int = 32,
        layer_ids: tuple[int, ...] = (3, 7, 11),
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_actions = int(num_actions)
        self.num_predicates = int(num_predicates)
        self.layer_ids = tuple(int(layer) for layer in layer_ids)
        self.layer_id_to_index = {layer: index for index, layer in enumerate(self.layer_ids)}
        self.read_order = tuple(reversed(self.layer_ids))
        self.query_norm = nn.LayerNorm(dim)
        self.query_proj = nn.ModuleDict({str(layer): nn.Linear(dim, dim) for layer in self.layer_ids})
        self.key_proj = nn.ModuleDict({str(layer): nn.Linear(dim, dim) for layer in self.layer_ids})
        self.value_proj = nn.ModuleDict({str(layer): nn.Linear(dim, dim) for layer in self.layer_ids})
        self.update_norm = nn.ModuleDict({str(layer): nn.LayerNorm(dim) for layer in self.layer_ids})
        self.residual_gain_raw = nn.Parameter(torch.zeros(len(self.layer_ids)))

    @staticmethod
    def _region_masks(grid_hw: tuple[int, int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        height, width = grid_hw
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, height, device=device, dtype=dtype),
            torch.linspace(0, 1, width, device=device, dtype=dtype),
            indexing="ij",
        )
        masks = torch.stack(
            [
                xx < 0.40,
                xx > 0.60,
                ((xx >= 0.35) & (xx <= 0.65) & (yy >= 0.35)),
                yy < 0.45,
                torch.ones_like(xx, dtype=torch.bool),
            ],
            dim=0,
        ).to(dtype=dtype).flatten(1)
        # These are spatial membership masks. Keeping their natural [0, 1]
        # scale makes A @ R an interpretable fraction of attention mass.
        return masks

    def forward(
        self,
        patch_tokens_by_layer: torch.Tensor,
        action_nodes: torch.Tensor,
        predicate_tokens: torch.Tensor,
        predicate_identities: torch.Tensor,
        *,
        grid_hw: tuple[int, int],
    ) -> dict[str, torch.Tensor | tuple[int, ...]]:
        if patch_tokens_by_layer.ndim != 4:
            raise ValueError("patch_tokens_by_layer must be [B,S,N,D]")
        batch, layers, patches, dim = patch_tokens_by_layer.shape
        if layers != len(self.layer_ids) or dim != self.dim:
            raise ValueError("DINO layer field does not match reader contract")
        if patches != grid_hw[0] * grid_hw[1]:
            raise ValueError("grid_hw does not match patch count")
        if action_nodes.shape != (batch, self.num_actions, dim):
            raise ValueError("action_nodes shape mismatch")
        if predicate_tokens.shape != (batch, self.num_predicates, dim):
            raise ValueError("predicate_tokens shape mismatch")
        if predicate_identities.shape != (self.num_predicates, dim):
            raise ValueError("predicate_identities shape mismatch")

        predicate_queries = self.query_norm(
            predicate_tokens.detach() + predicate_identities.to(predicate_tokens)[None]
        )
        query = torch.cat([action_nodes.detach(), predicate_queries], dim=1)
        layer_attention: list[torch.Tensor] = []
        layer_update_norms: list[torch.Tensor] = []
        for layer in self.read_order:
            field_index = self.layer_id_to_index[layer]
            field = patch_tokens_by_layer[:, field_index]
            q = self.query_proj[str(layer)](self.query_norm(query))
            key = self.key_proj[str(layer)](field)
            value = self.value_proj[str(layer)](field)
            score = torch.einsum("bqd,bnd->bqn", q, key) / math.sqrt(dim)
            attention = entmax15_bisect(score, dim=-1)
            update = torch.einsum("bqn,bnd->bqd", attention, value)
            gain_index = self.layer_id_to_index[layer]
            gain = 0.25 * torch.sigmoid(self.residual_gain_raw[gain_index])
            query = query + gain * self.update_norm[str(layer)](update)
            layer_attention.append(attention)
            layer_update_norms.append(update.norm(dim=-1).mean())

        final_attention = layer_attention[-1]
        masks = self._region_masks(grid_hw, final_attention.device, final_attention.dtype)
        region_mass = torch.einsum("bqn,rn->bqr", final_attention, masks)
        return {
            "query_tokens": query,
            "query_attention": final_attention,
            "query_attention_by_layer": torch.stack(layer_attention, dim=2),
            "query_region_mass": region_mass,
            "layer_update_norm": torch.stack(layer_update_norms),
            "layer_order": self.read_order,
        }
