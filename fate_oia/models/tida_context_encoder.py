from __future__ import annotations

import weakref
from typing import Any

import torch
from torch import nn


class TIDAContextEncoder(nn.Module):
    """Chunked history encoder that does not register a second DINO owner."""

    def __init__(
        self,
        dino_extractor: nn.Module,
        query_reader: nn.Module | None,
        context_chunk_size: int = 2,
        motion_topk: int = 12,
    ) -> None:
        super().__init__()
        if context_chunk_size < 1:
            raise ValueError("context_chunk_size must be positive")
        object.__setattr__(self, "_dino_reference", weakref.ref(dino_extractor))
        object.__setattr__(self, "_query_reader_reference", None if query_reader is None else weakref.ref(query_reader))
        self.context_chunk_size = int(context_chunk_size)
        self.motion_topk = int(motion_topk)
        if self.motion_topk < 1:
            raise ValueError("motion_topk must be positive")

    @property
    def dino_extractor(self) -> nn.Module:
        extractor = self._dino_reference()
        if extractor is None:
            raise RuntimeError("shared DINO owner has been released")
        return extractor

    @property
    def query_reader(self) -> nn.Module | None:
        reference = self._query_reader_reference
        return None if reference is None else reference()

    def forward(
        self,
        context_images: torch.Tensor,
        action_nodes: torch.Tensor,
        predicate_tokens: torch.Tensor,
        predicate_identities: torch.Tensor,
        *,
        canonicalize_horizontal_flip: bool = False,
    ) -> dict[str, Any]:
        if self.query_reader is None:
            raise RuntimeError("query_reader is required for context encoding")
        if context_images.ndim != 5:
            raise ValueError("context_images must be [B,T,3,H,W]")
        batch, frames, channels, height, width = context_images.shape
        tokens, attentions, region_masses = [], [], []
        action_patch_tokens, action_patch_xy, action_patch_weights = [], [], []
        for start in range(0, frames, self.context_chunk_size):
            stop = min(start + self.context_chunk_size, frames)
            images = context_images[:, start:stop].reshape(-1, channels, height, width)
            field = self.dino_extractor.forward_at_resolution(images, expected_hw=(height, width))
            if canonicalize_horizontal_flip:
                grid_height, grid_width = field["grid_hw"]
                patches = field["patch_tokens_by_layer"]
                field["patch_tokens_by_layer"] = patches.view(
                    patches.shape[0], patches.shape[1], grid_height, grid_width, patches.shape[-1]
                ).flip(3).flatten(2, 3)
            repeats = stop - start
            read = self.query_reader(
                field["patch_tokens_by_layer"],
                action_nodes[:, None].expand(-1, repeats, -1, -1).reshape(-1, action_nodes.shape[1], action_nodes.shape[2]),
                predicate_tokens[:, None].expand(-1, repeats, -1, -1).reshape(-1, predicate_tokens.shape[1], predicate_tokens.shape[2]),
                predicate_identities,
                grid_hw=field["grid_hw"],
            )
            tokens.append(read["query_tokens"].reshape(batch, repeats, -1, action_nodes.shape[-1]))
            attentions.append(read["query_attention"].reshape(batch, repeats, -1, field["patch_tokens_by_layer"].shape[2]))
            region_masses.append(read["query_region_mass"].reshape(batch, repeats, -1, 5))
            action_attention = read["query_attention"][:, : action_nodes.shape[1]]
            topk = min(self.motion_topk, action_attention.shape[-1])
            top_weight, top_index = action_attention.topk(topk, dim=-1)
            top_weight = top_weight / top_weight.sum(-1, keepdim=True).clamp_min(1e-8)
            patch_field = field["patch_tokens_last"]
            gathered = torch.gather(
                patch_field[:, None].expand(-1, action_nodes.shape[1], -1, -1),
                2,
                top_index[..., None].expand(-1, -1, -1, patch_field.shape[-1]),
            )
            grid_height, grid_width = field["grid_hw"]
            yy, xx = torch.meshgrid(
                torch.linspace(-1.0, 1.0, grid_height, device=patch_field.device, dtype=patch_field.dtype),
                torch.linspace(-1.0, 1.0, grid_width, device=patch_field.device, dtype=patch_field.dtype),
                indexing="ij",
            )
            coordinates = torch.stack((xx, yy), dim=-1).flatten(0, 1)
            gathered_xy = coordinates[top_index]
            action_patch_tokens.append(gathered.reshape(batch, repeats, action_nodes.shape[1], topk, -1))
            action_patch_xy.append(gathered_xy.reshape(batch, repeats, action_nodes.shape[1], topk, 2))
            action_patch_weights.append(top_weight.reshape(batch, repeats, action_nodes.shape[1], topk))
        return {
            "history_query_tokens": torch.cat(tokens, dim=1),
            "history_query_attention": torch.cat(attentions, dim=1),
            "history_query_region_mass": torch.cat(region_masses, dim=1),
            "history_action_patch_tokens": torch.cat(action_patch_tokens, dim=1),
            "history_action_patch_xy": torch.cat(action_patch_xy, dim=1),
            "history_action_patch_weight": torch.cat(action_patch_weights, dim=1),
            "history_grid_hw": (height // self.dino_extractor.patch_size, width // self.dino_extractor.patch_size),
        }
