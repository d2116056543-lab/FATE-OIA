from __future__ import annotations

import weakref
from typing import Any

import torch
from torch import nn


class TIDAContextEncoder(nn.Module):
    """Chunked history encoder that does not register a second DINO owner."""

    def __init__(self, dino_extractor: nn.Module, query_reader: nn.Module | None, context_chunk_size: int = 2) -> None:
        super().__init__()
        if context_chunk_size < 1:
            raise ValueError("context_chunk_size must be positive")
        object.__setattr__(self, "_dino_reference", weakref.ref(dino_extractor))
        object.__setattr__(self, "_query_reader_reference", None if query_reader is None else weakref.ref(query_reader))
        self.context_chunk_size = int(context_chunk_size)

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
        return {
            "history_query_tokens": torch.cat(tokens, dim=1),
            "history_query_attention": torch.cat(attentions, dim=1),
            "history_query_region_mass": torch.cat(region_masses, dim=1),
            "history_grid_hw": (height // self.dino_extractor.patch_size, width // self.dino_extractor.patch_size),
        }
