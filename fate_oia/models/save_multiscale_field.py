from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn


SAVE_SELECTED_LAYERS = (3, 7, 11)
SAVE_GRID_HW = (45, 80)
SAVE_PATCH_TOKENS = SAVE_GRID_HW[0] * SAVE_GRID_HW[1]


def build_2d_sincos_position_embedding(
    grid_hw: tuple[int, int] = SAVE_GRID_HW,
    dim: int = 384,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Build a deterministic row-major 2D sine-cosine position table."""
    height, width = (int(grid_hw[0]), int(grid_hw[1]))
    dim = int(dim)
    if height <= 0 or width <= 0:
        raise ValueError(f"grid_hw must be positive, got {grid_hw}")
    if dim <= 0 or dim % 4 != 0:
        raise ValueError(f"2D sine-cosine encoding requires dim divisible by 4, got {dim}")
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise TypeError(f"position encoding dtype must be floating point, got {dtype}")

    # Each axis owns half of the channels: its first half is sine and its
    # second half is cosine. Flattening [row, column] matches DINO patches.
    quarter_dim = dim // 4
    frequencies = torch.arange(quarter_dim, device=device, dtype=torch.float32)
    frequencies = torch.exp(-math.log(10000.0) * frequencies / quarter_dim)
    rows = torch.arange(height, device=device, dtype=torch.float32)
    columns = torch.arange(width, device=device, dtype=torch.float32)

    row_angles = rows[:, None] * frequencies[None, :]
    column_angles = columns[:, None] * frequencies[None, :]
    row_encoding = torch.cat((row_angles.sin(), row_angles.cos()), dim=-1)
    column_encoding = torch.cat((column_angles.sin(), column_angles.cos()), dim=-1)
    encoding = torch.cat(
        (
            row_encoding[:, None, :].expand(height, width, -1),
            column_encoding[None, :, :].expand(height, width, -1),
        ),
        dim=-1,
    )
    return encoding.reshape(1, height * width, dim).to(dtype=dtype)


class SAVEMultiscaleField(nn.Module):
    """Construct full-resolution SAVE global and detail visual fields.

    The module consumes the single stacked output of the frozen DINO encoder.
    It deliberately performs no image encoding, token selection, compression,
    or caching; callers own the one-call direct-image encoding contract.
    """

    def __init__(
        self,
        dim: int = 384,
        input_dim: int = 384,
        grid_hw: tuple[int, int] = SAVE_GRID_HW,
        selected_layers: tuple[int, ...] = SAVE_SELECTED_LAYERS,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.input_dim = int(input_dim)
        self.grid_hw = (int(grid_hw[0]), int(grid_hw[1]))
        self.selected_layers = tuple(int(layer) for layer in selected_layers)
        if self.selected_layers != SAVE_SELECTED_LAYERS:
            raise ValueError(
                "SAVE multiscale fields require DINO layers (3, 7, 11), "
                f"got {self.selected_layers}"
            )
        if self.dim <= 0 or self.input_dim <= 0:
            raise ValueError("dim and input_dim must be positive")
        if self.grid_hw[0] <= 0 or self.grid_hw[1] <= 0:
            raise ValueError(f"grid_hw must be positive, got {self.grid_hw}")

        self.num_tokens = self.grid_hw[0] * self.grid_hw[1]
        self.global_projection = nn.Linear(self.input_dim, self.dim)
        self.detail_projection_3 = nn.Linear(self.input_dim, self.dim)
        self.detail_projection_7 = nn.Linear(self.input_dim, self.dim)
        self.global_norm = nn.LayerNorm(self.dim)
        self.detail_norm = nn.LayerNorm(self.dim)
        self.detail_output_projection = nn.Linear(self.dim, self.dim)
        nn.init.zeros_(self.detail_output_projection.weight)
        nn.init.zeros_(self.detail_output_projection.bias)
        self.register_buffer(
            "detail_position",
            build_2d_sincos_position_embedding(self.grid_hw, self.dim),
            persistent=True,
        )

    @property
    def projection_3(self) -> nn.Linear:
        return self.detail_projection_3

    @property
    def projection_7(self) -> nn.Linear:
        return self.detail_projection_7

    @property
    def projection_11(self) -> nn.Linear:
        return self.global_projection

    @property
    def detail_residual_projection(self) -> nn.Linear:
        return self.detail_output_projection

    def _validate_input(self, patch_tokens_by_layer: Tensor) -> Tensor:
        if not isinstance(patch_tokens_by_layer, Tensor):
            raise TypeError("patch_tokens_by_layer must be a tensor")
        if patch_tokens_by_layer.ndim != 4:
            raise ValueError(
                "SAVE multiscale fields require a rank-4 "
                f"[B,3,{self.num_tokens},{self.input_dim}] tensor"
            )
        if patch_tokens_by_layer.shape[1] != len(SAVE_SELECTED_LAYERS):
            raise ValueError("SAVE multiscale fields require exactly 3 selected layers")
        if patch_tokens_by_layer.shape[2] != self.num_tokens:
            raise ValueError(
                f"SAVE multiscale fields require exactly {self.num_tokens} patch tokens"
            )
        if patch_tokens_by_layer.shape[3] != self.input_dim:
            raise ValueError(
                f"SAVE multiscale fields require channel width {self.input_dim}"
            )
        return patch_tokens_by_layer

    def forward(
        self,
        patch_tokens_by_layer: Tensor | Mapping[str, Any],
    ) -> dict[str, Tensor]:
        if isinstance(patch_tokens_by_layer, Mapping):
            try:
                patch_tokens_by_layer = patch_tokens_by_layer["patch_tokens_by_layer"]
            except KeyError as exc:
                raise KeyError("field is missing 'patch_tokens_by_layer'") from exc
        patch_tokens_by_layer = self._validate_input(patch_tokens_by_layer)

        layer_3 = self.detail_projection_3(patch_tokens_by_layer[:, 0])
        layer_7 = self.detail_projection_7(patch_tokens_by_layer[:, 1])
        layer_11 = self.global_projection(patch_tokens_by_layer[:, 2])
        position = self.detail_position.to(device=layer_3.device, dtype=layer_3.dtype)

        global_field = self.global_norm(layer_11)
        detail_field = self.detail_norm(layer_3 + layer_7 + position)
        # Auxiliary supervision can train this zero-initialized path without
        # changing the canonical detail field consumed by downstream inquiry.
        detail_residual = self.detail_output_projection(detail_field)
        return {
            "global_field": global_field,
            "detail_field": detail_field,
            "detail_position": self.detail_position,
            "detail_residual": detail_residual,
        }


SAVEMultiScaleField = SAVEMultiscaleField


__all__ = [
    "SAVE_GRID_HW",
    "SAVE_PATCH_TOKENS",
    "SAVE_SELECTED_LAYERS",
    "SAVEMultiScaleField",
    "SAVEMultiscaleField",
    "build_2d_sincos_position_embedding",
]
