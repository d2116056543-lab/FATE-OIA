from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MOSAICVisualPyramid(nn.Module):
    """Project the selected DINO layers into the three MOSAIC spatial scales."""

    def __init__(
        self,
        input_dim: int = 384,
        output_dim: int = 384,
        grid_hw: tuple[int, int] = (45, 80),
    ) -> None:
        super().__init__()
        if type(input_dim) is not int or input_dim <= 0:
            raise ValueError("input_dim must be a positive integer")
        if type(output_dim) is not int or output_dim <= 0:
            raise ValueError("output_dim must be a positive integer")
        if grid_hw != (45, 80):
            raise ValueError("MOSAICVisualPyramid requires grid_hw=(45,80)")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.grid_hw = grid_hw
        self.proj_hi = nn.Conv2d(input_dim, output_dim, kernel_size=1)
        self.proj_mid = nn.Conv2d(input_dim, output_dim, kernel_size=1)
        self.proj_ctx = nn.Conv2d(input_dim, output_dim, kernel_size=1)
        self.local_residual = nn.Conv2d(
            output_dim,
            output_dim,
            kernel_size=3,
            padding=1,
            groups=output_dim,
            bias=False,
        )
        nn.init.zeros_(self.local_residual.weight)

    def forward(self, patch_tokens_by_layer: torch.Tensor) -> dict[str, torch.Tensor | tuple[int, int]]:
        expected_shape = (3, self.grid_hw[0] * self.grid_hw[1], self.input_dim)
        if patch_tokens_by_layer.ndim != 4 or tuple(patch_tokens_by_layer.shape[1:]) != expected_shape:
            raise ValueError(
                "MOSAICVisualPyramid expects [B,3,3600,input_dim], "
                f"got {tuple(patch_tokens_by_layer.shape)}"
            )
        if not patch_tokens_by_layer.is_floating_point():
            raise ValueError("MOSAICVisualPyramid requires floating-point DINO tokens")

        batch_size = patch_tokens_by_layer.shape[0]
        height, width = self.grid_hw
        layers_2d = patch_tokens_by_layer.transpose(2, 3).reshape(
            batch_size,
            3,
            self.input_dim,
            height,
            width,
        )
        layer3, layer7, layer11 = layers_2d.unbind(dim=1)

        high = self.proj_hi(layer3)
        high = high + self.local_residual(high)
        middle = self.proj_mid(F.adaptive_avg_pool2d(layer7, (23, 40)))
        context = self.proj_ctx(F.adaptive_avg_pool2d(layer11, (12, 20)))
        return {
            "F_hi": high,
            "F_mid": middle,
            "F_ctx": context,
            "grid_hw": self.grid_hw,
        }
