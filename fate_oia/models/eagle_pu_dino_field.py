from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

import utils
import vision_transformer as vits


@dataclass
class EaglePUFieldOutput:
    patch_tokens_by_layer: torch.Tensor
    cls_tokens_by_layer: torch.Tensor
    grid_hw: tuple[int, int]
    original_tokens: int


class MockDinoField(nn.Module):
    def __init__(self, dim: int = 384, selected_layers: tuple[int, ...] = (3, 7, 11), grid_hw: tuple[int, int] = (45, 80)) -> None:
        super().__init__()
        self.dim = dim
        self.selected_layers = tuple(selected_layers)
        self.grid_hw = grid_hw
        self.proj = nn.Linear(3, dim)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor | tuple[int, int] | int]:
        pooled = F.adaptive_avg_pool2d(images, self.grid_hw).flatten(2).transpose(1, 2)
        base = self.proj(pooled)
        layers = []
        cls = []
        for i, _layer in enumerate(self.selected_layers):
            token = base + (i + 1) * 0.01
            layers.append(token)
            cls.append(token.mean(1))
        return {
            "patch_tokens_by_layer": torch.stack(layers, dim=1),
            "cls_tokens_by_layer": torch.stack(cls, dim=1),
            "grid_hw": self.grid_hw,
            "original_tokens": self.grid_hw[0] * self.grid_hw[1] + 1,
        }


class EaglePUDinoFieldExtractor(nn.Module):
    def __init__(
        self,
        arch: str = "vit_small",
        patch_size: int = 8,
        selected_layers: tuple[int, ...] = (3, 7, 11),
        checkpoint_key: str = "teacher",
        freeze_backbone: bool = True,
        pretrained_weights: str = "ckp/reference/dino_deitsmall8_pretrain.pth",
        use_mock_dino: bool = False,
        mock_dim: int = 384,
    ) -> None:
        super().__init__()
        self.selected_layers = tuple(selected_layers)
        self.patch_size = patch_size
        self.grid_hw = (360 // patch_size, 640 // patch_size)
        self.original_tokens = self.grid_hw[0] * self.grid_hw[1] + 1
        self.use_mock_dino = use_mock_dino
        if use_mock_dino:
            self.backbone = MockDinoField(dim=mock_dim, selected_layers=self.selected_layers, grid_hw=self.grid_hw)
            self.embed_dim = mock_dim
        else:
            if arch not in vits.__dict__:
                raise ValueError(f"Unsupported DINO arch: {arch}")
            self.backbone = vits.__dict__[arch](patch_size=patch_size, num_classes=0)
            utils.load_pretrained_weights(self.backbone, pretrained_weights, checkpoint_key, arch, patch_size)
            self.embed_dim = getattr(self.backbone, "embed_dim", 384)
        if freeze_backbone:
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor | tuple[int, int] | int]:
        if self.use_mock_dino:
            return self.backbone(images)
        if images.shape[-2:] != (360, 640):
            raise ValueError(f"EaglePUDinoFieldExtractor expects [B,3,360,640], got {tuple(images.shape)}")
        with torch.no_grad():
            x = self.backbone.prepare_tokens(images)
            patch_layers: list[torch.Tensor] = []
            cls_layers: list[torch.Tensor] = []
            for idx, blk in enumerate(self.backbone.blocks):
                x = blk(x)
                if idx in self.selected_layers:
                    y = self.backbone.norm(x)
                    cls_layers.append(y[:, 0])
                    patch_layers.append(y[:, 1:])
        if len(patch_layers) != len(self.selected_layers):
            raise RuntimeError(f"Expected {len(self.selected_layers)} selected layers, got {len(patch_layers)}")
        return {
            "patch_tokens_by_layer": torch.stack(patch_layers, dim=1),
            "cls_tokens_by_layer": torch.stack(cls_layers, dim=1),
            "grid_hw": self.grid_hw,
            "original_tokens": self.original_tokens,
        }
