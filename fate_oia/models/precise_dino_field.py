from __future__ import annotations

from typing import Any

import torch
from torch import nn

import utils as dino_utils
import vision_transformer as vits


class PRECISEDinoFieldExtractor(nn.Module):
    """Frozen official DINO ViT-S/8 field with explicit lifecycle diagnostics."""

    def __init__(
        self,
        arch: str = "vit_small",
        patch_size: int = 8,
        selected_layers: tuple[int, ...] = (3, 7, 11),
        checkpoint_key: str = "teacher",
        pretrained_weights: str = "ckp/reference/dino_deitsmall8_pretrain.pth",
        use_mock_dino: bool = False,
    ) -> None:
        super().__init__()
        self.selected_layers = tuple(int(layer) for layer in selected_layers)
        self.patch_size = int(patch_size)
        self.grid_hw = (45, 80)
        self.original_tokens = 3601
        self.dino_call_count = 0
        self.use_mock_dino = bool(use_mock_dino)
        if self.use_mock_dino:
            self.dino = nn.Conv2d(3, 384, kernel_size=patch_size, stride=patch_size, bias=False)
        else:
            self.dino = vits.__dict__[arch](patch_size=patch_size, num_classes=0)
            dino_utils.load_pretrained_weights(self.dino, pretrained_weights, checkpoint_key, arch, patch_size)
        for parameter in self.dino.parameters():
            parameter.requires_grad = False
        self.dino.eval()

    def train(self, mode: bool = True):
        """Keep the frozen DINO backbone in inference mode permanently."""
        super().train(False)
        self.dino.eval()
        return self

    def _clear_internal_references(self) -> None:
        if self.use_mock_dino:
            return
        for block in self.dino.blocks:
            attention = block.attn
            attention.attention_map = None
            attention.attn_gradients = None
            attention.input = None
            attention.v = None
            attention.vproj = None

    def _mock_forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        patch = self.dino(images).flatten(2).transpose(1, 2)
        patch = patch[:, : 3600]
        layers = torch.stack([patch for _ in self.selected_layers], dim=1)
        cls = layers.mean(dim=2)
        return layers, cls

    def forward(self, images: torch.Tensor) -> dict[str, Any]:
        if tuple(images.shape[-2:]) != (360, 640):
            raise ValueError(f"PRECISE expects 360x640 images, got {tuple(images.shape[-2:])}")
        self.dino_call_count += 1
        with torch.no_grad(), torch.autocast(device_type=images.device.type, dtype=torch.bfloat16, enabled=images.is_cuda):
            if self.use_mock_dino:
                patches, cls = self._mock_forward(images)
            else:
                tokens = self.dino.prepare_tokens(images)
                selected: list[torch.Tensor] = []
                for index, block in enumerate(self.dino.blocks, start=1):
                    tokens = block(tokens)
                    if index in self.selected_layers:
                        selected.append(self.dino.norm(tokens))
                if len(selected) != len(self.selected_layers):
                    raise RuntimeError("Failed to collect the configured DINO layers")
                cls = torch.stack([item[:, 0] for item in selected], dim=1)
                patches = torch.stack([item[:, 1:] for item in selected], dim=1)
        self._clear_internal_references()
        if patches.shape[2:] != (3600, 384):
            raise RuntimeError(f"Expected [B,3,3600,384], received {tuple(patches.shape)}")
        return {
            "patch_tokens_by_layer": patches.detach(),
            "cls_tokens_by_layer": cls.detach(),
            "grid_hw": self.grid_hw,
            "original_tokens": self.original_tokens,
            "dino_call_count": torch.tensor(self.dino_call_count, device=images.device),
        }
